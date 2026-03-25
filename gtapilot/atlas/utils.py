from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def assert_rank(x: torch.Tensor, rank: int, name: str) -> None:
    if x.ndim != rank:
        raise ValueError(f"{name} must have rank {rank}, got shape {tuple(x.shape)}")


def assert_last_dim(x: torch.Tensor, dim: int, name: str) -> None:
    if x.shape[-1] != dim:
        raise ValueError(f"{name} last dim must be {dim}, got shape {tuple(x.shape)}")


def pad_image_to_size(x: torch.Tensor, padded_height: int, padded_width: int) -> torch.Tensor:
    assert_rank(x, 4, "rgb")
    _, _, h, w = x.shape
    if h > padded_height or w > padded_width:
        raise ValueError(f"Input image {(h, w)} larger than padded size {(padded_height, padded_width)}")
    pad_h = padded_height - h
    pad_w = padded_width - w
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    return F.pad(x, (left, right, top, bottom))


def flatten_hw(x: torch.Tensor) -> torch.Tensor:
    assert_rank(x, 4, "feature map")
    b, c, h, w = x.shape
    return x.flatten(2).transpose(1, 2).contiguous()


def unflatten_hw(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    assert_rank(x, 3, "tokens")
    b, n, c = x.shape
    if n != h * w:
        raise ValueError(f"Cannot unflatten {n} tokens into {(h, w)}")
    return x.transpose(1, 2).reshape(b, c, h, w).contiguous()


def sinusoidal_embedding(values: torch.Tensor, dim: int) -> torch.Tensor:
    if dim % 2 != 0:
        raise ValueError("sinusoidal embedding dim must be even")
    device = values.device
    half = dim // 2
    freqs = torch.exp(torch.arange(half, device=device, dtype=values.dtype) * (-math.log(10000.0) / max(half - 1, 1)))
    args = values * freqs
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + 1e-6)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = LayerNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(F.gelu(self.norm1(x)))
        y = self.dropout(y)
        y = self.conv2(F.gelu(self.norm2(y)))
        return x + y


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, out_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        out_dim = out_dim or dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LearnedQueryPool(nn.Module):
    def __init__(self, num_queries: int, dim: int, heads: int, dropout: float = 0.0):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, num_queries, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dim * 4, dim, dropout)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        assert_rank(src, 3, "src")
        b = src.shape[0]
        q = self.queries.expand(b, -1, -1)
        out, _ = self.attn(self.norm_q(q), self.norm_kv(src), self.norm_kv(src), need_weights=False)
        out = out + self.mlp(out)
        return out


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dim * 4, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dim * 4, dropout=dropout)

    def forward(self, x: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        y, _ = self.attn(self.norm_q(x), self.norm_kv(src), self.norm_kv(src), need_weights=False)
        x = x + y
        x = x + self.mlp(self.norm_mlp(x))
        return x


class WorldUpdateBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.0):
        super().__init__()
        self.self_block = SelfAttentionBlock(dim, heads, dropout)
        self.cross_block = CrossAttentionBlock(dim, heads, dropout)

    def forward(self, x: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        x = self.self_block(x)
        x = self.cross_block(x, obs)
        return x


def depth_bin_centers(min_depth: float, max_depth: float, bins: int, device, dtype) -> torch.Tensor:
    return torch.linspace(min_depth, max_depth, steps=bins, device=device, dtype=dtype)


def build_camera_rays(
    h: int,
    w: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    ys, xs = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing="ij",
    )
    xs = (xs - cx) / fx
    ys = (ys - cy) / fy
    z = torch.ones_like(xs)
    rays = torch.stack([xs, ys, z], dim=-1)
    rays = rays / torch.clamp(torch.linalg.norm(rays, dim=-1, keepdim=True), min=1e-6)
    return rays


def topk_indices(score: torch.Tensor, k: int) -> torch.Tensor:
    assert_rank(score, 3, "score")
    b, h, w = score.shape
    flat = score.reshape(b, h * w)
    k = min(k, h * w)
    idx = flat.topk(k, dim=-1).indices
    return idx


def gather_tokens_2d(feat: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    assert_rank(feat, 4, "feat")
    b, c, h, w = feat.shape
    flat = feat.flatten(2).transpose(1, 2)  # [B, HW, C]
    idx = indices[..., None].expand(-1, -1, c)
    return flat.gather(1, idx)


def gather_xy(indices: torch.Tensor, h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    y = torch.div(indices, w, rounding_mode='floor')
    x = indices % w
    return torch.stack([x.to(dtype), y.to(dtype)], dim=-1)


def pose_to_affine(
    pose_delta: torch.Tensor,
    grid_h: int,
    grid_w: int,
    cell_x_m: float,
    cell_y_m: float,
) -> torch.Tensor:
    # pose_delta: [B, 3] -> dx, dy, dyaw
    assert_rank(pose_delta, 2, "pose_delta")
    dx = pose_delta[:, 0]
    dy = pose_delta[:, 1]
    dyaw = pose_delta[:, 2]
    cos_t = torch.cos(dyaw)
    sin_t = torch.sin(dyaw)
    height_m = grid_h * cell_x_m
    width_m = grid_w * cell_y_m
    tx = dy / max(width_m / 2.0, 1e-6)
    ty = -dx / max(height_m / 2.0, 1e-6)
    theta = torch.zeros(pose_delta.shape[0], 2, 3, device=pose_delta.device, dtype=pose_delta.dtype)
    theta[:, 0, 0] = cos_t
    theta[:, 0, 1] = -sin_t
    theta[:, 1, 0] = sin_t
    theta[:, 1, 1] = cos_t
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty
    return theta


def warp_bev(
    grid: torch.Tensor,
    pose_delta: torch.Tensor,
    cell_x_m: float,
    cell_y_m: float,
) -> torch.Tensor:
    assert_rank(grid, 4, "static_grid")
    b, c, h, w = grid.shape
    theta = pose_to_affine(pose_delta, h, w, cell_x_m, cell_y_m)
    affine_grid = F.affine_grid(theta, size=grid.shape, align_corners=False)
    return F.grid_sample(grid, affine_grid, mode="bilinear", padding_mode="zeros", align_corners=False)


class KinematicIntegrator(nn.Module):
    def __init__(self, control_dt: float, waypoint_dt: float, control_steps: int):
        super().__init__()
        self.control_dt = control_dt
        self.waypoint_dt = waypoint_dt
        self.control_steps = control_steps

    def forward(self, curvature: torch.Tensor, speed: torch.Tensor, init_speed: torch.Tensor) -> torch.Tensor:
        # curvature, speed: [B, K, S], init_speed: [B]
        assert_rank(curvature, 3, "curvature")
        b, k, s = curvature.shape
        if speed.shape != curvature.shape:
            raise ValueError("speed must match curvature shape")
        repeat_factor = max(1, int(round(self.control_dt / self.waypoint_dt)))
        dt = self.control_dt / repeat_factor
        x = torch.zeros(b, k, 1, device=curvature.device, dtype=curvature.dtype)
        y = torch.zeros_like(x)
        yaw = torch.zeros_like(x)
        v = init_speed[:, None, None].expand(b, k, 1)
        xs, ys, yaws, vs = [], [], [], []
        for i in range(s):
            curv_step = curvature[:, :, i : i + 1]
            speed_step = speed[:, :, i : i + 1]
            for _ in range(repeat_factor):
                v = speed_step
                yaw = yaw + v * curv_step * dt
                x = x + v * torch.cos(yaw) * dt
                y = y + v * torch.sin(yaw) * dt
                xs.append(x)
                ys.append(y)
                yaws.append(yaw)
                vs.append(v)
        x = torch.cat(xs, dim=2)
        y = torch.cat(ys, dim=2)
        yaw = torch.cat(yaws, dim=2)
        v = torch.cat(vs, dim=2)
        traj = torch.stack([x, y, yaw, v], dim=-1)
        return traj
