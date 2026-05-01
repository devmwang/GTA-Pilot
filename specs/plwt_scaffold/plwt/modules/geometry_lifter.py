from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import PLWTConfig
from .common import (
    build_camera_rays,
    depth_bin_centers,
    topk_indices,
    gather_tokens_2d,
    gather_xy,
    assert_rank,
)


class GeometryLifter(nn.Module):
    def __init__(self, cfg: PLWTConfig):
        super().__init__()
        self.cfg = cfg
        in_ch = cfg.visual.ctx_dims[0]
        D = cfg.hidden_dim
        depth_bins = cfg.image.depth_bins
        self.depth_head = nn.Conv2d(in_ch, depth_bins, kernel_size=1)
        self.conf_head = nn.Conv2d(in_ch, 1, kernel_size=1)
        self.track_head = nn.Conv2d(in_ch, cfg.geometry.track_history * 2, kernel_size=1)
        self.feat_proj = nn.Linear(in_ch + 4 + D, D)
        self.norm = nn.LayerNorm(D)

    def forward(self, ctx_8x: torch.Tensor, cam_now: torch.Tensor, ego_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        ctx_8x:    [B, C8, H8, W8]
        cam_now:   [B, N_cam, D]
        ego_tokens:[B, N_ego, D]
        """
        assert_rank(ctx_8x, 4, "ctx_8x")
        b, c, h, w = ctx_8x.shape
        depth_logits = self.depth_head(ctx_8x)
        ray_conf = self.conf_head(ctx_8x).sigmoid()
        track_offsets = self.track_head(ctx_8x).reshape(b, self.cfg.geometry.track_history, 2, h, w)

        bins = depth_bin_centers(
            self.cfg.image.depth_min_m,
            self.cfg.image.depth_max_m,
            self.cfg.image.depth_bins,
            ctx_8x.device,
            ctx_8x.dtype,
        )
        depth_prob = F.softmax(depth_logits, dim=1)
        depth_mean = (depth_prob * bins[None, :, None, None]).sum(dim=1, keepdim=True)

        scores = ray_conf[:, 0]
        idx = topk_indices(scores, self.cfg.geometry.frustum_tokens)
        sampled_feat = gather_tokens_2d(ctx_8x, idx)
        sampled_xy = gather_xy(idx, h, w, ctx_8x.device, ctx_8x.dtype)
        sampled_depth = depth_mean.flatten(2).transpose(1, 2).gather(1, idx[..., None])
        sampled_conf = ray_conf.flatten(2).transpose(1, 2).gather(1, idx[..., None])

        fx = self.cfg.image.fx / 8.0
        fy = self.cfg.image.fy / 8.0
        cx = (self.cfg.image.padded_width / 2.0) / 8.0
        cy = (self.cfg.image.padded_height / 2.0) / 8.0
        rays = build_camera_rays(h, w, fx, fy, cx, cy, ctx_8x.device, ctx_8x.dtype).reshape(h * w, 3)
        sampled_rays = rays[idx]
        xyz = sampled_rays * sampled_depth
        cam_summary = cam_now.mean(dim=1, keepdim=True).expand(-1, xyz.shape[1], -1)
        tok_in = torch.cat([sampled_feat, xyz, sampled_conf, cam_summary], dim=-1)
        frustum_tokens = self.norm(self.feat_proj(tok_in))
        return {
            "depth_logits": depth_logits,
            "depth_mean": depth_mean,
            "ray_conf": ray_conf,
            "track_offsets": track_offsets,
            "frustum_tokens": frustum_tokens,
            "frustum_xyz": xyz,
        }
