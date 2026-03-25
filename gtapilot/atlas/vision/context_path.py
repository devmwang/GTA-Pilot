from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import AtlasConfig
from ..utils import MLP
from .positional import RelativePositionBias


def _window_partition(x: torch.Tensor, window_size: int) -> tuple[torch.Tensor, int, int]:
    b, h, w, c = x.shape
    pad_h = (window_size - h % window_size) % window_size
    pad_w = (window_size - w % window_size) % window_size
    if pad_h or pad_w:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    hp, wp = x.shape[1], x.shape[2]
    windows = x.view(
        b,
        hp // window_size,
        window_size,
        wp // window_size,
        window_size,
        c,
    )
    windows = windows.permute(0, 1, 3, 2, 4, 5).reshape(
        -1, window_size * window_size, c
    )
    return windows, hp, wp


def _window_reverse(
    windows: torch.Tensor,
    window_size: int,
    batch_size: int,
    padded_height: int,
    padded_width: int,
    out_height: int,
    out_width: int,
) -> torch.Tensor:
    c = windows.shape[-1]
    x = windows.view(
        batch_size,
        padded_height // window_size,
        padded_width // window_size,
        window_size,
        window_size,
        c,
    )
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(batch_size, padded_height, padded_width, c)
    return x[:, :out_height, :out_width]


class WindowedTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        dropout: float,
        shift: bool,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = window_size // 2 if shift else 0
        self.scale = (dim // num_heads) ** -0.5
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * 4), dim, dropout=dropout)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        self.relative_bias = RelativePositionBias(window_size, num_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        if self.shift_size > 0:
            x = torch.roll(
                x,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2),
            )

        windows, hp, wp = _window_partition(x, self.window_size)
        residual = windows
        tokens = self.norm1(windows)

        qkv = self.qkv(tokens).reshape(
            tokens.shape[0],
            tokens.shape[1],
            3,
            self.num_heads,
            self.dim // self.num_heads,
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn + self.relative_bias().unsqueeze(0)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(tokens.shape[0], tokens.shape[1], self.dim)
        out = self.proj_drop(self.proj(out))
        out = residual + out
        out = out + self.mlp(self.norm2(out))

        x = _window_reverse(out, self.window_size, b, hp, wp, h, w)
        if self.shift_size > 0:
            x = torch.roll(
                x,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2),
            )
        return x.permute(0, 3, 1, 2).contiguous()


class PatchMergingDownsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class HierarchicalWindowContextPath(nn.Module):
    def __init__(self, cfg: AtlasConfig):
        super().__init__()
        vc = cfg.vision
        dims = vc.ctx_dims
        blocks = vc.ctx_blocks
        heads = vc.ctx_heads
        windows = vc.ctx_window_sizes
        self.stage8_down = PatchMergingDownsample(vc.detail_dim, dims[0])
        self.stage16_down = PatchMergingDownsample(dims[0], dims[1])
        self.stage32_down = PatchMergingDownsample(dims[1], dims[2])
        self.stage64_down = PatchMergingDownsample(dims[2], dims[3])
        self.stage8 = nn.Sequential(
            *[
                WindowedTransformerBlock(
                    dims[0],
                    heads[0],
                    windows[0],
                    vc.dropout,
                    shift=bool(index % 2),
                )
                for index in range(blocks[0])
            ]
        )
        self.stage16 = nn.Sequential(
            *[
                WindowedTransformerBlock(
                    dims[1],
                    heads[1],
                    windows[1],
                    vc.dropout,
                    shift=bool(index % 2),
                )
                for index in range(blocks[1])
            ]
        )
        self.stage32 = nn.Sequential(
            *[
                WindowedTransformerBlock(
                    dims[2],
                    heads[2],
                    windows[2],
                    vc.dropout,
                    shift=bool(index % 2),
                )
                for index in range(blocks[2])
            ]
        )
        self.stage64 = nn.Sequential(
            *[
                WindowedTransformerBlock(
                    dims[3],
                    heads[3],
                    windows[3],
                    vc.dropout,
                    shift=bool(index % 2),
                )
                for index in range(blocks[3])
            ]
        )

    def forward(self, stem_4x: torch.Tensor) -> Dict[str, torch.Tensor]:
        ctx_8x = self.stage8(self.stage8_down(stem_4x))
        ctx_16x = self.stage16(self.stage16_down(ctx_8x))
        ctx_32x = self.stage32(self.stage32_down(ctx_16x))
        ctx_64x = self.stage64(self.stage64_down(ctx_32x))
        return {
            "ctx_8x": ctx_8x,
            "ctx_16x": ctx_16x,
            "ctx_32x": ctx_32x,
            "ctx_64x": ctx_64x,
        }
