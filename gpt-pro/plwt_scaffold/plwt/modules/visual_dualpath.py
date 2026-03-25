from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import PLWTConfig
from .common import ResidualConvBlock, LearnedQueryPool, flatten_hw, assert_rank


class DownsampleStage(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, blocks: int, dropout: float):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)
        self.blocks = nn.Sequential(*[ResidualConvBlock(out_ch, dropout=dropout) for _ in range(blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return self.blocks(x)


class VisualDualPathTokenizer(nn.Module):
    """
    Lightweight scaffold implementation for the 1080p visual tokenizer.

    The config and tensor interfaces match the spec. The internal blocks are deliberately
    compact and runnable; replace stage internals with stronger windowed-attention stages
    as you harden the model.
    """

    def __init__(self, cfg: PLWTConfig):
        super().__init__()
        vc = cfg.visual
        self.cfg = cfg
        self.stem = nn.Sequential(
            nn.Conv2d(cfg.image.channels, vc.detail_dim // 2, kernel_size=7, stride=2, padding=3),
            nn.GELU(),
            nn.Conv2d(vc.detail_dim // 2, vc.detail_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            ResidualConvBlock(vc.detail_dim, dropout=vc.dropout),
        )

        d8, d16, d32, d64 = vc.ctx_dims
        b8, b16, b32, b64 = vc.ctx_blocks
        self.stage8 = DownsampleStage(vc.detail_dim, d8, b8, vc.dropout)
        self.stage16 = DownsampleStage(d8, d16, b16, vc.dropout)
        self.stage32 = DownsampleStage(d16, d32, b32, vc.dropout)
        self.stage64 = DownsampleStage(d32, d64, b64, vc.dropout)

        D = cfg.hidden_dim
        self.detail_to_d = nn.Conv2d(vc.detail_dim, D, kernel_size=1)
        self.ctx16_to_d = nn.Conv2d(d16, D, kernel_size=1)
        self.ctx32_to_d = nn.Conv2d(d32, D, kernel_size=1)
        self.ctx64_to_d = nn.Conv2d(d64, D, kernel_size=1)

        self.token_pool = LearnedQueryPool(
            num_queries=vc.cam_tokens_per_frame,
            dim=D,
            heads=vc.query_pool_heads,
            dropout=vc.dropout,
        )

    def forward(self, rgb: torch.Tensor) -> tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Args:
            rgb: [B, T, 3, H, W]
        Returns:
            pyramid: dict of feature maps, each shaped [B, T, C, H_i, W_i]
            cam_tokens: [B, T, N_cam, D]
        """
        assert_rank(rgb, 5, "rgb")
        b, t, c, h, w = rgb.shape
        x = rgb.reshape(b * t, c, h, w)

        detail_4x = self.stem(x)
        ctx_8x = self.stage8(detail_4x)
        ctx_16x = self.stage16(ctx_8x)
        ctx_32x = self.stage32(ctx_16x)
        ctx_64x = self.stage64(ctx_32x)

        detail_ds = F.avg_pool2d(detail_4x, kernel_size=4, stride=4)
        detail_d = self.detail_to_d(detail_ds)
        ctx16_d = self.ctx16_to_d(ctx_16x)
        ctx32_d = F.interpolate(self.ctx32_to_d(ctx_32x), size=ctx16_d.shape[-2:], mode="bilinear", align_corners=False)
        ctx64_d = F.interpolate(self.ctx64_to_d(ctx_64x), size=ctx16_d.shape[-2:], mode="bilinear", align_corners=False)

        src = torch.cat(
            [
                flatten_hw(detail_d),
                flatten_hw(ctx16_d),
                flatten_hw(ctx32_d),
                flatten_hw(ctx64_d),
            ],
            dim=1,
        )
        cam_tokens = self.token_pool(src).reshape(b, t, -1, self.cfg.hidden_dim)

        pyramid = {
            "detail_4x": detail_4x.reshape(b, t, detail_4x.shape[1], detail_4x.shape[2], detail_4x.shape[3]),
            "ctx_8x": ctx_8x.reshape(b, t, ctx_8x.shape[1], ctx_8x.shape[2], ctx_8x.shape[3]),
            "ctx_16x": ctx_16x.reshape(b, t, ctx_16x.shape[1], ctx_16x.shape[2], ctx_16x.shape[3]),
            "ctx_32x": ctx_32x.reshape(b, t, ctx_32x.shape[1], ctx_32x.shape[2], ctx_32x.shape[3]),
            "ctx_64x": ctx_64x.reshape(b, t, ctx_64x.shape[1], ctx_64x.shape[2], ctx_64x.shape[3]),
        }
        return pyramid, cam_tokens
