from __future__ import annotations

import torch
import torch.nn as nn

from ..config import AtlasConfig
from ..utils import ResidualConvBlock


class VisionDetailPath(nn.Module):
    def __init__(self, cfg: AtlasConfig):
        super().__init__()
        vc = cfg.vision
        self.blocks = nn.Sequential(
            *[
                ResidualConvBlock(vc.detail_dim, dropout=vc.dropout)
                for _ in range(vc.detail_blocks)
            ]
        )
        self.downsample = nn.Conv2d(
            vc.detail_dim,
            vc.detail_8x_dim,
            kernel_size=3,
            stride=2,
            padding=1,
        )

    def forward(self, stem_4x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        detail_4x = self.blocks(stem_4x)
        detail_8x = self.downsample(detail_4x)
        return detail_4x, detail_8x
