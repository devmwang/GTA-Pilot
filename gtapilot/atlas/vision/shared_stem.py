from __future__ import annotations

import torch
import torch.nn as nn

from ..config import AtlasConfig
from ..utils import ResidualConvBlock


class SharedVisionStem(nn.Module):
    def __init__(self, cfg: AtlasConfig):
        super().__init__()
        vc = cfg.vision
        self.net = nn.Sequential(
            nn.Conv2d(
                cfg.image.channels,
                vc.stem_hidden_dim,
                kernel_size=7,
                stride=2,
                padding=3,
            ),
            nn.GELU(),
            nn.Conv2d(
                vc.stem_hidden_dim,
                vc.detail_dim,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.GELU(),
            ResidualConvBlock(vc.detail_dim, dropout=vc.dropout),
            ResidualConvBlock(vc.detail_dim, dropout=vc.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
