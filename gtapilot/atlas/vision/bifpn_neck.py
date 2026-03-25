from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import AtlasConfig
from ..utils import ResidualConvBlock


class WeightedFeatureFusion(nn.Module):
    def __init__(self, channels: int, num_inputs: int, dropout: float):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_inputs))
        self.post = ResidualConvBlock(channels, dropout=dropout)

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        if len(inputs) != self.weights.numel():
            raise ValueError("fusion input count does not match configured weights")
        weights = F.relu(self.weights)
        weights = weights / (weights.sum() + 1e-6)
        fused = sum(weight * tensor for weight, tensor in zip(weights, inputs, strict=True))
        return self.post(fused)


class BiFPNRepeat(nn.Module):
    def __init__(self, channels: int, dropout: float):
        super().__init__()
        self.top32 = WeightedFeatureFusion(channels, 2, dropout)
        self.top16 = WeightedFeatureFusion(channels, 2, dropout)
        self.top8 = WeightedFeatureFusion(channels, 2, dropout)
        self.out16 = WeightedFeatureFusion(channels, 3, dropout)
        self.out32 = WeightedFeatureFusion(channels, 3, dropout)
        self.out64 = WeightedFeatureFusion(channels, 3, dropout)

    def forward(
        self,
        p8: torch.Tensor,
        p16: torch.Tensor,
        p32: torch.Tensor,
        p64: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        td64 = p64
        td32 = self.top32(
            [p32, F.interpolate(td64, size=p32.shape[-2:], mode="nearest")]
        )
        td16 = self.top16(
            [p16, F.interpolate(td32, size=p16.shape[-2:], mode="nearest")]
        )
        td8 = self.top8(
            [p8, F.interpolate(td16, size=p8.shape[-2:], mode="nearest")]
        )

        out8 = td8
        out16 = self.out16(
            [p16, td16, F.avg_pool2d(out8, kernel_size=2, stride=2)]
        )
        out32 = self.out32(
            [p32, td32, F.avg_pool2d(out16, kernel_size=2, stride=2)]
        )
        out64 = self.out64(
            [p64, td64, F.avg_pool2d(out32, kernel_size=2, stride=2)]
        )
        return out8, out16, out32, out64


class WeightedBiFPNNeck(nn.Module):
    def __init__(self, cfg: AtlasConfig):
        super().__init__()
        vc = cfg.vision
        C = vc.fusion_dim
        self.detail8_proj = nn.Conv2d(vc.detail_8x_dim, C, kernel_size=1)
        self.ctx8_proj = nn.Conv2d(vc.ctx_dims[0], C, kernel_size=1)
        self.ctx16_proj = nn.Conv2d(vc.ctx_dims[1], C, kernel_size=1)
        self.ctx32_proj = nn.Conv2d(vc.ctx_dims[2], C, kernel_size=1)
        self.ctx64_proj = nn.Conv2d(vc.ctx_dims[3], C, kernel_size=1)
        self.p8_input = WeightedFeatureFusion(C, 2, dropout=vc.dropout)
        self.repeats = nn.ModuleList(
            [BiFPNRepeat(C, dropout=vc.dropout) for _ in range(vc.bifpn_repeats)]
        )

    def forward(
        self,
        detail_8x: torch.Tensor,
        ctx_8x: torch.Tensor,
        ctx_16x: torch.Tensor,
        ctx_32x: torch.Tensor,
        ctx_64x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        p8 = self.p8_input([self.detail8_proj(detail_8x), self.ctx8_proj(ctx_8x)])
        p16 = self.ctx16_proj(ctx_16x)
        p32 = self.ctx32_proj(ctx_32x)
        p64 = self.ctx64_proj(ctx_64x)

        for bifpn in self.repeats:
            p8, p16, p32, p64 = bifpn(p8, p16, p32, p64)

        return {
            "f8": p8,
            "f16": p16,
            "f32": p32,
            "f64": p64,
        }
