from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import AtlasHAConfig


def _group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )


class SeparableConvBlock(nn.Sequential):
    def __init__(self, channels: int):
        super().__init__(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(channels), channels),
            nn.SiLU(inplace=True),
        )


class BiFPNLayer(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.td_p4 = SeparableConvBlock(channels)
        self.td_p3 = SeparableConvBlock(channels)
        self.td_p2 = SeparableConvBlock(channels)
        self.bu_p3 = SeparableConvBlock(channels)
        self.bu_p4 = SeparableConvBlock(channels)
        self.bu_p5 = SeparableConvBlock(channels)

    @staticmethod
    def _upsample_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=ref.shape[-2:], mode="nearest")

    def forward(
        self,
        p2: torch.Tensor,
        p3: torch.Tensor,
        p4: torch.Tensor,
        p5: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        p4_td = self.td_p4(p4 + self._upsample_like(p5, p4))
        p3_td = self.td_p3(p3 + self._upsample_like(p4_td, p3))
        p2_out = self.td_p2(p2 + self._upsample_like(p3_td, p2))
        p3_out = self.bu_p3(p3_td + F.max_pool2d(p2_out, kernel_size=2, stride=2))
        p4_out = self.bu_p4(p4_td + F.max_pool2d(p3_out, kernel_size=2, stride=2))
        p5_out = self.bu_p5(p5 + F.max_pool2d(p4_out, kernel_size=2, stride=2))
        return p2_out, p3_out, p4_out, p5_out


class HighwayBiFPN(nn.Module):
    def __init__(self, cfg: AtlasHAConfig, in_channels: tuple[int, int, int, int]):
        super().__init__()
        self.lateral = nn.ModuleList(
            [ConvNormAct(in_ch, cfg.fpn_dim, kernel_size=1) for in_ch in in_channels]
        )
        self.layers = nn.ModuleList(
            [BiFPNLayer(cfg.fpn_dim) for _ in range(cfg.bifpn_repeats)]
        )

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        p2, p3, p4, p5 = [
            lateral(features[name])
            for lateral, name in zip(self.lateral, ("c2", "c3", "c4", "c5"))
        ]
        for layer in self.layers:
            p2, p3, p4, p5 = layer(p2, p3, p4, p5)
        return {"p2": p2, "p3": p3, "p4": p4, "p5": p5}
