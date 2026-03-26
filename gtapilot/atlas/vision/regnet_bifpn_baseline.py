from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from ..config import AtlasConfig
from ..utils import LayerNorm2d, pad_image_to_size
from .bifpn_neck import WeightedBiFPNNeck
from .detail_path import VisionDetailPath
from .shared_stem import SharedVisionStem
from .token_pool import MultiscaleSceneTokenPool


def _flatten_frames(rgb: torch.Tensor) -> tuple[torch.Tensor, Tuple[int, ...]]:
    if rgb.ndim == 4:
        return rgb, ()
    if rgb.ndim == 5:
        b, t, c, h, w = rgb.shape
        return rgb.reshape(b * t, c, h, w), (b, t)
    raise ValueError(f"rgb must be rank 4 or 5, got shape {tuple(rgb.shape)}")


def _restore_features(
    features: Dict[str, torch.Tensor],
    cam_tokens: torch.Tensor,
    batch_shape: Tuple[int, ...],
) -> tuple[Dict[str, torch.Tensor], torch.Tensor]:
    if not batch_shape:
        return features, cam_tokens
    b, t = batch_shape
    restored = {
        name: tensor.reshape(b, t, tensor.shape[1], tensor.shape[2], tensor.shape[3])
        for name, tensor in features.items()
    }
    return restored, cam_tokens.reshape(b, t, cam_tokens.shape[1], cam_tokens.shape[2])


class RegNetStyleBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int, dropout: float):
        super().__init__()
        mid_ch = max(out_ch // 2, 8)
        groups = max(1, mid_ch // 16)
        self.norm1 = LayerNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=1)
        self.norm2 = LayerNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(
            mid_ch,
            mid_ch,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=groups,
        )
        self.norm3 = LayerNorm2d(mid_ch)
        self.conv3 = nn.Conv2d(mid_ch, out_ch, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout)
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.conv1(torch.nn.functional.gelu(self.norm1(x)))
        x = self.conv2(torch.nn.functional.gelu(self.norm2(x)))
        x = self.dropout(x)
        x = self.conv3(torch.nn.functional.gelu(self.norm3(x)))
        return x + residual


class RegNetStage(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, blocks: int, dropout: float):
        super().__init__()
        layers = [RegNetStyleBlock(in_ch, out_ch, stride=2, dropout=dropout)]
        layers.extend(
            RegNetStyleBlock(out_ch, out_ch, stride=1, dropout=dropout)
            for _ in range(max(0, blocks - 1))
        )
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RegNetBiFPNBaseline(nn.Module):
    def __init__(self, cfg: AtlasConfig):
        super().__init__()
        if cfg.hidden_dim != cfg.vision.token_dim:
            raise ValueError("Atlas currently requires vision.token_dim == hidden_dim")
        self.cfg = cfg
        vc = cfg.vision
        d8, d16, d32, d64 = vc.ctx_dims
        b8, b16, b32, b64 = vc.ctx_blocks
        self.stem = SharedVisionStem(cfg)
        self.detail_path = VisionDetailPath(cfg)
        self.stage8 = RegNetStage(vc.detail_dim, d8, b8, vc.dropout)
        self.stage16 = RegNetStage(d8, d16, b16, vc.dropout)
        self.stage32 = RegNetStage(d16, d32, b32, vc.dropout)
        self.stage64 = RegNetStage(d32, d64, b64, vc.dropout)
        self.neck = WeightedBiFPNNeck(cfg)
        self.token_pool = MultiscaleSceneTokenPool(cfg)

    def forward(
        self,
        rgb: torch.Tensor,
        pad_to_native: bool = True,
    ) -> tuple[Dict[str, torch.Tensor], torch.Tensor]:
        flat_rgb, batch_shape = _flatten_frames(rgb)
        if pad_to_native:
            flat_rgb = pad_image_to_size(
                flat_rgb,
                self.cfg.image.padded_height,
                self.cfg.image.padded_width,
            )

        stem_4x = self.stem(flat_rgb)
        detail_4x, detail_8x = self.detail_path(stem_4x)
        ctx_8x = self.stage8(stem_4x)
        ctx_16x = self.stage16(ctx_8x)
        ctx_32x = self.stage32(ctx_16x)
        ctx_64x = self.stage64(ctx_32x)
        fused = self.neck(detail_8x, ctx_8x, ctx_16x, ctx_32x, ctx_64x)
        cam_tokens = self.token_pool(detail_4x, fused)
        features = {
            "detail_4x": detail_4x,
            "detail_8x": detail_8x,
            "ctx_8x": ctx_8x,
            "ctx_16x": ctx_16x,
            "ctx_32x": ctx_32x,
            "ctx_64x": ctx_64x,
            **fused,
        }
        return _restore_features(features, cam_tokens, batch_shape)
