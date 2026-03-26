from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from ..config import AtlasConfig
from ..utils import pad_image_to_size
from .bifpn_neck import WeightedBiFPNNeck
from .context_path import HierarchicalWindowContextPath
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


class DualPathHVTBiFPN(nn.Module):
    def __init__(self, cfg: AtlasConfig):
        super().__init__()
        if cfg.hidden_dim != cfg.vision.token_dim:
            raise ValueError("Atlas currently requires vision.token_dim == hidden_dim")
        self.cfg = cfg
        self.stem = SharedVisionStem(cfg)
        self.detail_path = VisionDetailPath(cfg)
        self.context_path = HierarchicalWindowContextPath(cfg)
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
        context = self.context_path(stem_4x)
        fused = self.neck(
            detail_8x,
            context["ctx_8x"],
            context["ctx_16x"],
            context["ctx_32x"],
            context["ctx_64x"],
        )
        cam_tokens = self.token_pool(detail_4x, fused)
        features = {
            "detail_4x": detail_4x,
            "detail_8x": detail_8x,
            **context,
            **fused,
        }
        return _restore_features(features, cam_tokens, batch_shape)
