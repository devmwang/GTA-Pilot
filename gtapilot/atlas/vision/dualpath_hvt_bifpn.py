from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

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

    def _use_activation_checkpoint(self, tensor: torch.Tensor) -> bool:
        return self.training and torch.is_grad_enabled() and tensor.device.type == "cuda"

    @staticmethod
    def _checkpoint(
        fn,
        *args: torch.Tensor,
    ):
        if not any(isinstance(arg, torch.Tensor) and arg.requires_grad for arg in args):
            return fn(*args)
        return checkpoint(
            fn,
            *args,
            use_reentrant=True,
        )

    def _run_context_path(
        self,
        stem_4x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        context = self.context_path(stem_4x)
        return (
            context["ctx_8x"],
            context["ctx_16x"],
            context["ctx_32x"],
            context["ctx_64x"],
        )

    def _run_neck(
        self,
        detail_8x: torch.Tensor,
        ctx_8x: torch.Tensor,
        ctx_16x: torch.Tensor,
        ctx_32x: torch.Tensor,
        ctx_64x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fused = self.neck(detail_8x, ctx_8x, ctx_16x, ctx_32x, ctx_64x)
        return fused["f8"], fused["f16"], fused["f32"], fused["f64"]

    def _run_token_pool(
        self,
        detail_4x: torch.Tensor,
        f8: torch.Tensor,
        f16: torch.Tensor,
        f32: torch.Tensor,
        f64: torch.Tensor,
    ) -> torch.Tensor:
        return self.token_pool(
            detail_4x,
            {
                "f8": f8,
                "f16": f16,
                "f32": f32,
                "f64": f64,
            },
        )

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

        checkpoint_ok = self._use_activation_checkpoint(flat_rgb)
        if checkpoint_ok:
            stem_4x = self._checkpoint(self.stem, flat_rgb)
            detail_4x, detail_8x = self._checkpoint(self.detail_path, stem_4x)
            ctx_8x, ctx_16x, ctx_32x, ctx_64x = self._checkpoint(
                self._run_context_path,
                stem_4x,
            )
            # The BiFPN neck hits a CUDA AMP checkpoint/backward bug in practice.
            # Keep checkpointing on the larger stem/detail/context paths and run
            # the neck normally to preserve GPU trainability.
            f8, f16, f32, f64 = self._run_neck(
                detail_8x,
                ctx_8x,
                ctx_16x,
                ctx_32x,
                ctx_64x,
            )
            cam_tokens = self._run_token_pool(detail_4x, f8, f16, f32, f64)
        else:
            stem_4x = self.stem(flat_rgb)
            detail_4x, detail_8x = self.detail_path(stem_4x)
            ctx_8x, ctx_16x, ctx_32x, ctx_64x = self._run_context_path(stem_4x)
            f8, f16, f32, f64 = self._run_neck(detail_8x, ctx_8x, ctx_16x, ctx_32x, ctx_64x)
            cam_tokens = self._run_token_pool(detail_4x, f8, f16, f32, f64)
        features = {
            "detail_4x": detail_4x,
            "detail_8x": detail_8x,
            "ctx_8x": ctx_8x,
            "ctx_16x": ctx_16x,
            "ctx_32x": ctx_32x,
            "ctx_64x": ctx_64x,
            "f8": f8,
            "f16": f16,
            "f32": f32,
            "f64": f64,
        }
        return _restore_features(features, cam_tokens, batch_shape)
