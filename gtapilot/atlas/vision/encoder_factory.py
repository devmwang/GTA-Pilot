from __future__ import annotations

import torch.nn as nn

from ..config import AtlasConfig
from .dualpath_hvt_bifpn import DualPathHVTBiFPN
from .regnet_bifpn_baseline import RegNetBiFPNBaseline

VISION_ENCODER_TYPES = {
    "dualpath_hvt_bifpn": DualPathHVTBiFPN,
    "regnet_bifpn_baseline": RegNetBiFPNBaseline,
}

_VISION_STAGE_PATHS: dict[str, tuple[tuple[str, ...], ...]] = {
    "dualpath_hvt_bifpn": (
        ("stem",),
        ("detail_path",),
        ("context_path.stage8_down", "context_path.stage8"),
        ("context_path.stage16_down", "context_path.stage16"),
        ("context_path.stage32_down", "context_path.stage32"),
        ("context_path.stage64_down", "context_path.stage64"),
        ("neck",),
        ("token_pool",),
    ),
    "regnet_bifpn_baseline": (
        ("stem",),
        ("detail_path",),
        ("stage8",),
        ("stage16",),
        ("stage32",),
        ("stage64",),
        ("neck",),
        ("token_pool",),
    ),
}


def _resolve_submodule(root: nn.Module, dotted_path: str) -> nn.Module:
    module: nn.Module = root
    for part in dotted_path.split("."):
        module = getattr(module, part)
    return module


def _freeze_module(module: nn.Module) -> None:
    setattr(module, "_atlas_frozen_stage", True)
    module.eval()
    for param in module.parameters():
        param.requires_grad = False


def _restore_frozen_eval(module: nn.Module) -> None:
    for child in module.modules():
        if getattr(child, "_atlas_frozen_stage", False):
            child.eval()


def _patch_train_to_keep_frozen_eval(encoder: nn.Module) -> None:
    if getattr(encoder, "_atlas_keep_frozen_eval_patched", False):
        return
    original_train = encoder.train

    def _train(mode: bool = True):
        result = original_train(mode)
        if mode:
            _restore_frozen_eval(encoder)
        return result

    encoder.train = _train  # type: ignore[method-assign]
    setattr(encoder, "_atlas_keep_frozen_eval_patched", True)


def _apply_freeze_stages(
    encoder: nn.Module,
    *,
    encoder_type: str,
    freeze_stages: int,
) -> None:
    if freeze_stages <= 0:
        return
    try:
        stage_paths = _VISION_STAGE_PATHS[encoder_type]
    except KeyError as exc:
        raise ValueError(
            f"freeze_stages is not supported for Atlas vision encoder '{encoder_type}'."
        ) from exc
    _patch_train_to_keep_frozen_eval(encoder)
    for stage_paths_group in stage_paths[:freeze_stages]:
        for dotted_path in stage_paths_group:
            _freeze_module(_resolve_submodule(encoder, dotted_path))


def build_camera_encoder(cfg: AtlasConfig) -> nn.Module:
    encoder_type = cfg.vision.encoder_type.lower()
    try:
        encoder_cls = VISION_ENCODER_TYPES[encoder_type]
    except KeyError as exc:
        available = ", ".join(sorted(VISION_ENCODER_TYPES))
        raise ValueError(
            f"Unknown Atlas vision encoder '{cfg.vision.encoder_type}'. "
            f"Available: {available}"
        ) from exc
    encoder = encoder_cls(cfg)
    _apply_freeze_stages(
        encoder,
        encoder_type=encoder_type,
        freeze_stages=int(cfg.vision.freeze_stages),
    )
    return encoder
