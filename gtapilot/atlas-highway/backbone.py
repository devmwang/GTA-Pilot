from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn
import torchvision.models as tv_models

from .config import AtlasHAConfig


REGNET_STAGE_CHANNELS: Mapping[str, tuple[int, int, int, int]] = {
    "regnet_y_800mf": (64, 144, 320, 784),
    "regnet_y_1_6gf": (48, 120, 336, 888),
}

_WEIGHT_ENUM_NAMES: Mapping[str, str] = {
    "regnet_y_800mf": "RegNet_Y_800MF_Weights",
    "regnet_y_1_6gf": "RegNet_Y_1_6GF_Weights",
}


def _resolve_regnet_weights(backbone_name: str, pretrained: bool):
    if not pretrained:
        return None
    enum_name = _WEIGHT_ENUM_NAMES.get(backbone_name)
    if enum_name is None:
        raise ValueError(f"No pretrained weight enum is registered for {backbone_name!r}.")
    weights_enum = getattr(tv_models, enum_name)
    return weights_enum.DEFAULT


class RegNetBackbone(nn.Module):
    def __init__(self, cfg: AtlasHAConfig):
        super().__init__()
        if cfg.backbone_name not in REGNET_STAGE_CHANNELS:
            raise ValueError(
                f"Unsupported Atlas-HA backbone {cfg.backbone_name!r}. "
                f"Expected one of {sorted(REGNET_STAGE_CHANNELS)}."
            )
        model_fn = getattr(tv_models, cfg.backbone_name)
        weights = _resolve_regnet_weights(cfg.backbone_name, cfg.pretrained_backbone)
        model = model_fn(weights=weights)
        self.stem = model.stem
        self.stages = model.trunk_output
        self.out_channels = REGNET_STAGE_CHANNELS[cfg.backbone_name]
        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, scene_rgb: torch.Tensor) -> dict[str, torch.Tensor]:
        if scene_rgb.ndim != 4 or scene_rgb.shape[1] != 3:
            raise ValueError("scene_rgb must have shape [B, 3, H, W].")
        x = scene_rgb
        if x.dtype == torch.uint8:
            x = x.float().mul_(1.0 / 255.0)
        elif not torch.is_floating_point(x):
            x = x.float()
        x = (x - self.imagenet_mean.to(dtype=x.dtype)) / self.imagenet_std.to(dtype=x.dtype)
        x = self.stem(x)
        c2 = self.stages.block1(x)
        c3 = self.stages.block2(c2)
        c4 = self.stages.block3(c3)
        c5 = self.stages.block4(c4)
        return {"c2": c2, "c3": c3, "c4": c4, "c5": c5}


def build_regnet_backbone(cfg: AtlasHAConfig) -> RegNetBackbone:
    return RegNetBackbone(cfg)
