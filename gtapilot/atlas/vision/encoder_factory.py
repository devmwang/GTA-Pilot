from __future__ import annotations

import torch.nn as nn

from ..config import AtlasConfig
from .dualpath_hvt_bifpn import DualPathHVTBiFPN
from .regnet_bifpn_baseline import RegNetBiFPNBaseline

VISION_ENCODER_TYPES = {
    "dualpath_hvt_bifpn": DualPathHVTBiFPN,
    "regnet_bifpn_baseline": RegNetBiFPNBaseline,
}


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
    return encoder_cls(cfg)
