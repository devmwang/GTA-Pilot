from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .. import build_atlas_config
from ..config import AtlasConfig
from ..model import Atlas
from ..teacher.privileged_teacher_model import PrivilegedTeacherModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def build_model_config(model_variant: str) -> AtlasConfig:
    return build_atlas_config(model_variant)


def build_model_from_variant(model_variant: str) -> Atlas:
    cfg = build_model_config(model_variant)
    if cfg.enable_privileged_teacher_adapters:
        return PrivilegedTeacherModel(cfg)
    return Atlas(cfg)


def batch_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=device.type == "cuda")
    if isinstance(value, dict):
        return {key: batch_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [batch_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(batch_to_device(item, device) for item in value)
    return value


def batch_to_device_except(
    value: Any,
    device: torch.device,
    *,
    skip_top_level_keys: set[str] | None = None,
    _depth: int = 0,
) -> Any:
    skip_top_level_keys = skip_top_level_keys or set()
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=device.type == "cuda")
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if _depth == 0 and key in skip_top_level_keys:
                out[key] = item
            else:
                out[key] = batch_to_device_except(
                    item,
                    device,
                    skip_top_level_keys=skip_top_level_keys,
                    _depth=_depth + 1,
                )
        return out
    if isinstance(value, list):
        return [
            batch_to_device_except(
                item,
                device,
                skip_top_level_keys=skip_top_level_keys,
                _depth=_depth + 1,
            )
            for item in value
        ]
    if isinstance(value, tuple):
        return tuple(
            batch_to_device_except(
                item,
                device,
                skip_top_level_keys=skip_top_level_keys,
                _depth=_depth + 1,
            )
            for item in value
        )
    return value


def detach_state(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach()
    if hasattr(value, "__dataclass_fields__"):
        fields = {
            key: detach_state(getattr(value, key))
            for key in value.__dataclass_fields__.keys()
        }
        return type(value)(**fields)
    if isinstance(value, dict):
        return {key: detach_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [detach_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(detach_state(item) for item in value)
    return value


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out
