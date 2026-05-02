from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .config import AtlasCruiseConfig
from .ui import UIState, build_ui_mask, neutralize_ui


@dataclass(slots=True)
class CruisePreprocessOutput:
    scene_rgb: torch.Tensor
    ui_mask: torch.Tensor
    ui_state: UIState


def _resize_chw(x: torch.Tensor, h: int, w: int, *, mode: str) -> torch.Tensor:
    return F.interpolate(
        x.unsqueeze(0),
        size=(h, w),
        mode=mode,
        align_corners=False if mode in {"bilinear", "bicubic"} else None,
    ).squeeze(0)


def preprocess_cruise_frame(
    raw_rgb: torch.Tensor,
    cfg: AtlasCruiseConfig,
    *,
    apply_ui_neutralization: bool = True,
) -> CruisePreprocessOutput:
    ui_mask_source, ui_state = build_ui_mask(raw_rgb, cfg)
    scene_source = neutralize_ui(raw_rgb, ui_mask_source) if apply_ui_neutralization and cfg.use_ui_mask else raw_rgb
    if scene_source.ndim == 3 and scene_source.shape[-1] == 3:
        scene_source = scene_source.permute(2, 0, 1).contiguous()
    if scene_source.dtype == torch.uint8:
        scene_source = scene_source.float().mul_(1.0 / 255.0)
    elif not torch.is_floating_point(scene_source):
        scene_source = scene_source.float()
    scene_rgb = _resize_chw(scene_source.clamp(0.0, 1.0), cfg.input_h, cfg.input_w, mode="bilinear")
    ui_mask = _resize_chw(ui_mask_source, cfg.input_h, cfg.input_w, mode="nearest").clamp(0.0, 1.0)
    return CruisePreprocessOutput(
        scene_rgb=scene_rgb.contiguous(),
        ui_mask=ui_mask.contiguous(),
        ui_state=ui_state,
    )


def preprocess_frame_sequence(
    frames: list[torch.Tensor],
    cfg: AtlasCruiseConfig,
) -> tuple[torch.Tensor, torch.Tensor, list[UIState]]:
    outputs = [preprocess_cruise_frame(frame, cfg) for frame in frames]
    return (
        torch.stack([out.scene_rgb for out in outputs], dim=0),
        torch.stack([out.ui_mask for out in outputs], dim=0),
        [out.ui_state for out in outputs],
    )
