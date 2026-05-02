from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import AtlasCruiseConfig


@dataclass(slots=True)
class UIState:
    minimap_masked: bool
    phone_present: bool
    popup_present: bool
    reticle_masked: bool
    occlusion_fraction: float

    @property
    def any_ui_occluded(self) -> bool:
        return bool(self.minimap_masked or self.phone_present or self.popup_present or self.reticle_masked)


def _as_chw_float(raw_rgb: torch.Tensor) -> torch.Tensor:
    x = raw_rgb
    if x.ndim != 3:
        raise ValueError("raw_rgb must have shape [H, W, 3] or [3, H, W].")
    if x.shape[0] == 3:
        x = x.contiguous()
    elif x.shape[-1] == 3:
        x = x.permute(2, 0, 1).contiguous()
    else:
        raise ValueError("raw_rgb must have an RGB channel dimension.")
    if x.dtype == torch.uint8:
        x = x.float().mul_(1.0 / 255.0)
    elif not torch.is_floating_point(x):
        x = x.float()
    return x.clamp(0.0, 1.0)


def _rect(mask: torch.Tensor, x0: float, y0: float, x1: float, y1: float) -> None:
    _, h, w = mask.shape
    ix0 = max(0, min(w, int(round(x0 * w))))
    ix1 = max(0, min(w, int(round(x1 * w))))
    iy0 = max(0, min(h, int(round(y0 * h))))
    iy1 = max(0, min(h, int(round(y1 * h))))
    if ix1 > ix0 and iy1 > iy0:
        mask[:, iy0:iy1, ix0:ix1] = 1.0


def _region_stats(x: torch.Tensor, x0: float, y0: float, x1: float, y1: float) -> tuple[float, float]:
    _, h, w = x.shape
    ix0 = max(0, min(w, int(round(x0 * w))))
    ix1 = max(0, min(w, int(round(x1 * w))))
    iy0 = max(0, min(h, int(round(y0 * h))))
    iy1 = max(0, min(h, int(round(y1 * h))))
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0, 0.0
    region = x[:, iy0:iy1, ix0:ix1]
    return float(region.mean().item()), float(region.std(unbiased=False).item())


def detect_phone(raw_rgb_chw: torch.Tensor) -> bool:
    mean, std = _region_stats(raw_rgb_chw, 0.72, 0.44, 1.0, 1.0)
    return bool(std > 0.10 and mean < 0.55)


def detect_top_left_popup(raw_rgb_chw: torch.Tensor) -> bool:
    mean, std = _region_stats(raw_rgb_chw, 0.0, 0.0, 0.40, 0.20)
    return bool(std > 0.08 and mean > 0.08)


def build_ui_mask(raw_rgb: torch.Tensor, cfg: AtlasCruiseConfig) -> tuple[torch.Tensor, UIState]:
    raw = _as_chw_float(raw_rgb)
    mask = torch.zeros(1, raw.shape[1], raw.shape[2], dtype=torch.float32, device=raw.device)
    phone_present = detect_phone(raw) if cfg.neutralize_phone else False
    popup_present = detect_top_left_popup(raw) if cfg.neutralize_top_left_popup else False
    minimap_masked = bool(cfg.neutralize_minimap)
    reticle_masked = bool(cfg.neutralize_center_reticle)
    if minimap_masked:
        _rect(mask, 0.0, 0.62, 0.245, 1.0)
    if phone_present:
        _rect(mask, 0.70, 0.40, 1.0, 1.0)
    if popup_present:
        _rect(mask, 0.0, 0.0, 0.42, 0.22)
    if reticle_masked:
        _rect(mask, 0.492, 0.492, 0.508, 0.508)
    state = UIState(
        minimap_masked=minimap_masked,
        phone_present=phone_present,
        popup_present=popup_present,
        reticle_masked=reticle_masked,
        occlusion_fraction=float(mask.mean().item()),
    )
    return mask, state


def neutralize_ui(raw_rgb: torch.Tensor, ui_mask: torch.Tensor, *, fill_value: float = 0.5) -> torch.Tensor:
    raw = _as_chw_float(raw_rgb)
    mask = ui_mask.to(device=raw.device, dtype=raw.dtype).clamp(0.0, 1.0)
    return raw * (1.0 - mask) + float(fill_value) * mask
