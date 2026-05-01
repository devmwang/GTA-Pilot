from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .config import AtlasHAConfig


@dataclass(slots=True)
class UIState:
    minimap_present: bool
    phone_present: bool
    popup_present: bool
    reticle_present: bool
    ui_mask: torch.Tensor
    phone_mask: torch.Tensor | None
    popup_mask: torch.Tensor | None
    reticle_mask: torch.Tensor | None


@dataclass(slots=True)
class UIPreprocessorOutput:
    scene_rgb: torch.Tensor
    ui_mask: torch.Tensor
    minimap_crop: torch.Tensor | None
    ui_state: UIState


class UIPreprocessor:
    def __init__(self, cfg: AtlasHAConfig):
        self.cfg = cfg
        self._last_clean_scene: torch.Tensor | None = None

    @staticmethod
    def _ensure_chw_float(frame_rgb: torch.Tensor) -> torch.Tensor:
        x = torch.as_tensor(frame_rgb)
        if x.ndim == 3 and x.shape[-1] == 3:
            x = x.permute(2, 0, 1)
        if x.ndim != 3 or x.shape[0] != 3:
            raise ValueError("frame_rgb must have shape [3,H,W] or [H,W,3].")
        if x.dtype == torch.uint8:
            x = x.float().mul_(1.0 / 255.0)
        elif not torch.is_floating_point(x):
            x = x.float()
        return x.clamp(0.0, 1.0)

    @staticmethod
    def _roi_mask(
        height: int,
        width: int,
        y0: int,
        y1: int,
        x0: int,
        x1: int,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        mask = torch.zeros(1, height, width, device=device)
        mask[:, max(0, y0) : min(height, y1), max(0, x0) : min(width, x1)] = 1.0
        return mask

    @staticmethod
    def _roi_stats(frame: torch.Tensor, mask: torch.Tensor) -> tuple[float, float]:
        values = frame[mask.expand_as(frame).bool()]
        if values.numel() == 0:
            return 0.0, 0.0
        return float(values.mean().item()), float(values.std(unbiased=False).item())

    def _detect_phone(self, frame: torch.Tensor, phone_mask: torch.Tensor) -> bool:
        if not self.cfg.enable_phone_detector:
            return False
        mean, std = self._roi_stats(frame, phone_mask)
        return std > 0.18 and mean > 0.08

    def _detect_popup(self, frame: torch.Tensor, popup_mask: torch.Tensor) -> bool:
        mean, std = self._roi_stats(frame, popup_mask)
        return std > 0.12 and mean > 0.10

    def _detect_reticle(self, frame: torch.Tensor, reticle_mask: torch.Tensor) -> bool:
        _, std = self._roi_stats(frame, reticle_mask)
        return std > 0.20

    def _fill_masked(self, frame: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self._last_clean_scene is not None and self._last_clean_scene.shape == frame.shape:
            fill = self._last_clean_scene.to(device=frame.device, dtype=frame.dtype)
        else:
            fill = torch.full_like(frame, 0.5)
        return frame * (1.0 - mask) + fill * mask

    def preprocess(self, frame_rgb: torch.Tensor) -> UIPreprocessorOutput:
        frame = self._ensure_chw_float(frame_rgb)
        height, width = frame.shape[-2:]
        minimap_mask = self._roi_mask(
            height,
            width,
            int(height * 0.68),
            height,
            0,
            int(width * 0.28),
            device=frame.device,
        )
        phone_mask = self._roi_mask(
            height,
            width,
            int(height * 0.42),
            height,
            int(width * 0.70),
            width,
            device=frame.device,
        )
        popup_mask = self._roi_mask(
            height,
            width,
            0,
            int(height * 0.22),
            0,
            int(width * 0.42),
            device=frame.device,
        )
        reticle_mask = self._roi_mask(
            height,
            width,
            int(height * 0.48),
            int(height * 0.52),
            int(width * 0.48),
            int(width * 0.52),
            device=frame.device,
        )

        minimap_present = self.cfg.enable_minimap_crop
        phone_present = self._detect_phone(frame, phone_mask)
        popup_present = self._detect_popup(frame, popup_mask)
        reticle_present = self._detect_reticle(frame, reticle_mask)

        ui_mask = torch.zeros(1, height, width, device=frame.device, dtype=frame.dtype)
        if minimap_present:
            ui_mask = torch.maximum(ui_mask, minimap_mask.to(dtype=frame.dtype))
        if phone_present:
            ui_mask = torch.maximum(ui_mask, phone_mask.to(dtype=frame.dtype))
        if popup_present:
            ui_mask = torch.maximum(ui_mask, popup_mask.to(dtype=frame.dtype))
        if reticle_present:
            ui_mask = torch.maximum(ui_mask, reticle_mask.to(dtype=frame.dtype))

        scene = self._fill_masked(frame, ui_mask)
        self._last_clean_scene = scene.detach()
        scene_resized = F.interpolate(
            scene.unsqueeze(0),
            size=(self.cfg.input_h, self.cfg.input_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        mask_resized = F.interpolate(
            ui_mask.unsqueeze(0),
            size=(self.cfg.input_h, self.cfg.input_w),
            mode="nearest",
        ).squeeze(0)
        minimap_crop = frame[:, int(height * 0.68) : height, 0 : int(width * 0.28)]
        ui_state = UIState(
            minimap_present=minimap_present,
            phone_present=phone_present,
            popup_present=popup_present,
            reticle_present=reticle_present,
            ui_mask=mask_resized,
            phone_mask=phone_mask if phone_present else None,
            popup_mask=popup_mask if popup_present else None,
            reticle_mask=reticle_mask if reticle_present else None,
        )
        return UIPreprocessorOutput(
            scene_rgb=scene_resized,
            ui_mask=mask_resized,
            minimap_crop=minimap_crop if minimap_present else None,
            ui_state=ui_state,
        )
