from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import torch

from .controller import CruiseControlCommand
from .trajectory_legalizer import LegalizedCruiseTrajectory
from .ui import UIState


@dataclass(slots=True)
class CruiseDebugFrame:
    raw_rgb: np.ndarray | None
    sanitized_rgb: torch.Tensor | None
    ui_mask: torch.Tensor | None
    ui_state: UIState | None
    outputs: dict[str, torch.Tensor]
    legalized: LegalizedCruiseTrajectory | None
    control: CruiseControlCommand | None
    frame_freshness_s: float
    latency_s: float


def trajectory_topdown_image(
    traj: torch.Tensor,
    *,
    target_traj: torch.Tensor | None = None,
    image_size: tuple[int, int] = (480, 360),
    long_max_m: float = 90.0,
    lat_span_m: float = 24.0,
) -> np.ndarray:
    h, w = image_size
    image = np.full((h, w, 3), 245, dtype=np.uint8)
    origin = (w // 2, h - 24)
    cv2.line(image, (origin[0], 0), origin, (220, 220, 220), 1)
    cv2.line(image, (0, origin[1]), (w, origin[1]), (220, 220, 220), 1)
    _draw_traj(image, traj, origin, long_max_m, lat_span_m, (20, 90, 220))
    if target_traj is not None:
        _draw_traj(image, target_traj, origin, long_max_m, lat_span_m, (40, 170, 60))
    cv2.circle(image, origin, 5, (20, 20, 20), -1)
    return image


def summarize_debug_frame(debug: CruiseDebugFrame) -> dict[str, Any]:
    outputs = debug.outputs
    traj = outputs["traj"].detach().float()
    ego = outputs["ego_kinematics"].detach().float()
    summary: dict[str, Any] = {
        "traj_conf": float(torch.sigmoid(outputs["traj_conf_logit"]).reshape(-1)[0].detach().cpu()),
        "slow_or_brake_prob": float(torch.sigmoid(outputs["slow_or_brake_logit"]).reshape(-1)[0].detach().cpu()),
        "fallback_prob": float(torch.sigmoid(outputs["fallback_logit"]).reshape(-1)[0].detach().cpu()),
        "pred_speed_mps": float(ego.reshape(-1)[0].detach().cpu()) if ego.numel() else 0.0,
        "traj_first": traj.reshape(-1, traj.shape[-2], 4)[0, 0].detach().cpu().tolist(),
        "traj_final": traj.reshape(-1, traj.shape[-2], 4)[0, -1].detach().cpu().tolist(),
        "frame_freshness_s": float(debug.frame_freshness_s),
        "latency_s": float(debug.latency_s),
    }
    if debug.ui_state is not None:
        summary["ui"] = {
            "phone_present": debug.ui_state.phone_present,
            "popup_present": debug.ui_state.popup_present,
            "occlusion_fraction": debug.ui_state.occlusion_fraction,
        }
    if debug.legalized is not None:
        summary["authorized_speed_mps"] = float(debug.legalized.authorized_speed[min(2, debug.legalized.authorized_speed.numel() - 1)].detach().cpu())
        summary["legalizer_valid"] = bool(debug.legalized.valid)
        summary["reject_reasons"] = list(debug.legalized.reject_reasons)
    if debug.control is not None:
        summary["control"] = {
            "steer": debug.control.steer,
            "throttle": debug.control.throttle,
            "brake": debug.control.brake,
            "target_speed_mps": debug.control.target_speed_mps,
            "fallback_active": debug.control.fallback_active,
        }
    return summary


def _draw_traj(
    image: np.ndarray,
    traj: torch.Tensor,
    origin: tuple[int, int],
    long_max_m: float,
    lat_span_m: float,
    color: tuple[int, int, int],
) -> None:
    arr = traj.detach().float().cpu().reshape(-1, 4).numpy()
    points: list[tuple[int, int]] = []
    h, w = image.shape[:2]
    for long_m, lat_m, _, _ in arr:
        x = int(round(origin[0] + (float(lat_m) / lat_span_m) * w))
        y = int(round(origin[1] - (float(long_m) / long_max_m) * (h - 36)))
        if 0 <= x < w and 0 <= y < h:
            points.append((x, y))
    for p0, p1 in zip(points, points[1:]):
        cv2.line(image, p0, p1, color, 2, lineType=cv2.LINE_AA)
    for point in points[:: max(1, len(points) // 8)]:
        cv2.circle(image, point, 3, color, -1)
