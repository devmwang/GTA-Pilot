from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
import torch

from .controller import HighwayControlCommand
from .trajectory_legalizer import LegalizedTrajectorySet
from .ui import UIState


@dataclass(slots=True)
class HighwayDebugFrame:
    raw_rgb: np.ndarray | None = None
    sanitized_rgb: np.ndarray | None = None
    ui_mask: np.ndarray | None = None
    ui_state: UIState | None = None
    outputs: dict[str, torch.Tensor | None] = field(default_factory=dict)
    legalized: LegalizedTrajectorySet | None = None
    control: HighwayControlCommand | None = None
    supervisor_mode: str = "NORMAL"
    takeover_reason: str = ""
    frame_freshness_s: float = 0.0
    latency_s: float = 0.0
    reject_reasons: list[str] = field(default_factory=list)


def _to_uint8_hwc(image: np.ndarray | torch.Tensor | None) -> np.ndarray | None:
    if image is None:
        return None
    if torch.is_tensor(image):
        tensor = image.detach().cpu()
        if tensor.ndim == 3 and tensor.shape[0] == 3:
            tensor = tensor.permute(1, 2, 0)
        elif tensor.ndim == 3 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        arr = tensor.numpy()
    else:
        arr = image
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255.0).astype(np.uint8)
    return arr


def render_highway_debug_overlay(debug: HighwayDebugFrame, size: tuple[int, int] | None = None) -> np.ndarray:
    base = _to_uint8_hwc(debug.sanitized_rgb)
    if base is None:
        base = _to_uint8_hwc(debug.raw_rgb)
    if base is None:
        base = np.zeros((720, 1280, 3), dtype=np.uint8)
    canvas = base.copy()
    if size is not None:
        canvas = cv2.resize(canvas, size, interpolation=cv2.INTER_LINEAR)
    height, width = canvas.shape[:2]
    if debug.ui_mask is not None:
        mask = _to_uint8_hwc(debug.ui_mask)
        if mask is not None:
            if mask.ndim == 3:
                mask = mask[..., 0]
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            canvas[mask > 128] = (0.5 * canvas[mask > 128] + np.array([128, 128, 128])).astype(np.uint8)

    _draw_model_outputs(canvas, debug.outputs, debug.legalized)
    text_lines = [
        f"mode={debug.supervisor_mode} freshness={debug.frame_freshness_s:.3f}s latency={debug.latency_s:.3f}s",
    ]
    if debug.control is not None:
        text_lines.append(
            "ctrl "
            f"steer={debug.control.steer:+.2f} throttle={debug.control.throttle:.2f} "
            f"brake={debug.control.brake:.2f} target={debug.control.target_speed_mps:.1f}m/s "
            f"cand={debug.control.selected_idx}"
        )
    if debug.ui_state is not None:
        text_lines.append(
            "ui "
            f"phone={int(debug.ui_state.phone_present)} popup={int(debug.ui_state.popup_present)} "
            f"reticle={int(debug.ui_state.reticle_present)} minimap={int(debug.ui_state.minimap_present)}"
        )
    if debug.takeover_reason:
        text_lines.append(f"takeover_reason={debug.takeover_reason}")
    if debug.legalized is not None:
        idx = int(debug.legalized.selected_idx)
        text_lines.append(
            "caps "
            f"curve={float(debug.legalized.speed_cap_curve[idx, 0].detach().cpu()):.1f} "
            f"lead={float(debug.legalized.speed_cap_lead[idx, 0].detach().cpu()):.1f} "
            f"ui={float(debug.legalized.speed_cap_ui[idx, 0].detach().cpu()):.1f} "
            f"odd={float(debug.legalized.speed_cap_odd[idx, 0].detach().cpu()):.1f} "
            f"final={float(debug.legalized.speed_cap_final[idx, 0].detach().cpu()):.1f}"
        )
    for reason in debug.reject_reasons[:4]:
        text_lines.append(reason)
    for idx, line in enumerate(text_lines[:8]):
        cv2.putText(
            canvas,
            line,
            (12, 24 + 24 * idx),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return canvas


def summarize_debug_frame(debug: HighwayDebugFrame) -> dict[str, Any]:
    outputs = debug.outputs
    summary: dict[str, Any] = {
        "supervisor_mode": debug.supervisor_mode,
        "takeover_reason": debug.takeover_reason,
        "frame_freshness_s": debug.frame_freshness_s,
        "latency_s": debug.latency_s,
        "reject_reasons": list(debug.reject_reasons),
        "raw_rgb_shape": None if debug.raw_rgb is None else list(debug.raw_rgb.shape),
        "sanitized_rgb_shape": (
            None
            if debug.sanitized_rgb is None
            else list(torch.as_tensor(debug.sanitized_rgb).shape)
        ),
    }
    if debug.ui_state is not None:
        summary["ui_state"] = {
            "minimap_present": debug.ui_state.minimap_present,
            "phone_present": debug.ui_state.phone_present,
            "popup_present": debug.ui_state.popup_present,
            "reticle_present": debug.ui_state.reticle_present,
        }
    for key in (
        "traj_candidates",
        "candidate_logits",
        "lane_lat_pred",
        "lane_valid_logit",
        "lane_conf",
        "road_edge_lat_pred",
        "road_edge_conf",
        "road_followable_conf",
        "lane_tracking_conf",
        "lead_present_logit",
        "lead_state",
        "adjacent_left",
        "adjacent_right",
        "ego_kinematics",
    ):
        value = outputs.get(key)
        if value is not None:
            summary[key] = value.detach().cpu().reshape(-1).tolist()
    if debug.legalized is not None:
        summary["legal_mask"] = debug.legalized.legal_mask.detach().cpu().int().tolist()
        summary["selected_idx"] = int(debug.legalized.selected_idx)
        summary["legalizer_cost"] = debug.legalized.cost.detach().cpu().tolist()
        summary["speed_cap_curve"] = debug.legalized.speed_cap_curve.detach().cpu().tolist()
        summary["speed_cap_lead"] = debug.legalized.speed_cap_lead.detach().cpu().tolist()
        summary["speed_cap_ui"] = debug.legalized.speed_cap_ui.detach().cpu().tolist()
        summary["speed_cap_odd"] = debug.legalized.speed_cap_odd.detach().cpu().tolist()
        summary["speed_cap_final"] = debug.legalized.speed_cap_final.detach().cpu().tolist()
    if debug.control is not None:
        summary["control"] = {
            "steer": debug.control.steer,
            "throttle": debug.control.throttle,
            "brake": debug.control.brake,
            "target_speed_mps": debug.control.target_speed_mps,
            "selected_idx": debug.control.selected_idx,
        }
    return summary


def _draw_model_outputs(
    canvas: np.ndarray,
    outputs: dict[str, torch.Tensor | None],
    legalized: LegalizedTrajectorySet | None,
) -> None:
    height, width = canvas.shape[:2]
    colors = [
        (80, 255, 120),
        (255, 220, 80),
        (80, 160, 255),
        (255, 120, 80),
        (220, 220, 220),
    ]
    traj = legalized.traj if legalized is not None else outputs.get("traj_candidates")
    if traj is not None:
        if traj.ndim == 4:
            traj = traj[0]
        for idx in range(min(traj.shape[0], len(colors))):
            pts = [_ground_to_overlay(float(p[0]), float(p[1]), width, height) for p in traj[idx].detach().cpu()]
            thickness = 4 if legalized is not None and idx == legalized.selected_idx else 2
            for a, b in zip(pts[:-1], pts[1:]):
                cv2.line(canvas, a, b, colors[idx], thickness, cv2.LINE_AA)
    lane_lat = outputs.get("lane_lat_pred")
    lane_long = outputs.get("lane_long_samples_m")
    if lane_lat is not None and lane_long is not None:
        lanes = lane_lat[0].detach().cpu()
        longs = lane_long.detach().cpu()
        for lane_idx, color in enumerate(((120, 200, 255), (255, 255, 255), (120, 200, 255))):
            pts = [_ground_to_overlay(float(long_m), float(lat_m), width, height) for long_m, lat_m in zip(longs, lanes[lane_idx])]
            for a, b in zip(pts[:-1], pts[1:]):
                cv2.line(canvas, a, b, color, 1, cv2.LINE_AA)
    road_edges = outputs.get("road_edge_lat_pred")
    lane_long = outputs.get("lane_long_samples_m")
    if road_edges is not None and lane_long is not None:
        edges = road_edges[0].detach().cpu()
        longs = lane_long.detach().cpu()
        for edge_idx, color in enumerate(((255, 120, 255), (255, 120, 255))):
            pts = [_ground_to_overlay(float(long_m), float(lat_m), width, height) for long_m, lat_m in zip(longs, edges[edge_idx])]
            for a, b in zip(pts[:-1], pts[1:]):
                cv2.line(canvas, a, b, color, 1, cv2.LINE_AA)
    lead_state = outputs.get("lead_state")
    lead_logit = outputs.get("lead_present_logit")
    if lead_state is not None and lead_logit is not None:
        if float(torch.sigmoid(lead_logit.reshape(-1)[0]).detach().cpu()) > 0.5:
            lead = lead_state.reshape(-1, 5)[0].detach().cpu()
            cv2.circle(
                canvas,
                _ground_to_overlay(float(lead[0]), float(lead[1]), width, height),
                6,
                (255, 255, 0),
                -1,
                cv2.LINE_AA,
            )


def _ground_to_overlay(long_m: float, lat_m: float, width: int, height: int) -> tuple[int, int]:
    x = int(width * 0.5 + lat_m * width / 48.0)
    y = int(height * 0.96 - long_m * height / 130.0)
    return (max(0, min(width - 1, x)), max(0, min(height - 1, y)))
