from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .config import AtlasCruiseConfig


@dataclass(slots=True)
class LegalizedCruiseTrajectory:
    traj: torch.Tensor
    valid: bool
    curvature: torch.Tensor
    speed_cap_curve: torch.Tensor
    authorized_speed: torch.Tensor
    reject_reasons: list[str]
    slow_or_brake_prob: float
    fallback_prob: float


def estimate_curvature(traj: torch.Tensor) -> torch.Tensor:
    if traj.shape[-1] != 4:
        raise ValueError("traj must end with [long_m, lat_m, yaw_rad, speed_mps].")
    long_m = traj[..., 0]
    lat_m = traj[..., 1]
    dx = torch.gradient(long_m, dim=-1)[0]
    dy = torch.gradient(lat_m, dim=-1)[0]
    ddx = torch.gradient(dx, dim=-1)[0]
    ddy = torch.gradient(dy, dim=-1)[0]
    return (dx * ddy - dy * ddx) / (dx * dx + dy * dy).clamp_min(1e-4).pow(1.5)


def recompute_yaw_from_geometry(traj: torch.Tensor) -> torch.Tensor:
    out = traj.clone()
    dlong = torch.gradient(out[..., 0], dim=-1)[0]
    dlat = torch.gradient(out[..., 1], dim=-1)[0]
    out[..., 2] = torch.atan2(dlat, dlong.clamp_min(1e-3))
    return out


def smooth_trajectory(traj: torch.Tensor) -> torch.Tensor:
    if traj.shape[-2] < 3:
        return traj
    original_shape = traj.shape
    flat = traj.reshape(-1, traj.shape[-2], traj.shape[-1]).transpose(1, 2)
    padded = F.pad(flat, (1, 1), mode="replicate")
    smoothed = F.avg_pool1d(padded, kernel_size=3, stride=1).transpose(1, 2)
    out = smoothed.reshape(original_shape)
    out[..., 0] = torch.maximum(out[..., 0], traj[..., 0])
    return out


class CruiseTrajectoryLegalizer:
    def __init__(self, cfg: AtlasCruiseConfig | None = None):
        self.cfg = cfg or AtlasCruiseConfig()
        self.cfg.validate()

    def sanitize(self, traj: torch.Tensor) -> tuple[torch.Tensor, bool, list[str]]:
        if traj.ndim == 3:
            if traj.shape[0] != 1:
                raise ValueError("Cruise runtime legalizer expects one trajectory at a time.")
            traj = traj[0]
        if traj.ndim != 2 or traj.shape[-1] != 4:
            raise ValueError("traj must have shape [N, 4].")
        reasons: list[str] = []
        valid = True
        out = traj.detach().clone()
        if not bool(torch.isfinite(out).all().item()):
            valid = False
            reasons.append("reject_nan")
        out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        if bool((out[:, 3] < -0.5).any().item()):
            valid = False
            reasons.append("reject_negative_speed")
        out[:, 3] = out[:, 3].clamp_min(0.0)
        if bool((torch.diff(out[:, 0]) < -0.5).any().item()):
            reasons.append("repair_nonmonotonic_long")
        out[:, 0] = torch.cummax(out[:, 0], dim=0).values
        if bool((out[:, 2].abs() > 1.7).any().item()):
            reasons.append("repair_yaw_from_geometry")
            out = recompute_yaw_from_geometry(out)
        out = smooth_trajectory(out)
        if bool((torch.diff(out[:, 0]).abs() < 1e-4).all().item()):
            valid = False
            reasons.append("reject_invalid_forward_progress")
        return out, valid, reasons

    def legalize(
        self,
        traj: torch.Tensor,
        *,
        slow_or_brake_logit: torch.Tensor | float | None = None,
        fallback_logit: torch.Tensor | float | None = None,
        frame_stale_s: float = 0.0,
        user_max_speed_mps: float | None = None,
        caution: bool = False,
        rain: bool = False,
    ) -> LegalizedCruiseTrajectory:
        sanitized, valid, reasons = self.sanitize(traj)
        curvature = estimate_curvature(sanitized)
        lat_limit = self.cfg.comfort_lat_acc_mps2 * (self.cfg.rain_lat_multiplier if rain else 1.0)
        speed_cap_curve = torch.sqrt(
            torch.full_like(curvature, lat_limit) / curvature.abs().clamp_min(1e-4)
        )
        max_speed = self.cfg.user_max_speed_mps if user_max_speed_mps is None else float(user_max_speed_mps)
        speed_cap = torch.full_like(speed_cap_curve, max_speed)
        slow_prob = _sigmoid_float(slow_or_brake_logit)
        fallback_prob = _sigmoid_float(fallback_logit)
        if caution:
            speed_cap = torch.minimum(speed_cap, torch.full_like(speed_cap, self.cfg.caution_speed_cap_mps))
        if slow_prob > 0.5:
            reduce = max(0.25, 1.0 - slow_prob)
            speed_cap = torch.minimum(speed_cap, sanitized[:, 3].clamp_min(0.0) * reduce)
        if fallback_prob > 0.5 or frame_stale_s > self.cfg.max_frame_staleness_s:
            speed_cap = torch.minimum(speed_cap, torch.full_like(speed_cap, self.cfg.fallback_speed_cap_mps))
            reasons.append("fallback_or_stale_slowdown")
        authorized = torch.minimum(torch.minimum(sanitized[:, 3].clamp_min(0.0), speed_cap_curve), speed_cap)
        out = sanitized.clone()
        out[:, 3] = authorized
        hard_cap = torch.sqrt(
            torch.full_like(curvature, self.cfg.hard_lat_acc_mps2) / curvature.abs().clamp_min(1e-4)
        )
        if bool((sanitized[:, 3] > hard_cap + 2.0).any().item()):
            reasons.append("curve_speed_clamped")
        return LegalizedCruiseTrajectory(
            traj=out,
            valid=valid,
            curvature=curvature,
            speed_cap_curve=speed_cap_curve,
            authorized_speed=authorized,
            reject_reasons=reasons,
            slow_or_brake_prob=slow_prob,
            fallback_prob=fallback_prob,
        )


def _sigmoid_float(value: torch.Tensor | float | None) -> float:
    if value is None:
        return 0.0
    if isinstance(value, torch.Tensor):
        return float(torch.sigmoid(value.detach().reshape(-1)[0]).cpu().item())
    return float(torch.sigmoid(torch.tensor(float(value))).item())
