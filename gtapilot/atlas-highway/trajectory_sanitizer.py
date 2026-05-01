from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import AtlasHAConfig
from .vehicle_profile import VehicleDynamicsProfile


@dataclass(slots=True)
class SanitizedTrajectorySet:
    traj: torch.Tensor
    valid_mask: torch.Tensor
    curvature: torch.Tensor
    reject_reasons: list[list[str]]


def estimate_curvature(traj: torch.Tensor) -> torch.Tensor:
    if traj.shape[-1] != 4:
        raise ValueError("traj must end with [long_m, lat_m, yaw_rad, speed_mps].")
    long_m = traj[..., 0]
    lat_m = traj[..., 1]
    dx = torch.gradient(long_m, dim=-1)[0]
    dy = torch.gradient(lat_m, dim=-1)[0]
    ddx = torch.gradient(dx, dim=-1)[0]
    ddy = torch.gradient(dy, dim=-1)[0]
    denom = (dx * dx + dy * dy).clamp_min(1e-4).pow(1.5)
    return (dx * ddy - dy * ddx) / denom


def recompute_yaw_from_geometry(traj: torch.Tensor) -> torch.Tensor:
    out = traj.clone()
    dlong = torch.gradient(out[..., 0], dim=-1)[0]
    dlat = torch.gradient(out[..., 1], dim=-1)[0]
    out[..., 2] = torch.atan2(dlat, dlong.clamp_min(1e-3))
    return out


def _smooth_traj(traj: torch.Tensor) -> torch.Tensor:
    if traj.shape[-2] < 5:
        return traj
    out = traj.clone()
    center = traj[..., 1:-1, :]
    out[..., 1:-1, :] = 0.25 * traj[..., :-2, :] + 0.50 * center + 0.25 * traj[..., 2:, :]
    out[..., 0] = torch.maximum(out[..., 0], traj[..., 0])
    return out


class TrajectorySanitizer:
    def __init__(self, cfg: AtlasHAConfig, vehicle_profile: VehicleDynamicsProfile):
        self.cfg = cfg
        self.profile = vehicle_profile

    def sanitize(self, traj_candidates: torch.Tensor) -> SanitizedTrajectorySet:
        if traj_candidates.ndim != 3 or traj_candidates.shape[-1] != 4:
            raise ValueError("traj_candidates must have shape [K, N, 4].")
        traj = traj_candidates.detach().clone()
        valid = torch.ones(traj.shape[0], device=traj.device, dtype=torch.bool)
        reasons: list[list[str]] = [[] for _ in range(traj.shape[0])]

        finite = torch.isfinite(traj).all(dim=(-1, -2))
        for idx in torch.where(~finite)[0].tolist():
            valid[idx] = False
            reasons[idx].append("reject_nan")
        traj = torch.nan_to_num(traj, nan=0.0, posinf=0.0, neginf=0.0)

        negative_speed = (traj[..., 3] < -0.5).any(dim=-1)
        for idx in torch.where(negative_speed)[0].tolist():
            valid[idx] = False
            reasons[idx].append("reject_negative_speed")
        traj[..., 3] = traj[..., 3].clamp_min(0.0)

        long_diff = torch.diff(traj[..., 0], dim=-1)
        nonmonotonic = (long_diff < -0.5).any(dim=-1)
        for idx in torch.where(nonmonotonic)[0].tolist():
            valid[idx] = False
            reasons[idx].append("reject_nonmonotonic_long")
        traj[..., 0] = torch.cummax(traj[..., 0], dim=-1).values

        yaw_bad = (~torch.isfinite(traj[..., 2])).any(dim=-1) | (traj[..., 2].abs() > 1.8).any(dim=-1)
        if bool(yaw_bad.any().item()):
            traj = recompute_yaw_from_geometry(traj)
            for idx in torch.where(yaw_bad)[0].tolist():
                reasons[idx].append("reject_bad_yaw")

        traj = _smooth_traj(traj)
        curvature = estimate_curvature(traj)
        steer = torch.atan(self.profile.wheelbase_eff_m * curvature)
        curvature_excess = (steer.abs() > self.profile.steer_max_rad * 1.25).any(dim=-1)
        for idx in torch.where(curvature_excess)[0].tolist():
            valid[idx] = False
            reasons[idx].append("reject_curvature_excess")

        bad_geometry = (torch.diff(traj[..., 0], dim=-1).abs() < 1e-4).all(dim=-1)
        for idx in torch.where(bad_geometry)[0].tolist():
            valid[idx] = False
            reasons[idx].append("reject_invalid_geometry")
        return SanitizedTrajectorySet(
            traj=traj,
            valid_mask=valid,
            curvature=curvature,
            reject_reasons=reasons,
        )
