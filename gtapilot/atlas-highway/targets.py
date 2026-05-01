from __future__ import annotations

import torch

from .config import AtlasHAConfig
from .trajectory_legalizer import LegalizedTrajectorySet


def build_future_trajectory_target(
    future_ego_poses: torch.Tensor,
    *,
    current_pose: torch.Tensor | None = None,
    latency_steps: int = 0,
) -> torch.Tensor:
    poses = future_ego_poses.float()
    if poses.ndim != 2 or poses.shape[-1] < 4:
        raise ValueError("future_ego_poses must be [T, >=4] with [long_m, lat_m, yaw_rad, speed_mps].")
    start = min(max(0, int(latency_steps)), poses.shape[0] - 1)
    target = poses[start:, :4]
    if current_pose is not None:
        origin = current_pose.float().reshape(-1)
        target = target.clone()
        target[:, 0] -= origin[0]
        target[:, 1] -= origin[1]
        target[:, 2] -= origin[2]
    return target


def classify_candidate(target_traj: torch.Tensor) -> torch.Tensor:
    lat_final = float(target_traj[-1, 1].item())
    speed_final = float(target_traj[-1, 3].item())
    speed_initial = float(target_traj[0, 3].item())
    if speed_final < 2.0 or speed_final < speed_initial - 8.0:
        return torch.tensor(4, dtype=torch.long)
    if lat_final > 2.2:
        return torch.tensor(2, dtype=torch.long)
    if lat_final < -2.2:
        return torch.tensor(3, dtype=torch.long)
    if speed_final < speed_initial - 2.0:
        return torch.tensor(1, dtype=torch.long)
    return torch.tensor(0, dtype=torch.long)


def build_ego_target(speed_mps: float, a_long_mps2: float, yaw_rate_radps: float, curvature_inv_m: float) -> torch.Tensor:
    return torch.tensor(
        [speed_mps, a_long_mps2, yaw_rate_radps, curvature_inv_m],
        dtype=torch.float32,
    )


def build_lead_target(
    lead_long_m: float | None,
    lead_lat_m: float | None,
    lead_speed_mps: float | None,
    ego_speed_mps: float,
    *,
    max_ttc_s: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if lead_long_m is None or lead_lat_m is None or lead_speed_mps is None:
        return torch.zeros(1), torch.zeros(5)
    distance = max(0.0, float(lead_long_m))
    rel_speed = float(lead_speed_mps) - float(ego_speed_mps)
    closing = max(0.0, -rel_speed)
    ttc = max_ttc_s if closing <= 1e-3 else min(max_ttc_s, distance / closing)
    return (
        torch.ones(1),
        torch.tensor([lead_long_m, lead_lat_m, distance, rel_speed, ttc], dtype=torch.float32),
    )


def build_lane_targets_from_centerline(
    center_lat_m: torch.Tensor,
    cfg: AtlasHAConfig,
    *,
    lane_width_m: float = 3.6,
    valid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    center = center_lat_m.float()
    if center.numel() != cfg.num_lane_points:
        center = torch.nn.functional.interpolate(
            center.view(1, 1, -1),
            size=cfg.num_lane_points,
            mode="linear",
            align_corners=False,
        ).view(-1)
    lane_lat = torch.stack([center + lane_width_m, center, center - lane_width_m], dim=0)
    lane_valid = torch.ones_like(lane_lat) if valid is None else valid.float().expand_as(lane_lat)
    lane_conf = lane_valid.mean(dim=-1).clamp(0.0, 1.0)
    return lane_lat, lane_valid, lane_conf


def build_legal_speed_target(legalized: LegalizedTrajectorySet, target_candidate: torch.Tensor) -> torch.Tensor:
    idx = int(target_candidate.reshape(-1)[0].item())
    return legalized.speed_cap_final[idx].detach().float()
