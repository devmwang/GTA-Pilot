from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import AtlasHAConfig
from .ego_state import EgoState


@dataclass(slots=True)
class MinimumRiskPlan:
    traj: torch.Tensor
    request_takeover: bool
    reason: str


class MinimumRiskManeuver:
    def __init__(self, cfg: AtlasHAConfig):
        self.cfg = cfg

    def build_plan(
        self,
        *,
        ego_state: EgoState,
        previous_stable_path: torch.Tensor | None,
        current_lane_lat: torch.Tensor | None,
        confidence: float,
        low_confidence_age_s: float,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> MinimumRiskPlan:
        long_m = torch.linspace(1.0, 45.0, self.cfg.num_traj_points, device=device, dtype=dtype)
        speed = torch.linspace(
            max(0.0, ego_state.speed_mps),
            0.0,
            self.cfg.num_traj_points,
            device=device,
            dtype=dtype,
        )
        traj = torch.zeros(self.cfg.num_traj_points, 4, device=device, dtype=dtype)
        traj[:, 0] = long_m
        traj[:, 3] = speed
        reason = "straight_braking_path"
        if previous_stable_path is not None and previous_stable_path.shape[-2:] == traj.shape[-2:]:
            traj[:, :3] = previous_stable_path.to(device=device, dtype=dtype)[:, :3]
            traj[:, 3] = speed
            reason = "previous_stable_path"
        elif current_lane_lat is not None and current_lane_lat.numel() >= self.cfg.num_traj_points:
            traj[:, 1] = current_lane_lat[: self.cfg.num_traj_points].to(device=device, dtype=dtype)
            reason = "current_lane"
        request_takeover = confidence < 0.15 and low_confidence_age_s > 1.5
        return MinimumRiskPlan(
            traj=traj,
            request_takeover=request_takeover,
            reason=reason,
        )
