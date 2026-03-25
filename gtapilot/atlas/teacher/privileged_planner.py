from __future__ import annotations

from dataclasses import dataclass

import torch

from ..utils import KinematicIntegrator
from .teacher_targets import TeacherTrajectorySet


@dataclass
class PlannerCostWeights:
    collision: float = 5.0
    offroad: float = 3.0
    route: float = 1.0
    comfort: float = 0.2


class PrivilegedTrajectoryPlanner:
    """
    Simple classical teacher scaffold for top-K trajectory supervision.

    This is intentionally lightweight: it produces a ranked trajectory set from a
    curvature/speed lattice so Atlas training code has a concrete teacher interface
    to wire against before a richer GTA-specific planner exists.
    """

    def __init__(
        self,
        control_steps: int = 20,
        control_dt: float = 0.2,
        proposal_curvatures: tuple[float, ...] = (-0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20),
        speed_scale: tuple[float, ...] = (0.7, 0.9, 1.0, 1.1),
        top_k: int = 8,
    ):
        self.control_steps = control_steps
        self.control_dt = control_dt
        self.proposal_curvatures = proposal_curvatures
        self.speed_scale = speed_scale
        self.top_k = top_k
        self.rollout = KinematicIntegrator(
            control_dt=control_dt, waypoint_dt=control_dt, control_steps=control_steps
        )

    def _candidate_controls(
        self, batch_size: int, init_speed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        curvatures = []
        speeds = []
        for curvature in self.proposal_curvatures:
            for scale in self.speed_scale:
                curvatures.append(
                    torch.full(
                        (batch_size, 1, self.control_steps),
                        curvature,
                        dtype=init_speed.dtype,
                        device=init_speed.device,
                    )
                )
                speeds.append(
                    (init_speed[:, None, None] * scale).expand(
                        batch_size, 1, self.control_steps
                    )
                )
        return torch.cat(curvatures, dim=1), torch.cat(speeds, dim=1)

    def _route_cost(
        self, trajectories: torch.Tensor, route_polyline: torch.Tensor | None
    ) -> torch.Tensor:
        if route_polyline is None:
            return trajectories.new_zeros(trajectories.shape[:2])
        route_anchor = route_polyline[:, None, -1, :2]
        return torch.linalg.norm(trajectories[:, :, -1, :2] - route_anchor, dim=-1)

    def rank(
        self,
        init_speed: torch.Tensor,
        route_polyline: torch.Tensor | None = None,
        occupancy_penalty: torch.Tensor | None = None,
        cost_weights: PlannerCostWeights | None = None,
    ) -> TeacherTrajectorySet:
        cost_weights = cost_weights or PlannerCostWeights()
        batch_size = init_speed.shape[0]
        curvature, speed = self._candidate_controls(batch_size, init_speed)
        trajectories = self.rollout(curvature, speed, init_speed)

        route_cost = self._route_cost(trajectories, route_polyline)
        comfort_cost = curvature.abs().mean(dim=-1)
        collision_cost = (
            occupancy_penalty
            if occupancy_penalty is not None
            else trajectories.new_zeros(route_cost.shape)
        )
        total_cost = (
            cost_weights.route * route_cost
            + cost_weights.comfort * comfort_cost
            + cost_weights.collision * collision_cost
        )
        top_costs, top_idx = torch.topk(total_cost, k=min(self.top_k, total_cost.shape[1]), largest=False, dim=1)
        gather_idx = top_idx[:, :, None, None].expand(-1, -1, trajectories.shape[2], trajectories.shape[3])
        top_traj = trajectories.gather(1, gather_idx)
        best_index = torch.zeros(batch_size, dtype=torch.long, device=init_speed.device)
        return TeacherTrajectorySet(
            trajectories=top_traj,
            costs=top_costs,
            best_index=best_index,
        )
