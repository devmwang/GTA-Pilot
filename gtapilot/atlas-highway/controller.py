from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .actuator_calibration import ActuatorCalibration
from .ego_state import EgoState
from .state import SupervisorState
from .trajectory_legalizer import LegalizedTrajectorySet
from .vehicle_profile import VehicleDynamicsProfile


@dataclass(slots=True)
class HighwayControlCommand:
    steer: float
    throttle: float
    brake: float
    handbrake: float
    reverse: float
    pilot_active: float
    target_speed_mps: float
    supervisor_mode: str
    selected_idx: int

    def as_action_vector(self) -> list[float]:
        return [
            self.steer,
            self.throttle,
            self.brake,
            self.handbrake,
            self.reverse,
            self.pilot_active,
        ]


class HighwayController:
    def __init__(
        self,
        vehicle_profile: VehicleDynamicsProfile,
        calibration: ActuatorCalibration | None = None,
    ):
        self.profile = vehicle_profile
        self.calibration = calibration or ActuatorCalibration()
        self._prev_throttle = 0.0
        self._prev_brake = 0.0

    def _pure_pursuit(self, traj: torch.Tensor, ego_state: EgoState) -> float:
        speed = max(ego_state.speed_mps, 1.0)
        lookahead = max(6.0, min(35.0, 0.9 * speed + 5.0))
        long_m = traj[:, 0]
        idx = int(torch.argmin((long_m - lookahead).abs()).item())
        point_long = max(1e-3, float(traj[idx, 0].item()))
        point_lat = float(traj[idx, 1].item())
        alpha = math.atan2(point_lat, point_long)
        curvature = 2.0 * math.sin(alpha) / max(point_long, 1e-3)
        steer_rad = math.atan(self.profile.wheelbase_eff_m * curvature)
        return self.calibration.map_steer(steer_rad / max(self.profile.steer_max_rad, 1e-3))

    def _longitudinal(
        self,
        target_speed_mps: float,
        ego_state: EgoState,
        lead_present: bool,
        lead_state: torch.Tensor | None,
        supervisor_mode: str,
        dt_s: float,
    ) -> tuple[float, float]:
        error = float(target_speed_mps) - float(ego_state.speed_mps)
        if supervisor_mode in {"MINIMUM_RISK", "TAKEOVER_REQUESTED"}:
            error = min(error, -2.0)
        if lead_present and lead_state is not None:
            lead = lead_state.detach().float().reshape(-1)
            if lead.numel() >= 5 and float(lead[4].item()) < 1.5:
                error = min(error, -5.0)
        throttle = max(0.0, min(1.0, 0.18 * error))
        brake = max(0.0, min(1.0, -0.22 * error))
        max_delta = max(0.05, float(dt_s) * 3.0)
        throttle = _rate_limit(throttle, self._prev_throttle, max_delta)
        brake = _rate_limit(brake, self._prev_brake, max_delta)
        self._prev_throttle = throttle
        self._prev_brake = brake
        return self.calibration.map_throttle(throttle), self.calibration.map_brake(brake)

    def compute(
        self,
        legalized: LegalizedTrajectorySet,
        *,
        ego_state: EgoState,
        supervisor_state: SupervisorState,
        selected_idx: int | None = None,
        lead_present: bool = False,
        lead_state: torch.Tensor | None = None,
        dt_s: float = 1.0 / 60.0,
    ) -> HighwayControlCommand:
        idx = legalized.selected_idx if selected_idx is None else int(selected_idx)
        traj = legalized.traj[idx]
        target_speed = float(legalized.speed_cap_final[idx, min(2, traj.shape[0] - 1)].item())
        steer = self._pure_pursuit(traj, ego_state)
        throttle, brake = self._longitudinal(
            target_speed,
            ego_state,
            lead_present,
            lead_state,
            supervisor_state.mode,
            dt_s,
        )
        if supervisor_state.mode == "TAKEOVER_REQUESTED":
            throttle = 0.0
            brake = max(brake, 0.35)
        return HighwayControlCommand(
            steer=steer,
            throttle=throttle,
            brake=brake,
            handbrake=0.0,
            reverse=0.0,
            pilot_active=1.0,
            target_speed_mps=target_speed,
            supervisor_mode=supervisor_state.mode,
            selected_idx=idx,
        )


def _rate_limit(value: float, previous: float, max_delta: float) -> float:
    return max(previous - max_delta, min(previous + max_delta, value))
