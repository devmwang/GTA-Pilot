from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .config import AtlasCruiseConfig
from .trajectory_legalizer import LegalizedCruiseTrajectory


@dataclass(slots=True)
class CruiseControlCommand:
    steer: float
    throttle: float
    brake: float
    handbrake: float
    reverse: float
    pilot_active: float
    target_speed_mps: float
    fallback_active: bool

    def as_action_vector(self) -> list[float]:
        return [
            self.steer,
            self.throttle,
            self.brake,
            self.handbrake,
            self.reverse,
            self.pilot_active,
        ]


class CruiseController:
    def __init__(self, cfg: AtlasCruiseConfig | None = None):
        self.cfg = cfg or AtlasCruiseConfig()
        self.cfg.validate()
        self._prev_steer = 0.0
        self._prev_throttle = 0.0
        self._prev_brake = 0.0
        self._speed_error_integral = 0.0

    def reset(self) -> None:
        self._prev_steer = 0.0
        self._prev_throttle = 0.0
        self._prev_brake = 0.0
        self._speed_error_integral = 0.0

    def _pure_pursuit(self, traj: torch.Tensor, speed_mps: float) -> float:
        speed = max(0.0, float(speed_mps))
        lookahead = max(5.0, min(35.0, 0.8 * speed + 6.0))
        idx = int(torch.argmin((traj[:, 0] - lookahead).abs()).item())
        point_long = max(1e-3, float(traj[idx, 0].item()))
        point_lat = float(traj[idx, 1].item())
        alpha = math.atan2(point_lat, point_long)
        curvature = 2.0 * math.sin(alpha) / max(point_long, 1e-3)
        steer_rad = math.atan(self.cfg.wheelbase_eff_m * curvature)
        return max(-1.0, min(1.0, steer_rad / max(self.cfg.steer_max_rad, 1e-3)))

    def _longitudinal(
        self,
        target_speed_mps: float,
        current_speed_mps: float,
        *,
        slow_or_brake_prob: float,
        fallback_active: bool,
        dt_s: float,
    ) -> tuple[float, float]:
        error = float(target_speed_mps) - float(current_speed_mps)
        if slow_or_brake_prob > 0.5:
            error = min(error, -2.0 * slow_or_brake_prob)
        if fallback_active:
            error = min(error, -4.0)
        self._speed_error_integral = max(-10.0, min(10.0, self._speed_error_integral + error * dt_s))
        command = 0.18 * error + 0.02 * self._speed_error_integral
        throttle = max(0.0, min(1.0, command))
        brake = max(0.0, min(1.0, -0.24 * error))
        if brake > 0.02:
            throttle = 0.0
        return throttle, brake

    def compute(
        self,
        legalized: LegalizedCruiseTrajectory,
        *,
        ego_kinematics: torch.Tensor,
        dt_s: float = 1.0 / 60.0,
    ) -> CruiseControlCommand:
        ego = ego_kinematics.detach().float().reshape(-1)
        current_speed = float(ego[0].item()) if ego.numel() else 0.0
        traj = legalized.traj.detach().float()
        steer = self._pure_pursuit(traj, current_speed)
        target_idx = min(2, traj.shape[0] - 1)
        target_speed = float(legalized.authorized_speed[target_idx].detach().cpu().item())
        fallback_active = legalized.fallback_prob > 0.5 or not legalized.valid
        throttle, brake = self._longitudinal(
            target_speed,
            current_speed,
            slow_or_brake_prob=legalized.slow_or_brake_prob,
            fallback_active=fallback_active,
            dt_s=dt_s,
        )
        steer = _rate_limit(steer, self._prev_steer, self.cfg.steer_rate_limit_per_s * dt_s)
        throttle = _rate_limit(throttle, self._prev_throttle, self.cfg.throttle_rate_limit_per_s * dt_s)
        brake = _rate_limit(brake, self._prev_brake, self.cfg.brake_rate_limit_per_s * dt_s)
        self._prev_steer = steer
        self._prev_throttle = throttle
        self._prev_brake = brake
        return CruiseControlCommand(
            steer=max(-1.0, min(1.0, steer)),
            throttle=max(0.0, min(1.0, throttle)),
            brake=max(0.0, min(1.0, brake)),
            handbrake=0.0,
            reverse=0.0,
            pilot_active=1.0,
            target_speed_mps=target_speed,
            fallback_active=fallback_active,
        )


def _rate_limit(value: float, previous: float, max_delta: float) -> float:
    return max(previous - max_delta, min(previous + max_delta, float(value)))
