from __future__ import annotations

import math

import torch

from .ego_state import EgoState


def predict_ego_state(ego_state: EgoState, dt_s: float) -> EgoState:
    dt_s = max(0.0, float(dt_s))
    speed = max(0.0, ego_state.speed_mps + ego_state.a_long_mps2 * dt_s)
    return EgoState(
        speed_mps=speed,
        a_long_mps2=ego_state.a_long_mps2,
        yaw_rate_radps=ego_state.yaw_rate_radps,
        curvature_inv_m=ego_state.curvature_inv_m,
        confidence=ego_state.confidence,
        timestamp_ns=int(ego_state.timestamp_ns + dt_s * 1e9),
    )


def latency_compensate_trajectory(
    traj: torch.Tensor,
    ego_state: EgoState,
    latency_s: float,
) -> torch.Tensor:
    if traj.ndim < 2 or traj.shape[-1] != 4:
        raise ValueError("traj must have shape [..., N, 4].")
    dt = max(0.0, float(latency_s))
    predicted = predict_ego_state(ego_state, dt)
    forward_shift = ego_state.speed_mps * dt + 0.5 * ego_state.a_long_mps2 * dt * dt
    yaw_shift = ego_state.yaw_rate_radps * dt
    out = traj.clone()
    long_m = out[..., 0] - forward_shift
    lat_m = out[..., 1]
    cos_yaw = math.cos(-yaw_shift)
    sin_yaw = math.sin(-yaw_shift)
    out[..., 0] = long_m * cos_yaw - lat_m * sin_yaw
    out[..., 1] = long_m * sin_yaw + lat_m * cos_yaw
    out[..., 2] = out[..., 2] - yaw_shift
    out[..., 3] = torch.clamp(out[..., 3], min=0.0)
    del predicted
    return out
