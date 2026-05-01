from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch


@dataclass(slots=True)
class EgoState:
    speed_mps: float = 0.0
    a_long_mps2: float = 0.0
    yaw_rate_radps: float = 0.0
    curvature_inv_m: float = 0.0
    confidence: float = 0.0
    timestamp_ns: int = 0


class HighwayEgoStateEstimator:
    def __init__(self, smoothing: float = 0.25):
        self.smoothing = float(smoothing)
        self.state = EgoState()

    def update(
        self,
        model_ego_kinematics: torch.Tensor,
        previous_controls: Mapping[str, float] | None,
        dt_s: float,
        model_confidence: float,
        timestamp_ns: int = 0,
    ) -> EgoState:
        del previous_controls
        kin = model_ego_kinematics.detach().float().reshape(-1)
        if kin.numel() < 4:
            raise ValueError("model_ego_kinematics must contain [speed, a_long, yaw_rate, curvature].")
        alpha = max(0.0, min(1.0, self.smoothing * max(0.1, min(2.0, float(dt_s) * 20.0))))
        confidence = max(0.0, min(1.0, float(model_confidence)))
        measured = EgoState(
            speed_mps=max(0.0, float(kin[0].item())),
            a_long_mps2=float(kin[1].item()),
            yaw_rate_radps=float(kin[2].item()),
            curvature_inv_m=float(kin[3].item()),
            confidence=confidence,
            timestamp_ns=int(timestamp_ns),
        )
        prev = self.state
        self.state = EgoState(
            speed_mps=(1.0 - alpha) * prev.speed_mps + alpha * measured.speed_mps,
            a_long_mps2=(1.0 - alpha) * prev.a_long_mps2 + alpha * measured.a_long_mps2,
            yaw_rate_radps=(1.0 - alpha) * prev.yaw_rate_radps + alpha * measured.yaw_rate_radps,
            curvature_inv_m=(1.0 - alpha) * prev.curvature_inv_m + alpha * measured.curvature_inv_m,
            confidence=max(confidence, (1.0 - alpha) * prev.confidence),
            timestamp_ns=int(timestamp_ns),
        )
        return self.state
