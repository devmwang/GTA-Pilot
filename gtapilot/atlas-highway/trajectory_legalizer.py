from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import AtlasHAConfig
from .speed_profiler import SpeedProfiler
from .trajectory_sanitizer import TrajectorySanitizer
from .vehicle_profile import VehicleDynamicsProfile


@dataclass(slots=True)
class LegalizedTrajectorySet:
    traj: torch.Tensor
    legal_mask: torch.Tensor
    cost: torch.Tensor
    selected_idx: int
    reject_reasons: list[str]
    speed_cap_curve: torch.Tensor
    speed_cap_lead: torch.Tensor
    speed_cap_ui: torch.Tensor
    speed_cap_odd: torch.Tensor
    speed_cap_final: torch.Tensor


class TrajectoryLegalizer:
    def __init__(self, cfg: AtlasHAConfig, vehicle_profile: VehicleDynamicsProfile):
        self.cfg = cfg
        self.profile = vehicle_profile
        self.sanitizer = TrajectorySanitizer(cfg, vehicle_profile)
        self.speed_profiler = SpeedProfiler(vehicle_profile)

    def _fallback_path(self, like: torch.Tensor, current_speed_mps: float) -> torch.Tensor:
        device, dtype = like.device, like.dtype
        long_m = torch.linspace(1.0, 45.0, self.cfg.num_traj_points, device=device, dtype=dtype)
        speed = torch.linspace(
            max(0.0, float(current_speed_mps)),
            0.0,
            self.cfg.num_traj_points,
            device=device,
            dtype=dtype,
        )
        path = torch.zeros(self.cfg.num_traj_points, 4, device=device, dtype=dtype)
        path[:, 0] = long_m
        path[:, 3] = speed
        return path

    def legalize(
        self,
        traj_candidates: torch.Tensor,
        candidate_logits: torch.Tensor,
        *,
        supervisor_mode: str,
        current_speed_mps: float = 0.0,
        lead_present: bool = False,
        lead_state: torch.Tensor | None = None,
        ui_occluded: bool = False,
        user_max_speed_mps: float = 38.0,
        rain: bool = False,
    ) -> LegalizedTrajectorySet:
        if traj_candidates.ndim == 4:
            if traj_candidates.shape[0] != 1:
                raise ValueError("Runtime legalizer expects one batch item at a time.")
            traj_candidates = traj_candidates[0]
        if candidate_logits.ndim == 2:
            if candidate_logits.shape[0] != 1:
                raise ValueError("Runtime legalizer expects one batch item at a time.")
            candidate_logits = candidate_logits[0]
        sanitized = self.sanitizer.sanitize(traj_candidates)
        legal_mask = sanitized.valid_mask.clone()
        speed_caps = self.speed_profiler.profile(
            sanitized.traj,
            sanitized.curvature,
            candidate_valid=legal_mask,
            supervisor_mode=supervisor_mode,
            lead_present=lead_present,
            lead_state=lead_state,
            ui_occluded=ui_occluded,
            current_speed_mps=current_speed_mps,
            user_max_speed_mps=user_max_speed_mps,
            rain=rain,
        )
        traj = sanitized.traj.clone()
        traj[..., 3] = speed_caps.speed_cap_final

        steer = torch.atan(self.profile.wheelbase_eff_m * sanitized.curvature)
        dsteer = torch.diff(steer, dim=-1).abs() / max(self.cfg.traj_dt_s, 1e-3)
        steer_bad = (steer.abs() > self.profile.steer_max_rad).any(dim=-1)
        steer_rate_bad = (dsteer > self.profile.steer_rate_max_radps).any(dim=-1)
        legal_mask = legal_mask & ~steer_bad & ~steer_rate_bad

        reject_reasons: list[str] = []
        for idx, reasons in enumerate(sanitized.reject_reasons):
            for reason in reasons:
                reject_reasons.append(f"candidate_{idx}:{reason}")
            if bool(steer_bad[idx].item()):
                reject_reasons.append(f"candidate_{idx}:reject_steer_limit")
            if bool(steer_rate_bad[idx].item()):
                reject_reasons.append(f"candidate_{idx}:reject_steer_rate_limit")

        legalizer_cost = sanitized.curvature.abs().mean(dim=-1) * 6.0
        speed_cost = (traj_candidates[..., 3].clamp_min(0.0) - traj[..., 3]).abs().mean(dim=-1) * 0.05
        cost = -candidate_logits.detach().float() + legalizer_cost.float() + speed_cost.float()
        cost = torch.where(legal_mask, cost, torch.full_like(cost, 1e6))

        if not bool(legal_mask.any().item()):
            fallback_idx = min(4, traj.shape[0] - 1)
            traj[fallback_idx] = self._fallback_path(traj[fallback_idx], current_speed_mps)
            legal_mask[fallback_idx] = True
            speed_caps.speed_cap_final[fallback_idx] = traj[fallback_idx, :, 3]
            speed_caps.speed_cap_curve[fallback_idx] = torch.full_like(traj[fallback_idx, :, 3], 80.0)
            speed_caps.speed_cap_lead[fallback_idx] = torch.full_like(traj[fallback_idx, :, 3], 80.0)
            speed_caps.speed_cap_ui[fallback_idx] = torch.full_like(traj[fallback_idx, :, 3], 80.0)
            speed_caps.speed_cap_odd[fallback_idx] = torch.full_like(traj[fallback_idx, :, 3], 6.0)
            cost[fallback_idx] = 0.0
            reject_reasons.append("all_candidates_invalid:CONTROL_FALLBACK_BRAKE_ALONG_CURRENT_PATH")

        selected_idx = int(torch.argmin(cost).item())
        return LegalizedTrajectorySet(
            traj=traj,
            legal_mask=legal_mask,
            cost=cost,
            selected_idx=selected_idx,
            reject_reasons=reject_reasons,
            speed_cap_curve=speed_caps.speed_cap_curve,
            speed_cap_lead=speed_caps.speed_cap_lead,
            speed_cap_ui=speed_caps.speed_cap_ui,
            speed_cap_odd=speed_caps.speed_cap_odd,
            speed_cap_final=speed_caps.speed_cap_final,
        )
