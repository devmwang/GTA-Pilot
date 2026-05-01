from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .vehicle_profile import VehicleDynamicsProfile


@dataclass(slots=True)
class SpeedCapSet:
    speed_cap_curve: torch.Tensor
    speed_cap_lead: torch.Tensor
    speed_cap_ui: torch.Tensor
    speed_cap_odd: torch.Tensor
    speed_cap_final: torch.Tensor


class SpeedProfiler:
    def __init__(self, profile: VehicleDynamicsProfile):
        self.vehicle_profile = profile

    def curve_speed_cap(
        self,
        curvature: torch.Tensor,
        *,
        caution: bool = False,
        rain: bool = False,
    ) -> torch.Tensor:
        limit = self.vehicle_profile.a_lat_comfort_mps2
        if caution:
            limit *= self.vehicle_profile.caution_lat_multiplier
        if rain:
            limit *= self.vehicle_profile.rain_lat_multiplier
        return torch.sqrt(torch.tensor(limit, device=curvature.device, dtype=curvature.dtype) / (curvature.abs() + 1e-4))

    def lead_speed_cap(
        self,
        traj: torch.Tensor,
        lead_present: bool,
        lead_state: torch.Tensor | None,
    ) -> torch.Tensor:
        cap = torch.full_like(traj[..., 3], 80.0)
        if not lead_present or lead_state is None:
            return cap
        lead = lead_state.detach().float().reshape(-1)
        if lead.numel() < 5:
            return cap
        distance = max(0.0, float(lead[2].item()))
        rel_speed = float(lead[3].item())
        ttc = float(lead[4].item())
        if distance < 8.0 or ttc < 1.2:
            cap[:] = torch.minimum(cap, torch.full_like(cap, 2.0))
        elif distance < 25.0:
            cap[:] = torch.minimum(cap, torch.full_like(cap, max(4.0, 12.0 + rel_speed)))
        elif distance < 45.0:
            cap[:] = torch.minimum(cap, torch.full_like(cap, max(8.0, 20.0 + rel_speed)))
        return cap

    def ui_speed_cap(self, traj: torch.Tensor, ui_occluded: bool) -> torch.Tensor:
        cap_value = 13.0 if ui_occluded else 80.0
        return torch.full_like(traj[..., 3], cap_value)

    def odd_speed_cap(self, traj: torch.Tensor, supervisor_mode: str) -> torch.Tensor:
        if supervisor_mode == "MINIMUM_RISK":
            cap_value = 6.0
        elif supervisor_mode == "TAKEOVER_REQUESTED":
            cap_value = 4.0
        elif supervisor_mode == "CAUTION":
            cap_value = 16.0
        else:
            cap_value = 38.0
        return torch.full_like(traj[..., 3], cap_value)

    def apply_accel_limits(
        self,
        speed: torch.Tensor,
        long_m: torch.Tensor,
        current_speed_mps: float,
        *,
        hard_brake: bool = False,
    ) -> torch.Tensor:
        out = speed.clone().clamp_min(0.0)
        if out.numel() == 0:
            return out
        out[..., 0] = torch.minimum(
            out[..., 0],
            torch.tensor(max(0.0, current_speed_mps), device=out.device, dtype=out.dtype),
        )
        ds = torch.diff(long_m, dim=-1).abs().clamp_min(0.25)
        accel = self.vehicle_profile.accel_max_mps2
        decel = (
            self.vehicle_profile.decel_hard_mps2
            if hard_brake
            else self.vehicle_profile.decel_comfort_mps2
        )
        for idx in range(1, out.shape[-1]):
            cap = torch.sqrt(out[..., idx - 1] ** 2 + 2.0 * accel * ds[..., idx - 1])
            out[..., idx] = torch.minimum(out[..., idx], cap)
        for idx in range(out.shape[-1] - 2, -1, -1):
            cap = torch.sqrt(out[..., idx + 1] ** 2 + 2.0 * decel * ds[..., idx])
            out[..., idx] = torch.minimum(out[..., idx], cap)
        if out.shape[-1] >= 3:
            out[..., 1:-1] = 0.25 * out[..., :-2] + 0.5 * out[..., 1:-1] + 0.25 * out[..., 2:]
        return out

    def profile(
        self,
        traj: torch.Tensor,
        curvature: torch.Tensor,
        *,
        candidate_valid: torch.Tensor,
        supervisor_mode: str,
        lead_present: bool,
        lead_state: torch.Tensor | None,
        ui_occluded: bool,
        current_speed_mps: float,
        user_max_speed_mps: float = math.inf,
        rain: bool = False,
    ) -> SpeedCapSet:
        caution = supervisor_mode in {"CAUTION", "MINIMUM_RISK", "TAKEOVER_REQUESTED"}
        curve = self.curve_speed_cap(curvature, caution=caution, rain=rain)
        lead = self.lead_speed_cap(traj, lead_present, lead_state)
        ui = self.ui_speed_cap(traj, ui_occluded)
        odd = self.odd_speed_cap(traj, supervisor_mode)
        user = torch.full_like(traj[..., 3], float(user_max_speed_mps))
        final = torch.minimum(torch.minimum(torch.minimum(curve, lead), torch.minimum(ui, odd)), user)
        final = torch.minimum(final, traj[..., 3].clamp_min(0.0))
        final = self.apply_accel_limits(
            final,
            traj[..., 0],
            current_speed_mps,
            hard_brake=supervisor_mode in {"MINIMUM_RISK", "TAKEOVER_REQUESTED"},
        )
        final = torch.where(candidate_valid[:, None], final, torch.zeros_like(final))
        return SpeedCapSet(
            speed_cap_curve=curve,
            speed_cap_lead=lead,
            speed_cap_ui=ui,
            speed_cap_odd=odd,
            speed_cap_final=final,
        )
