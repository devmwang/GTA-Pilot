from __future__ import annotations

from dataclasses import dataclass

import torch

from .trajectory_legalizer import LegalizedTrajectorySet


@dataclass(slots=True)
class CandidateSelection:
    selected_idx: int
    total_cost: torch.Tensor
    switched: bool
    reason: str


@dataclass(slots=True)
class CandidateSelectionConfig:
    w_model: float = 1.0
    w_legal: float = 1.0
    w_lane: float = 0.25
    w_lead: float = 0.35
    w_ui: float = 0.50
    w_switch: float = 0.30
    w_comfort: float = 0.20
    w_cmd: float = 0.50
    switch_margin: float = 0.35


class CandidateSelector:
    def __init__(self, cfg: CandidateSelectionConfig | None = None):
        self.cfg = cfg or CandidateSelectionConfig()

    def select(
        self,
        legalized: LegalizedTrajectorySet,
        candidate_logits: torch.Tensor,
        *,
        previous_idx: int | None = None,
        nav_cmd: torch.Tensor | None = None,
        ui_occluded: bool = False,
        lead_ttc_s: float | None = None,
    ) -> CandidateSelection:
        logits = candidate_logits.detach().float()
        if logits.ndim == 2:
            logits = logits[0]
        cost = self.cfg.w_model * (-logits)
        cost = cost + self.cfg.w_legal * legalized.cost.float()
        lateral_final = legalized.traj[:, -1, 1].abs().float()
        comfort = legalized.traj[:, :, 2].abs().mean(dim=-1).float()
        cost = cost + self.cfg.w_lane * lateral_final + self.cfg.w_comfort * comfort
        if ui_occluded:
            cost[2:4] += self.cfg.w_ui
        if lead_ttc_s is not None and lead_ttc_s < 2.0:
            cost[0] += self.cfg.w_lead
            cost[1] -= self.cfg.w_lead * 0.5
            cost[4] -= self.cfg.w_lead * 0.25
        if nav_cmd is not None:
            cmd = nav_cmd.detach().float().reshape(-1)
            if cmd.numel() >= 6:
                prefer_left = float(cmd[1].item() + cmd[3].item())
                prefer_right = float(cmd[2].item() + cmd[4].item())
                slow = float(cmd[5].item())
                cost[2] -= self.cfg.w_cmd * prefer_left
                cost[3] -= self.cfg.w_cmd * prefer_right
                cost[1] -= self.cfg.w_cmd * slow
        cost = torch.where(legalized.legal_mask, cost, torch.full_like(cost, 1e6))
        best_idx = int(torch.argmin(cost).item())
        switched = False
        reason = "lowest_cost_legal"
        if previous_idx is not None and 0 <= previous_idx < cost.numel():
            if bool(legalized.legal_mask[previous_idx].item()):
                delta = float(cost[previous_idx].item() - cost[best_idx].item())
                if best_idx != previous_idx and delta < self.cfg.switch_margin:
                    best_idx = int(previous_idx)
                    reason = "hysteresis_keep_previous"
                else:
                    switched = best_idx != previous_idx
        return CandidateSelection(
            selected_idx=best_idx,
            total_cost=cost,
            switched=switched,
            reason=reason,
        )
