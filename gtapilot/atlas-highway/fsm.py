from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from .frame_freshness import FrameFreshnessState
from .state import LaneChangeFSMState, SupervisorState
from .trajectory_legalizer import LegalizedTrajectorySet
from .ui import UIState


@dataclass(slots=True)
class LaneChangeDecision:
    accepted_idx: int
    state: LaneChangeFSMState
    vetoed: bool
    reason: str


class LaneChangeFSM:
    lane_change_candidates = {2: "LEFT", 3: "RIGHT"}

    def update(
        self,
        state: LaneChangeFSMState,
        selected_idx: int,
        *,
        outputs: dict[str, torch.Tensor | None],
        legalized: LegalizedTrajectorySet,
        nav_cmd: torch.Tensor | None,
        ui_state: UIState | None,
        dt_s: float,
        operational_need: bool = False,
    ) -> LaneChangeDecision:
        next_state = replace(state, state_age_s=state.state_age_s + max(0.0, float(dt_s)))
        if selected_idx not in self.lane_change_candidates:
            return LaneChangeDecision(
                accepted_idx=selected_idx,
                state=replace(next_state, state="KEEP_LANE", target_direction=None),
                vetoed=False,
                reason="keep_lane_candidate",
            )

        direction = self.lane_change_candidates[selected_idx]
        target_conf_idx = 0 if direction == "LEFT" else 2
        lane_conf = outputs.get("lane_conf")
        adjacent = outputs.get("adjacent_left" if direction == "LEFT" else "adjacent_right")
        lead_state = outputs.get("lead_state")

        reasons: list[str] = []
        nav_support = operational_need
        if nav_cmd is not None:
            cmd = nav_cmd.detach().float().reshape(-1)
            if cmd.numel() >= 6:
                nav_support = nav_support or (
                    direction == "LEFT" and float(cmd[1].item() + cmd[3].item()) > 0.35
                )
                nav_support = nav_support or (
                    direction == "RIGHT" and float(cmd[2].item() + cmd[4].item()) > 0.35
                )
        if not nav_support:
            reasons.append("no_nav_or_operational_need")
        if lane_conf is not None and float(lane_conf.reshape(-1, 3)[0, target_conf_idx].item()) < 0.55:
            reasons.append("target_lane_conf_low")
        if adjacent is not None:
            adj = adjacent.reshape(-1, 4)[0]
            if float(adj[0].item()) < 0.65:
                reasons.append("adjacent_available_low")
            if float(adj[1].item()) < 18.0:
                reasons.append("front_gap_low")
            if float(adj[2].item()) > 0.35:
                reasons.append("rear_risk_high")
            if float(adj[3].item()) < 0.45:
                reasons.append("adjacent_conf_low")
        if ui_state is not None and (ui_state.phone_present or ui_state.popup_present):
            reasons.append("ui_occlusion")
        if lead_state is not None and float(lead_state.reshape(-1, 5)[0, 4].item()) < 1.5:
            reasons.append("lead_ttc_low")
        if not bool(legalized.legal_mask[selected_idx].item()):
            reasons.append("legalizer_rejected")

        if reasons:
            fallback_idx = 1 if bool(legalized.legal_mask[1].item()) else 4
            return LaneChangeDecision(
                accepted_idx=fallback_idx,
                state=replace(next_state, state="ABORT", abort_reason=",".join(reasons)),
                vetoed=True,
                reason=",".join(reasons),
            )

        fsm_name = "CHANGE_LEFT" if direction == "LEFT" else "CHANGE_RIGHT"
        if state.state == "KEEP_LANE":
            fsm_name = "PREPARE_LEFT" if direction == "LEFT" else "PREPARE_RIGHT"
        elif state.state.startswith("PREPARE"):
            fsm_name = "CHANGE_LEFT" if direction == "LEFT" else "CHANGE_RIGHT"
        elif state.state.startswith("CHANGE") and next_state.state_age_s > 2.0:
            fsm_name = "SETTLE_LEFT" if direction == "LEFT" else "SETTLE_RIGHT"
        return LaneChangeDecision(
            accepted_idx=selected_idx,
            state=replace(
                next_state,
                state=fsm_name,
                target_direction=direction,
                last_accepted_candidate=selected_idx,
                abort_reason="",
            ),
            vetoed=False,
            reason="accepted",
        )


class HighwaySupervisor:
    def update(
        self,
        previous: SupervisorState,
        *,
        outputs: dict[str, torch.Tensor | None],
        freshness: FrameFreshnessState,
        legal_mask: torch.Tensor,
        ui_state: UIState | None,
        dt_s: float,
    ) -> SupervisorState:
        road_conf = _first_float(outputs.get("road_followable_conf"), default=0.0)
        lane_conf = _first_float(outputs.get("lane_tracking_conf"), default=0.0)
        takeover_prob = float(torch.sigmoid(outputs["takeover_required_logit"]).reshape(-1)[0].item()) if outputs.get("takeover_required_logit") is not None else 0.0
        scene_logits = outputs.get("scene_type_logits")
        scene_idx = int(torch.argmax(scene_logits.reshape(-1, 4)[0]).item()) if scene_logits is not None else 0
        ui_occluded = ui_state is not None and (ui_state.phone_present or ui_state.popup_present)
        dt_s = max(0.0, float(dt_s))
        reason = ""
        mode = "NORMAL"
        if freshness.frame_age_s > 0.30 or not bool(legal_mask.any().item()):
            mode = "MINIMUM_RISK"
            reason = "stale_frame_or_no_legal_candidate"
        elif (
            freshness.frame_age_s > 0.10
            or not freshness.is_fresh
            or ui_occluded
            or road_conf < 0.45
            or lane_conf < 0.45
            or scene_idx in {1, 2}
        ):
            mode = "CAUTION"
            reason = "reduced_confidence_or_caution_scene"
        takeover_condition = takeover_prob > 0.95 and road_conf < 0.15 and lane_conf < 0.15
        takeover_age = previous.takeover_age_s + dt_s if takeover_condition else 0.0
        if takeover_age >= 0.75 or (mode == "MINIMUM_RISK" and previous.minimum_risk_age_s > 2.0):
            mode = "TAKEOVER_REQUESTED"
            reason = "takeover_hysteresis_met"
        return SupervisorState(
            mode=mode,
            caution_age_s=previous.caution_age_s + dt_s if mode == "CAUTION" else 0.0,
            minimum_risk_age_s=previous.minimum_risk_age_s + dt_s if mode == "MINIMUM_RISK" else 0.0,
            takeover_age_s=takeover_age if mode != "TAKEOVER_REQUESTED" else max(takeover_age, previous.takeover_age_s + dt_s),
            takeover_requested=mode == "TAKEOVER_REQUESTED",
            reason=reason,
        )


def _first_float(value: torch.Tensor | None, default: float) -> float:
    if value is None:
        return default
    return float(value.detach().float().reshape(-1)[0].item())
