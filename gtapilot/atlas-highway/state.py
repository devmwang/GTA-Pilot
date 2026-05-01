from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from .config import AtlasHAConfig
from .ego_state import EgoState
from .frame_freshness import FrameFreshnessState
from .timing import TimingState


@dataclass(slots=True)
class KVCache:
    key: torch.Tensor
    value: torch.Tensor
    valid: torch.Tensor | None = None


@dataclass(slots=True)
class ActionRingBuffer:
    values: torch.Tensor
    valid: torch.Tensor

    @classmethod
    def init_empty(
        cls,
        batch_size: int,
        steps: int,
        action_dim: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> "ActionRingBuffer":
        return cls(
            values=torch.zeros(batch_size, steps, action_dim, device=device, dtype=dtype),
            valid=torch.zeros(batch_size, steps, device=device, dtype=torch.bool),
        )

    def append(self, value: torch.Tensor) -> "ActionRingBuffer":
        value = value.to(device=self.values.device, dtype=self.values.dtype)
        if value.ndim == 2:
            value = value[:, None]
        if value.shape[-1] != self.values.shape[-1]:
            raise ValueError("Action vector dimension mismatch.")
        values = torch.cat([self.values, value], dim=1)[:, -self.values.shape[1] :]
        valid_new = torch.ones(value.shape[0], value.shape[1], device=self.valid.device, dtype=torch.bool)
        valid = torch.cat([self.valid, valid_new], dim=1)[:, -self.valid.shape[1] :]
        return ActionRingBuffer(values=values, valid=valid)


@dataclass(slots=True)
class DtRingBuffer:
    values: torch.Tensor
    valid: torch.Tensor

    @classmethod
    def init_empty(
        cls,
        batch_size: int,
        steps: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> "DtRingBuffer":
        return cls(
            values=torch.zeros(batch_size, steps, 1, device=device, dtype=dtype),
            valid=torch.zeros(batch_size, steps, device=device, dtype=torch.bool),
        )

    def append(self, value: torch.Tensor) -> "DtRingBuffer":
        value = value.to(device=self.values.device, dtype=self.values.dtype)
        if value.ndim == 1:
            value = value[:, None, None]
        elif value.ndim == 2:
            value = value[:, None]
        values = torch.cat([self.values, value], dim=1)[:, -self.values.shape[1] :]
        valid_new = torch.ones(value.shape[0], value.shape[1], device=self.valid.device, dtype=torch.bool)
        valid = torch.cat([self.valid, valid_new], dim=1)[:, -self.valid.shape[1] :]
        return DtRingBuffer(values=values, valid=valid)


@dataclass(slots=True)
class SupervisorState:
    mode: str = "NORMAL"
    caution_age_s: float = 0.0
    minimum_risk_age_s: float = 0.0
    takeover_age_s: float = 0.0
    takeover_requested: bool = False
    reason: str = ""


@dataclass(slots=True)
class LaneChangeFSMState:
    state: str = "KEEP_LANE"
    state_age_s: float = 0.0
    target_direction: str | None = None
    last_accepted_candidate: int = 0
    abort_reason: str = ""


@dataclass(slots=True)
class AtlasHAState:
    full_token_cache: torch.Tensor
    full_kv_cache: list[KVCache]
    summary_cache: torch.Tensor
    action_ring: ActionRingBuffer
    dt_ring: DtRingBuffer

    previous_selected_traj: torch.Tensor | None
    previous_stable_lane: torch.Tensor | None
    previous_stable_path: torch.Tensor | None
    lane_memory_age_s: float

    ego_state: EgoState
    timing_state: TimingState
    frame_freshness_state: FrameFreshnessState
    supervisor_state: SupervisorState
    lane_change_fsm_state: LaneChangeFSMState
    ui_state: object | None

    full_valid: torch.Tensor
    summary_valid: torch.Tensor
    step_index: int = 0

    @classmethod
    def init_empty(
        cls,
        cfg: AtlasHAConfig,
        batch_size: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "AtlasHAState":
        full_steps = cfg.full_context_frames
        summary_steps = cfg.summary_context_steps
        action_steps = cfg.action_history_steps
        return cls(
            full_token_cache=torch.zeros(
                batch_size,
                full_steps,
                cfg.tokens_per_frame,
                cfg.hidden_dim,
                device=device,
                dtype=dtype,
            ),
            full_kv_cache=[],
            summary_cache=torch.zeros(
                batch_size,
                summary_steps,
                cfg.hidden_dim,
                device=device,
                dtype=dtype,
            ),
            action_ring=ActionRingBuffer.init_empty(
                batch_size,
                action_steps,
                cfg.action_dim,
                device=device,
                dtype=dtype,
            ),
            dt_ring=DtRingBuffer.init_empty(
                batch_size,
                action_steps,
                device=device,
                dtype=dtype,
            ),
            previous_selected_traj=None,
            previous_stable_lane=None,
            previous_stable_path=None,
            lane_memory_age_s=0.0,
            ego_state=EgoState(),
            timing_state=TimingState(),
            frame_freshness_state=FrameFreshnessState(),
            supervisor_state=SupervisorState(),
            lane_change_fsm_state=LaneChangeFSMState(),
            ui_state=None,
            full_valid=torch.zeros(batch_size, full_steps, device=device, dtype=torch.bool),
            summary_valid=torch.zeros(batch_size, summary_steps, device=device, dtype=torch.bool),
            step_index=0,
        )

    def with_runtime_state(
        self,
        *,
        ego_state: EgoState | None = None,
        timing_state: TimingState | None = None,
        frame_freshness_state: FrameFreshnessState | None = None,
        supervisor_state: SupervisorState | None = None,
        lane_change_fsm_state: LaneChangeFSMState | None = None,
        ui_state: object | None = None,
    ) -> "AtlasHAState":
        return replace(
            self,
            ego_state=self.ego_state if ego_state is None else ego_state,
            timing_state=self.timing_state if timing_state is None else timing_state,
            frame_freshness_state=(
                self.frame_freshness_state
                if frame_freshness_state is None
                else frame_freshness_state
            ),
            supervisor_state=(
                self.supervisor_state if supervisor_state is None else supervisor_state
            ),
            lane_change_fsm_state=(
                self.lane_change_fsm_state
                if lane_change_fsm_state is None
                else lane_change_fsm_state
            ),
            ui_state=self.ui_state if ui_state is None else ui_state,
        )
