from __future__ import annotations

import argparse
import collections
import ctypes
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from gtapilot.ipc.channel import ChannelPublisher, ChannelSubscriber
from gtapilot.ipc.channels import INPUT_ACTIONS_CHANNEL, VISION_FRAMES_CHANNEL, ActionPacket
from gtapilot.ipc.codecs import JsonDataclassCodec
from gtapilot.ipc.types import ChannelSpec

from .actuator_calibration import ActuatorCalibration
from .candidate_selector import CandidateSelector
from .config import AtlasHAConfig
from .controller import HighwayController
from .ego_predictor import latency_compensate_trajectory
from .ego_state import HighwayEgoStateEstimator
from .frame_freshness import FrameFreshnessMonitor
from .fsm import HighwaySupervisor, LaneChangeFSM
from .model import AtlasHA
from .timing import LatencyEstimator
from .trajectory_legalizer import TrajectoryLegalizer
from .ui import UIPreprocessor
from .vehicle_profile import build_vehicle_profile
from .visualization import HighwayDebugFrame, summarize_debug_frame


ATLAS_HA_ACTIONS_CHANNEL = ChannelSpec(
    name="atlas_ha.actions",
    port="55555",
    topic=b"atlas_ha_actions",
    codec=JsonDataclassCodec(ActionPacket),
    default_buffer_size=16,
    default_latest_only=True,
    default_sndhwm=8,
    default_rcvhwm=8,
)


@dataclass(slots=True)
class _TimedAction:
    timestamp_ns: int
    vector: list[float]


class ActionHistorySampler:
    def __init__(self, cfg: AtlasHAConfig):
        self.cfg = cfg
        self.history: collections.deque[_TimedAction] = collections.deque(
            maxlen=int(cfg.source_capture_hz * cfg.action_context_s * 2)
        )

    def append_packet(self, packet: ActionPacket, timestamp_ns: int) -> None:
        self.history.append(_TimedAction(timestamp_ns=int(timestamp_ns), vector=packet.vector))

    def tensors(self, now_ns: int, device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
        steps = self.cfg.action_history_steps
        stride_ns = int(round(1e9 / self.cfg.action_sample_hz))
        selected: list[list[float]] = []
        dts: list[float] = []
        last_ts = int(now_ns) - steps * stride_ns
        for idx in range(steps):
            target_ts = int(now_ns) - (steps - idx - 1) * stride_ns
            packet = self._latest_before(target_ts)
            selected.append(packet.vector if packet is not None else [0.0] * self.cfg.action_dim)
            dts.append(max(0.0, (target_ts - last_ts) / 1e9))
            last_ts = target_ts
        actions = torch.tensor(selected, device=device, dtype=torch.float32).unsqueeze(0)
        dt_hist = torch.tensor(dts, device=device, dtype=torch.float32).view(1, steps, 1)
        return actions, dt_hist

    def _latest_before(self, timestamp_ns: int) -> _TimedAction | None:
        for item in reversed(self.history):
            if item.timestamp_ns <= timestamp_ns:
                return item
        return None


class PhoneCloser:
    def __init__(self, enabled: bool = True, cooldown_s: float = 1.0):
        self.enabled = bool(enabled)
        self.cooldown_s = float(cooldown_s)
        self._last_sent_s = 0.0

    def maybe_close_phone(self, phone_present: bool) -> bool:
        if not self.enabled or not phone_present or not hasattr(ctypes, "windll"):
            return False
        now = time.monotonic()
        if now - self._last_sent_s < self.cooldown_s:
            return False
        self._last_sent_s = now
        return _send_right_click()


def _send_right_click() -> bool:
    if not hasattr(ctypes, "windll"):
        return False
    mouse_event = ctypes.windll.user32.mouse_event
    mouse_event.argtypes = [ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
    mouse_event.restype = None
    right_down = 0x0008
    right_up = 0x0010
    try:
        mouse_event(right_down, 0, 0, 0, 0)
        mouse_event(right_up, 0, 0, 0, 0)
    except Exception:
        return False
    return True


class KeyboardActuator:
    VK_A = 0x41
    VK_D = 0x44
    VK_S = 0x53
    VK_W = 0x57
    VK_SPACE = 0x20
    KEYEVENTF_KEYUP = 0x0002

    def __init__(self, enabled: bool = False):
        self.enabled = bool(enabled)
        self._pressed: set[int] = set()

    def apply(self, steer: float, throttle: float, brake: float, handbrake: float) -> None:
        if not self.enabled or not hasattr(ctypes, "windll"):
            return
        desired: set[int] = set()
        if throttle > 0.15:
            desired.add(self.VK_W)
        if brake > 0.15:
            desired.add(self.VK_S)
        if steer < -0.20:
            desired.add(self.VK_A)
        if steer > 0.20:
            desired.add(self.VK_D)
        if handbrake > 0.50:
            desired.add(self.VK_SPACE)
        for vk_code in sorted(self._pressed - desired):
            self._set_key(vk_code, pressed=False)
        for vk_code in sorted(desired - self._pressed):
            self._set_key(vk_code, pressed=True)
        self._pressed = desired

    def release_all(self) -> None:
        if not hasattr(ctypes, "windll"):
            self._pressed.clear()
            return
        for vk_code in sorted(self._pressed):
            self._set_key(vk_code, pressed=False)
        self._pressed.clear()

    def _set_key(self, vk_code: int, *, pressed: bool) -> None:
        flags = 0 if pressed else self.KEYEVENTF_KEYUP
        ctypes.windll.user32.keybd_event(ctypes.c_ubyte(vk_code), 0, ctypes.c_ulong(flags), 0)


class HighwayAssistRuntime:
    def __init__(
        self,
        cfg: AtlasHAConfig,
        *,
        device: str,
        publish_actions: bool = False,
        debug_log_path: str | None = None,
        close_phone: bool = True,
        inject_keyboard: bool = False,
    ):
        self.cfg = cfg
        self.device = torch.device(device)
        self.model = AtlasHA(cfg).to(self.device).eval()
        self.state = self.model.init_state(1, device=self.device)
        profile = build_vehicle_profile(cfg.default_vehicle_profile)
        self.legalizer = TrajectoryLegalizer(cfg, profile)
        self.selector = CandidateSelector()
        self.fsm = LaneChangeFSM()
        self.supervisor = HighwaySupervisor()
        self.controller = HighwayController(profile, ActuatorCalibration())
        self.keyboard_actuator = KeyboardActuator(enabled=inject_keyboard)
        self.ego_estimator = HighwayEgoStateEstimator()
        self.latency = LatencyEstimator(cfg.nominal_control_latency_s)
        self.ui = UIPreprocessor(cfg)
        self.phone_closer = PhoneCloser(enabled=close_phone)
        self.freshness = FrameFreshnessMonitor(cfg)
        self.action_history = ActionHistorySampler(cfg)
        self.action_publisher = (
            ChannelPublisher(ATLAS_HA_ACTIONS_CHANNEL, source_name="atlas_ha")
            if publish_actions
            else None
        )
        self.debug_log_path = Path(debug_log_path) if debug_log_path else None

    def close(self) -> None:
        self.keyboard_actuator.release_all()
        if self.action_publisher is not None:
            self.action_publisher.close()

    @torch.no_grad()
    def step(self, frame_message, action_messages: list) -> dict:
        for action_message in action_messages:
            self.action_history.append_packet(
                action_message.payload,
                action_message.envelope.message_timestamp_ns,
            )
        now_ns = time.time_ns()
        frame = torch.as_tensor(np.asarray(frame_message.payload))
        source_shape_ok = tuple(frame.shape[:2]) == self.cfg.expected_source_shape
        ui_out = self.ui.preprocess(frame)
        phone_close_sent = self.phone_closer.maybe_close_phone(ui_out.ui_state.phone_present)
        actions_hist, dt_hist = self.action_history.tensors(now_ns, self.device)
        freshness = self.freshness.update(
            source_frame_id=int(frame_message.envelope.metadata.get("capture_frame_id", frame_message.envelope.sequence_id)),
            capture_timestamp_ns=int(frame_message.envelope.metadata.get("capture_timestamp_ns", frame_message.envelope.message_timestamp_ns)),
            model_timestamp_ns=now_ns,
            is_repeat=bool(frame_message.envelope.metadata.get("is_repeat", False)),
        )
        model_start_ns = time.time_ns()
        outputs, next_state = self.model(
            ui_out.scene_rgb.unsqueeze(0).to(self.device),
            actions_hist,
            dt_hist,
            ui_mask=ui_out.ui_mask.unsqueeze(0).to(self.device),
            state=self.state,
        )
        model_output_ns = time.time_ns()
        ego_state = self.ego_estimator.update(
            outputs["ego_kinematics"][0],
            previous_controls=None,
            dt_s=1.0 / self.cfg.model_hz,
            model_confidence=float(outputs["lane_tracking_conf"][0, 0].detach().cpu()),
            timestamp_ns=now_ns,
        )
        provisional_mode = self.freshness.recommended_mode()
        lead_present = bool(torch.sigmoid(outputs["lead_present_logit"][0, 0]).item() > 0.5)
        control_send_estimate_ns = time.time_ns()
        timing_state = self.latency.update(
            capture_time_ns=int(frame_message.envelope.metadata.get("capture_timestamp_ns", frame_message.envelope.message_timestamp_ns)),
            model_start_time_ns=model_start_ns,
            model_output_time_ns=model_output_ns,
            control_send_time_ns=control_send_estimate_ns,
        )
        compensated_traj = latency_compensate_trajectory(
            outputs["traj_candidates"][0],
            ego_state,
            timing_state.estimated_capture_to_control_s,
        )
        legalized = self.legalizer.legalize(
            compensated_traj,
            outputs["candidate_logits"],
            supervisor_mode=provisional_mode,
            current_speed_mps=ego_state.speed_mps,
            lead_present=lead_present,
            lead_state=outputs["lead_state"][0],
            ui_occluded=ui_out.ui_state.phone_present or ui_out.ui_state.popup_present,
        )
        supervisor_state = self.supervisor.update(
            self.state.supervisor_state,
            outputs=outputs,
            freshness=freshness,
            legal_mask=legalized.legal_mask,
            ui_state=ui_out.ui_state,
            dt_s=1.0 / self.cfg.model_hz,
        )
        if not source_shape_ok and supervisor_state.mode == "NORMAL":
            supervisor_state.mode = "CAUTION"
            supervisor_state.reason = "source_resolution_mismatch"
        selection = self.selector.select(
            legalized,
            outputs["candidate_logits"],
            previous_idx=self.state.lane_change_fsm_state.last_accepted_candidate,
            ui_occluded=ui_out.ui_state.phone_present or ui_out.ui_state.popup_present,
            lead_ttc_s=float(outputs["lead_state"][0, 4].detach().cpu()),
        )
        lane_decision = self.fsm.update(
            self.state.lane_change_fsm_state,
            selection.selected_idx,
            outputs=outputs,
            legalized=legalized,
            nav_cmd=None,
            ui_state=ui_out.ui_state,
            dt_s=1.0 / self.cfg.model_hz,
        )
        legalized.selected_idx = lane_decision.accepted_idx
        control = self.controller.compute(
            legalized,
            ego_state=ego_state,
            supervisor_state=supervisor_state,
            selected_idx=lane_decision.accepted_idx,
            lead_present=lead_present,
            lead_state=outputs["lead_state"][0],
            dt_s=1.0 / self.cfg.controller_hz,
        )
        if self.action_publisher is not None:
            self.action_publisher.publish(
                ActionPacket(
                    steer=control.steer,
                    throttle=control.throttle,
                    brake=control.brake,
                    handbrake=control.handbrake,
                    reverse=control.reverse,
                    pilot_active=control.pilot_active,
                    active_device="atlas_ha",
                ),
                timestamp_ns=now_ns,
                metadata={"supervisor_mode": supervisor_state.mode},
            )
        self.keyboard_actuator.apply(
            control.steer,
            control.throttle,
            control.brake,
            control.handbrake,
        )
        self.state = next_state.with_runtime_state(
            ego_state=ego_state,
            timing_state=timing_state,
            frame_freshness_state=freshness,
            supervisor_state=supervisor_state,
            lane_change_fsm_state=lane_decision.state,
            ui_state=ui_out.ui_state,
        )
        debug = HighwayDebugFrame(
            raw_rgb=np.asarray(frame_message.payload),
            sanitized_rgb=ui_out.scene_rgb.detach().cpu(),
            ui_mask=ui_out.ui_mask.detach().cpu(),
            ui_state=ui_out.ui_state,
            outputs=outputs,
            legalized=legalized,
            control=control,
            supervisor_mode=supervisor_state.mode,
            takeover_reason=supervisor_state.reason,
            frame_freshness_s=freshness.frame_age_s,
            latency_s=timing_state.estimated_capture_to_control_s,
            reject_reasons=legalized.reject_reasons,
        )
        summary = summarize_debug_frame(debug)
        summary["source_shape_ok"] = source_shape_ok
        summary["phone_close_sent"] = phone_close_sent
        summary["keyboard_injection_enabled"] = self.keyboard_actuator.enabled
        if self.debug_log_path is not None:
            with self.debug_log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(summary, sort_keys=True) + "\n")
        return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Atlas-HA live against GTA Pilot IPC streams.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--publish-actions", action="store_true")
    parser.add_argument("--inject-keyboard", action="store_true")
    parser.add_argument("--no-close-phone", action="store_true")
    parser.add_argument("--debug-log")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    cfg = AtlasHAConfig(pretrained_backbone=not args.no_pretrained)
    runtime = HighwayAssistRuntime(
        cfg,
        device=args.device,
        publish_actions=args.publish_actions,
        debug_log_path=args.debug_log,
        close_phone=not args.no_close_phone,
        inject_keyboard=args.inject_keyboard,
    )
    frame_sub = ChannelSubscriber(VISION_FRAMES_CHANNEL, latest_only=True)
    action_sub = ChannelSubscriber(INPUT_ACTIONS_CHANNEL, latest_only=False)
    try:
        while True:
            frame_message = frame_sub.receive(blocking=True, timeout_sec=1.0)
            if frame_message is None:
                continue
            action_messages = action_sub.drain()
            summary = runtime.step(frame_message, action_messages)
            print(json.dumps(summary, sort_keys=True))
    finally:
        runtime.close()
        frame_sub.close()
        action_sub.close()


if __name__ == "__main__":
    main()
