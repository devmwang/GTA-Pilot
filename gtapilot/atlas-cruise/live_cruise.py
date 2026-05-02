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

from .config import AtlasCruiseConfig
from .controller import CruiseController
from .model import AtlasCruise
from .preprocess import preprocess_cruise_frame
from .trajectory_legalizer import CruiseTrajectoryLegalizer
from .visualization import CruiseDebugFrame, summarize_debug_frame


ATLAS_CRUISE_ACTIONS_CHANNEL = ChannelSpec(
    name="atlas_cruise.actions",
    port="55556",
    topic=b"atlas_cruise_actions",
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


@dataclass(slots=True)
class _TimedFrame:
    timestamp_ns: int
    rgb: torch.Tensor
    ui_mask: torch.Tensor
    is_repeat: bool


class ActionHistorySampler:
    def __init__(self, cfg: AtlasCruiseConfig):
        self.cfg = cfg
        self.history: collections.deque[_TimedAction] = collections.deque(
            maxlen=int(cfg.source_capture_hz * cfg.action_context_s * 2)
        )

    def append_packet(self, packet: ActionPacket, timestamp_ns: int) -> None:
        self.history.append(_TimedAction(timestamp_ns=int(timestamp_ns), vector=packet.vector))

    def tensors(self, now_ns: int, device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
        steps = self.cfg.action_history_steps
        stride_ns = int(round(1_000_000_000.0 / self.cfg.action_sample_hz))
        selected: list[list[float]] = []
        dts: list[float] = []
        previous_ts = int(now_ns) - steps * stride_ns
        for idx in range(steps):
            target_ts = int(now_ns) - (steps - idx - 1) * stride_ns
            packet = self._latest_before(target_ts)
            selected.append(packet.vector if packet is not None else [0.0] * self.cfg.action_dim)
            dts.append(max(0.0, (target_ts - previous_ts) / 1_000_000_000.0))
            previous_ts = target_ts
        return (
            torch.tensor(selected, device=device, dtype=torch.float32).unsqueeze(0),
            torch.tensor(dts, device=device, dtype=torch.float32).view(1, steps, 1),
        )

    def _latest_before(self, timestamp_ns: int) -> _TimedAction | None:
        for item in reversed(self.history):
            if item.timestamp_ns <= timestamp_ns:
                return item
        return None


class FrameHistorySampler:
    def __init__(self, cfg: AtlasCruiseConfig):
        self.cfg = cfg
        self.history: collections.deque[_TimedFrame] = collections.deque(
            maxlen=int(cfg.source_capture_hz * cfg.visual_context_s * 2) + cfg.num_visual_frames
        )

    def append(self, frame: _TimedFrame) -> None:
        self.history.append(frame)

    def tensors(self, now_ns: int, device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        offsets = self.cfg.resolved_visual_offsets_s()
        frames: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        freshness: list[list[float]] = []
        latest_age = 0.0
        for offset in offsets:
            desired_ns = int(now_ns + round(offset * 1_000_000_000.0))
            frame = self._latest_before(desired_ns) or (self.history[0] if self.history else None)
            if frame is None:
                frames.append(torch.zeros(3, self.cfg.input_h, self.cfg.input_w))
                masks.append(torch.zeros(1, self.cfg.input_h, self.cfg.input_w))
                freshness.append([self.cfg.max_frame_staleness_s, 1.0])
                continue
            age_s = max(0.0, float(desired_ns - frame.timestamp_ns) / 1_000_000_000.0)
            latest_age = max(latest_age, age_s)
            frames.append(frame.rgb)
            masks.append(frame.ui_mask)
            freshness.append([age_s, 1.0 if frame.is_repeat else 0.0])
        return (
            torch.stack(frames, dim=0).unsqueeze(0).to(device),
            torch.stack(masks, dim=0).unsqueeze(0).to(device),
            torch.tensor(freshness, device=device, dtype=torch.float32).unsqueeze(0),
            latest_age,
        )

    def _latest_before(self, timestamp_ns: int) -> _TimedFrame | None:
        for item in reversed(self.history):
            if item.timestamp_ns <= timestamp_ns:
                return item
        return None


class KeyboardActuator:
    VK_A = 0x41
    VK_D = 0x44
    VK_S = 0x53
    VK_W = 0x57
    KEYEVENTF_KEYUP = 0x0002

    def __init__(self, enabled: bool = False):
        self.enabled = bool(enabled)
        self._pressed: set[int] = set()

    def apply(self, steer: float, throttle: float, brake: float) -> None:
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
        for vk_code in sorted(self._pressed - desired):
            self._set_key(vk_code, pressed=False)
        for vk_code in sorted(desired - self._pressed):
            self._set_key(vk_code, pressed=True)
        self._pressed = desired

    def release_all(self) -> None:
        if hasattr(ctypes, "windll"):
            for vk_code in sorted(self._pressed):
                self._set_key(vk_code, pressed=False)
        self._pressed.clear()

    def _set_key(self, vk_code: int, *, pressed: bool) -> None:
        flags = 0 if pressed else self.KEYEVENTF_KEYUP
        ctypes.windll.user32.keybd_event(ctypes.c_ubyte(vk_code), 0, ctypes.c_ulong(flags), 0)


class AtlasCruiseRuntime:
    def __init__(
        self,
        cfg: AtlasCruiseConfig,
        *,
        device: str,
        checkpoint: str | None = None,
        publish_actions: bool = False,
        inject_keyboard: bool = False,
        debug_log_path: str | None = None,
    ):
        self.cfg = cfg
        self.device = torch.device(device)
        self.model = AtlasCruise(cfg).to(self.device).eval()
        if checkpoint is not None:
            payload = torch.load(checkpoint, map_location=self.device)
            self.model.load_state_dict(payload.get("model", payload))
        self.legalizer = CruiseTrajectoryLegalizer(cfg)
        self.controller = CruiseController(cfg)
        self.action_history = ActionHistorySampler(cfg)
        self.frame_history = FrameHistorySampler(cfg)
        self.keyboard_actuator = KeyboardActuator(enabled=inject_keyboard)
        self.action_publisher = (
            ChannelPublisher(ATLAS_CRUISE_ACTIONS_CHANNEL, source_name="atlas_cruise")
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
            self.action_history.append_packet(action_message.payload, action_message.envelope.message_timestamp_ns)
        now_ns = time.time_ns()
        raw_frame = torch.as_tensor(np.asarray(frame_message.payload))
        prep = preprocess_cruise_frame(raw_frame, self.cfg)
        capture_ts = int(frame_message.envelope.metadata.get("capture_timestamp_ns", frame_message.envelope.message_timestamp_ns))
        self.frame_history.append(
            _TimedFrame(
                timestamp_ns=capture_ts,
                rgb=prep.scene_rgb,
                ui_mask=prep.ui_mask,
                is_repeat=bool(frame_message.envelope.metadata.get("is_repeat", False)),
            )
        )
        rgb_recent, ui_mask_recent, frame_freshness, frame_age_s = self.frame_history.tensors(now_ns, self.device)
        actions_hist, dt_hist = self.action_history.tensors(now_ns, self.device)
        model_start = time.time()
        outputs = self.model(
            rgb_recent,
            actions_hist,
            dt_hist,
            ui_mask_recent=ui_mask_recent,
            frame_freshness=frame_freshness,
        )
        latency_s = time.time() - model_start
        legalized = self.legalizer.legalize(
            outputs["traj"][0],
            slow_or_brake_logit=outputs["slow_or_brake_logit"][0],
            fallback_logit=outputs["fallback_logit"][0],
            frame_stale_s=frame_age_s,
            caution=prep.ui_state.phone_present or prep.ui_state.popup_present,
        )
        control = self.controller.compute(
            legalized,
            ego_kinematics=outputs["ego_kinematics"][0],
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
                    active_device="atlas_cruise",
                ),
                timestamp_ns=now_ns,
                metadata={"fallback_active": control.fallback_active},
            )
        self.keyboard_actuator.apply(control.steer, control.throttle, control.brake)
        debug = CruiseDebugFrame(
            raw_rgb=np.asarray(frame_message.payload),
            sanitized_rgb=prep.scene_rgb.detach().cpu(),
            ui_mask=prep.ui_mask.detach().cpu(),
            ui_state=prep.ui_state,
            outputs=outputs,
            legalized=legalized,
            control=control,
            frame_freshness_s=frame_age_s,
            latency_s=latency_s,
        )
        summary = summarize_debug_frame(debug)
        summary["keyboard_injection_enabled"] = self.keyboard_actuator.enabled
        if self.debug_log_path is not None:
            with self.debug_log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(summary, sort_keys=True) + "\n")
        return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Atlas-Cruise live against GTA Pilot IPC streams.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--publish-actions", action="store_true")
    parser.add_argument("--inject-keyboard", action="store_true")
    parser.add_argument("--debug-log")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    cfg = AtlasCruiseConfig(pretrained_backbone=not args.no_pretrained)
    runtime = AtlasCruiseRuntime(
        cfg,
        device=args.device,
        checkpoint=args.checkpoint,
        publish_actions=args.publish_actions,
        inject_keyboard=args.inject_keyboard,
        debug_log_path=args.debug_log,
    )
    frame_sub = ChannelSubscriber(VISION_FRAMES_CHANNEL, latest_only=True)
    action_sub = ChannelSubscriber(INPUT_ACTIONS_CHANNEL, latest_only=False)
    try:
        while True:
            frame_message = frame_sub.receive(blocking=True, timeout_sec=1.0)
            if frame_message is None:
                continue
            summary = runtime.step(frame_message, action_sub.drain())
            print(json.dumps(summary, sort_keys=True))
    finally:
        runtime.close()
        frame_sub.close()
        action_sub.close()


if __name__ == "__main__":
    main()
