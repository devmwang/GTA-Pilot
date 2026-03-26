from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from gtapilot.ipc.codecs import JsonDataclassCodec, RawRGBFrameCodec
from gtapilot.ipc.types import ChannelMessage, ChannelSpec

ACTION_KEYS = (
    "steer",
    "throttle",
    "brake",
    "handbrake",
    "reverse",
    "pilot_active",
)


def _zero_action_values() -> dict[str, float]:
    return {key: 0.0 for key in ACTION_KEYS}


def _normalize_device_actions(
    payload: dict[str, Any] | None,
) -> dict[str, dict[str, float]]:
    normalized_payload = dict(payload or {})
    normalized_actions: dict[str, dict[str, float]] = {}
    for device_name in ("keyboard", "xinput_controller"):
        device_payload = dict(normalized_payload.get(device_name, {}))
        normalized_actions[device_name] = {
            key: float(device_payload.get(key, 0.0)) for key in ACTION_KEYS
        }
    return normalized_actions


def _normalize_device_inputs(payload: dict[str, Any] | None) -> dict[str, Any]:
    normalized_payload = copy.deepcopy(dict(payload or {}))
    normalized_payload.setdefault(
        "keyboard",
        {
            "connected": True,
            "driving_active": False,
            "keys": {
                "w": False,
                "a": False,
                "s": False,
                "d": False,
                "up": False,
                "down": False,
                "left": False,
                "right": False,
                "space": False,
            },
        },
    )
    normalized_payload.setdefault(
        "xinput_controller",
        {
            "connected": False,
            "controller_index": None,
            "packet_number": None,
            "driving_active": False,
            "buttons": {
                "a": False,
                "b": False,
                "x": False,
                "y": False,
                "lb": False,
                "rb": False,
                "back": False,
                "start": False,
                "left_thumb": False,
                "right_thumb": False,
                "dpad_up": False,
                "dpad_down": False,
                "dpad_left": False,
                "dpad_right": False,
            },
            "triggers": {"left": 0.0, "right": 0.0},
            "sticks": {
                "left_x": 0.0,
                "left_y": 0.0,
                "right_x": 0.0,
                "right_y": 0.0,
            },
        },
    )
    return normalized_payload


@dataclass(slots=True)
class ActionPacket:
    steer: float
    throttle: float
    brake: float
    handbrake: float
    reverse: float
    pilot_active: float
    active_device: str = "none"
    device_actions: dict[str, dict[str, float]] = field(default_factory=dict)
    device_inputs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "steer": float(self.steer),
            "throttle": float(self.throttle),
            "brake": float(self.brake),
            "handbrake": float(self.handbrake),
            "reverse": float(self.reverse),
            "pilot_active": float(self.pilot_active),
            "active_device": str(self.active_device),
            "device_actions": copy.deepcopy(
                _normalize_device_actions(self.device_actions)
            ),
            "device_inputs": copy.deepcopy(_normalize_device_inputs(self.device_inputs)),
        }

    @property
    def vector(self) -> list[float]:
        return [
            float(self.steer),
            float(self.throttle),
            float(self.brake),
            float(self.handbrake),
            float(self.reverse),
            float(self.pilot_active),
        ]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ActionPacket":
        return cls(
            steer=float(payload.get("steer", 0.0)),
            throttle=float(payload.get("throttle", 0.0)),
            brake=float(payload.get("brake", 0.0)),
            handbrake=float(payload.get("handbrake", 0.0)),
            reverse=float(payload.get("reverse", 0.0)),
            pilot_active=float(payload.get("pilot_active", 0.0)),
            active_device=str(payload.get("active_device", "none")),
            device_actions=_normalize_device_actions(payload.get("device_actions")),
            device_inputs=_normalize_device_inputs(payload.get("device_inputs")),
        )


def frame_capture_timestamp_ns(frame_message: ChannelMessage[Any]) -> int:
    return int(
        frame_message.envelope.metadata.get(
            "capture_timestamp_ns",
            frame_message.envelope.message_timestamp_ns,
        )
    )


VISION_FRAMES_CHANNEL = ChannelSpec(
    name="vision.frames",
    port="55550",
    topic=b"frames",
    codec=RawRGBFrameCodec(),
    default_buffer_size=10,
    default_latest_only=False,
    default_sndhwm=1,
    default_rcvhwm=1,
)


INPUT_ACTIONS_CHANNEL = ChannelSpec(
    name="input.actions",
    port="55552",
    topic=b"actions",
    codec=JsonDataclassCodec(ActionPacket),
    default_buffer_size=256,
    default_latest_only=False,
    default_sndhwm=256,
    default_rcvhwm=32,
)
