from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from gtapilot.ipc.codecs import JsonDataclassCodec, RawRGBFrameCodec
from gtapilot.ipc.types import ChannelSpec


@dataclass(slots=True)
class ActionPacket:
    steer: float
    throttle: float
    brake: float
    handbrake: float
    reverse: float
    pilot_active: float
    raw_inputs: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "steer": float(self.steer),
            "throttle": float(self.throttle),
            "brake": float(self.brake),
            "handbrake": float(self.handbrake),
            "reverse": float(self.reverse),
            "pilot_active": float(self.pilot_active),
            "raw_inputs": dict(self.raw_inputs),
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
            raw_inputs=dict(payload.get("raw_inputs", {})),
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
