from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import math


@dataclass(slots=True)
class ActuatorCalibration:
    steer_deadzone: float = 0.02
    steer_gain: float = 1.0
    steer_nonlinearity: float = 1.25

    throttle_deadzone: float = 0.02
    throttle_gain: float = 1.0

    brake_deadzone: float = 0.02
    brake_gain: float = 1.0

    input_lag_s: float = 0.10

    def map_steer(self, steer: float) -> float:
        value = _apply_deadzone(float(steer), self.steer_deadzone)
        sign = -1.0 if value < 0.0 else 1.0
        value = sign * (abs(value) ** self.steer_nonlinearity) * self.steer_gain
        return max(-1.0, min(1.0, value))

    def map_throttle(self, throttle: float) -> float:
        value = _apply_deadzone(float(throttle), self.throttle_deadzone)
        return max(0.0, min(1.0, value * self.throttle_gain))

    def map_brake(self, brake: float) -> float:
        value = _apply_deadzone(float(brake), self.brake_deadzone)
        return max(0.0, min(1.0, value * self.brake_gain))

    def to_json_file(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def from_json_file(cls, path: str | Path) -> "ActuatorCalibration":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def _apply_deadzone(value: float, deadzone: float) -> float:
    if abs(value) <= deadzone:
        return 0.0
    if value > 0.0:
        return (value - deadzone) / max(1e-6, 1.0 - deadzone)
    return (value + deadzone) / max(1e-6, 1.0 - deadzone)


def estimate_steer_gain(radius_m: float, speed_mps: float, command: float) -> float:
    if radius_m <= 0.0 or abs(command) <= 1e-6:
        return 1.0
    curvature = 1.0 / radius_m
    nominal = math.atan(curvature * 2.6) / 0.55
    del speed_mps
    return max(0.1, min(4.0, nominal / abs(command)))
