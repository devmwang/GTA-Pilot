from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TimingState:
    capture_time_ns: int = 0
    model_start_time_ns: int = 0
    model_output_time_ns: int = 0
    control_send_time_ns: int = 0
    estimated_apply_time_ns: int = 0
    estimated_capture_to_control_s: float = 0.0


class LatencyEstimator:
    def __init__(self, nominal_control_latency_s: float = 0.10, smoothing: float = 0.2):
        self.nominal_control_latency_s = float(nominal_control_latency_s)
        self.smoothing = float(smoothing)
        self._estimate_s = float(nominal_control_latency_s)

    @property
    def estimate_s(self) -> float:
        return self._estimate_s

    def update(
        self,
        *,
        capture_time_ns: int,
        model_start_time_ns: int,
        model_output_time_ns: int,
        control_send_time_ns: int,
    ) -> TimingState:
        observed_s = max(0.0, (int(control_send_time_ns) - int(capture_time_ns)) / 1e9)
        observed_s += self.nominal_control_latency_s
        self._estimate_s = (
            (1.0 - self.smoothing) * self._estimate_s + self.smoothing * observed_s
        )
        estimated_apply_time_ns = int(int(capture_time_ns) + self._estimate_s * 1e9)
        return TimingState(
            capture_time_ns=int(capture_time_ns),
            model_start_time_ns=int(model_start_time_ns),
            model_output_time_ns=int(model_output_time_ns),
            control_send_time_ns=int(control_send_time_ns),
            estimated_apply_time_ns=estimated_apply_time_ns,
            estimated_capture_to_control_s=float(self._estimate_s),
        )
