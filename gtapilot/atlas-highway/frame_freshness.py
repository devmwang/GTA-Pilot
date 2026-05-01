from __future__ import annotations

from dataclasses import dataclass

from .config import AtlasHAConfig


@dataclass(slots=True)
class FrameFreshnessState:
    source_frame_id: int = 0
    is_fresh: bool = False
    frame_age_s: float = 0.0
    repeated_frame_count: int = 0
    capture_to_model_latency_s: float = 0.0


class FrameFreshnessMonitor:
    def __init__(self, cfg: AtlasHAConfig):
        self.cfg = cfg
        self.state = FrameFreshnessState()
        self._last_source_frame_id: int | None = None

    def update(
        self,
        *,
        source_frame_id: int,
        capture_timestamp_ns: int,
        model_timestamp_ns: int,
        is_repeat: bool = False,
    ) -> FrameFreshnessState:
        source_frame_id = int(source_frame_id)
        same_frame = self._last_source_frame_id == source_frame_id
        is_fresh = not bool(is_repeat) and not same_frame
        repeated_count = self.state.repeated_frame_count + 1 if not is_fresh else 0
        age_s = max(0.0, (int(model_timestamp_ns) - int(capture_timestamp_ns)) / 1e9)
        self._last_source_frame_id = source_frame_id
        self.state = FrameFreshnessState(
            source_frame_id=source_frame_id,
            is_fresh=is_fresh,
            frame_age_s=age_s,
            repeated_frame_count=repeated_count,
            capture_to_model_latency_s=age_s,
        )
        return self.state

    def recommended_mode(self) -> str:
        if self.state.frame_age_s > self.cfg.max_frame_staleness_fallback_s:
            return "MINIMUM_RISK"
        if (
            self.state.frame_age_s > self.cfg.max_frame_staleness_caution_s
            or self.state.repeated_frame_count > 0
        ):
            return "CAUTION"
        return "NORMAL"
