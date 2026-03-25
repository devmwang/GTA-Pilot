from __future__ import annotations

from typing import Set

from .config import AtlasConfig


class AtlasHeadScheduler:
    def __init__(self, cfg: AtlasConfig):
        self.cfg = cfg

    def _cadence_map(self, mode: str) -> dict[str, int]:
        if mode == "inspect":
            return self.cfg.scheduler.inspect_mode_cadence
        return self.cfg.scheduler.drive_mode_cadence

    def active_heads(self, step_index: int, mode: str = "drive") -> Set[str]:
        cadence = self._cadence_map(mode)
        active = set()
        for name, every in cadence.items():
            every = max(1, int(every))
            if step_index % every == 0:
                active.add(name)
        return active


HeadScheduler = AtlasHeadScheduler
