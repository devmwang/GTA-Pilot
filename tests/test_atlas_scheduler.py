from __future__ import annotations

from gtapilot.atlas import atlas_smoke_config
from gtapilot.atlas.scheduler import AtlasHeadScheduler


def test_drive_scheduler_decimates_heavy_heads() -> None:
    cfg = atlas_smoke_config()
    scheduler = AtlasHeadScheduler(cfg)

    step0 = scheduler.active_heads(0, mode="drive")
    step1 = scheduler.active_heads(1, mode="drive")
    step2 = scheduler.active_heads(2, mode="drive")

    assert "occupancy" in step0
    assert "occupancy" not in step1
    assert "actors" in step0
    assert "actors" in step2
