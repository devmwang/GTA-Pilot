from __future__ import annotations

from gtapilot.atlas.train.common import run_stage_smoke


def test_stage_smoke_runs() -> None:
    for stage in ("stage1a", "stage2", "stage3", "stage4"):
        result = run_stage_smoke(stage)
        assert result["losses"]["total"].isfinite()
