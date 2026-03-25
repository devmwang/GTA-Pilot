from __future__ import annotations

import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plwt import PLWT, plwt_smoke_config


def main() -> None:
    cfg = plwt_smoke_config()
    model = PLWT(cfg)
    state = model.init_state(batch_size=1)
    rgb = torch.randn(1, 3, cfg.image.raw_height, cfg.image.raw_width)
    action = torch.randn(1, cfg.action.action_dim)
    dt = torch.tensor([[0.1]])

    state.step_index = 1
    out, state = model.step(rgb, action, dt, state, mode="drive")

    assert out["best_traj"].shape[0] == 1
    assert out["best_traj"].shape[-1] == 4
    assert state.static_grid.shape[1] == cfg.hidden_dim
    assert state.step_index == 1
    print("smoke_test passed")


if __name__ == "__main__":
    main()
