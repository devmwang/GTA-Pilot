from __future__ import annotations

import torch

from gtapilot.atlas import Atlas, atlas_smoke_config


def test_atlas_step_smoke() -> None:
    cfg = atlas_smoke_config()
    model = Atlas(cfg)
    state = model.init_state(batch_size=1)
    rgb = torch.randn(1, 3, cfg.image.raw_height, cfg.image.raw_width)
    action = torch.randn(1, cfg.action.action_dim)
    dt = torch.tensor([[0.1]])

    best_traj, aux, next_state = model.step(rgb, action, dt, state, mode="drive")
    expected_waypoints = int(round(cfg.planner.control_steps * cfg.planner.control_dt / cfg.planner.waypoint_dt))

    assert best_traj.shape == (1, expected_waypoints, 4)
    assert aux["traj"].shape[1] == cfg.planner.proposals
    assert next_state.static_grid.shape[1] == cfg.hidden_dim
    assert next_state.step_index == 1
