from __future__ import annotations

import torch

from gtapilot.atlas import Atlas, atlas_smoke_config


def test_atlas_optional_route_and_reasoner_inputs() -> None:
    cfg = atlas_smoke_config()
    model = Atlas(cfg)
    state = model.init_state(batch_size=2)
    rgb = torch.randn(2, 3, cfg.image.raw_height, cfg.image.raw_width)
    action = torch.randn(2, cfg.action.action_dim)
    dt = torch.full((2, 1), 0.1)

    _, aux_no_optional, _ = model.step(rgb, action, dt, state, mode="drive")
    route = torch.randn(2, cfg.route_adapter.route_points, 3)
    nav_cmd = torch.randn(2, cfg.route_adapter.nav_cmd_dim)
    reasoner = torch.randn(2, cfg.reasoner_adapter.output_tokens, cfg.hidden_dim)
    _, aux_with_optional, _ = model.step(
        rgb,
        action,
        dt,
        state,
        route_polyline=route,
        nav_cmd=nav_cmd,
        reasoner_tok=reasoner,
        mode="drive",
    )

    assert aux_no_optional["traj"].shape == aux_with_optional["traj"].shape
    assert aux_with_optional["future_dyn"].shape[0] == 2
