from __future__ import annotations

import torch

from gtapilot.atlas import Atlas, atlas_smoke_config


def test_state_update_advances_history() -> None:
    cfg = atlas_smoke_config()
    model = Atlas(cfg)
    state = model.init_state(batch_size=1)
    rgb = torch.randn(1, 3, cfg.image.raw_height, cfg.image.raw_width)
    action = torch.randn(1, cfg.action.action_dim)
    dt = torch.tensor([[0.1]])

    _, _, next_state = model.step(rgb, action, dt, state, mode="drive")

    assert next_state.cam_token_cache.shape[1] == cfg.temporal.num_frames - 1
    assert torch.count_nonzero(next_state.action_buffer).item() > 0
    assert next_state.pose_belief.shape[-1] == 10
