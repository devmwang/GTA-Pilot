from __future__ import annotations

import torch

from gtapilot.atlas import Atlas, atlas_smoke_config
from gtapilot.atlas.vision import build_camera_encoder


def test_vision_backends_share_encoder_interface() -> None:
    expected_keys = {
        "detail_4x",
        "detail_8x",
        "ctx_8x",
        "ctx_16x",
        "ctx_32x",
        "ctx_64x",
        "f8",
        "f16",
        "f32",
        "f64",
    }
    rgb = None
    for encoder_type in ("dualpath_hvt_bifpn", "regnet_bifpn_baseline"):
        cfg = atlas_smoke_config()
        cfg.vision.encoder_type = encoder_type
        encoder = build_camera_encoder(cfg)
        if rgb is None:
            rgb = torch.randn(2, 3, cfg.image.raw_height, cfg.image.raw_width)
        features, cam_tokens = encoder(rgb)
        assert expected_keys.issubset(features.keys())
        assert cam_tokens.shape == (2, cfg.vision.cam_tokens_per_frame, cfg.hidden_dim)


def test_atlas_step_supports_regnet_baseline_backend() -> None:
    cfg = atlas_smoke_config()
    cfg.vision.encoder_type = "regnet_bifpn_baseline"
    model = Atlas(cfg)
    state = model.init_state(batch_size=1)
    rgb = torch.randn(1, 3, cfg.image.raw_height, cfg.image.raw_width)
    action = torch.randn(1, cfg.action.action_dim)
    dt = torch.tensor([[0.1]])

    best_traj, aux, next_state = model.step(rgb, action, dt, state, mode="drive")

    assert best_traj.shape[0] == 1
    assert aux["f8"].shape[-2:] == aux["ctx_8x"].shape[-2:]
    assert next_state.step_index == 1
