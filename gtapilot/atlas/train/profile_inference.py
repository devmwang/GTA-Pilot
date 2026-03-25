from __future__ import annotations

import time

import torch

from .. import Atlas, atlas_smoke_config


def main() -> None:
    cfg = atlas_smoke_config()
    model = Atlas(cfg)
    state = model.init_state(batch_size=1)
    rgb = torch.randn(1, 3, cfg.image.raw_height, cfg.image.raw_width)
    action = torch.randn(1, cfg.action.action_dim)
    dt = torch.tensor([[0.1]])

    start = time.perf_counter()
    _, aux, _ = model.step(rgb, action, dt, state, mode="inspect")
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    print(f"atlas_step_ms={elapsed_ms:.2f}")
    print(f"decoded_keys={sorted(aux.keys())}")


if __name__ == "__main__":
    main()
