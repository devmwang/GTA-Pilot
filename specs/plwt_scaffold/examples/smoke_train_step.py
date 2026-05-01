from __future__ import annotations

import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plwt import PLWT, plwt_smoke_config
from plwt.losses import compute_stage_losses


def main() -> None:
    cfg = plwt_smoke_config()
    model = PLWT(cfg)
    B, T = 1, 2
    rgb = torch.randn(B, T, 3, cfg.image.raw_height, cfg.image.raw_width)
    actions = torch.randn(B, T, cfg.action.action_dim)
    dt = torch.full((B, T, 1), 0.1)

    batch = {
        "rgb_seq": rgb,
        "actions_seq": actions,
        "dt_seq": dt,
    }
    state = model.init_state(B)
    state.step_index = 1
    out = model.forward_train(batch, init_state=state, mode="drive")
    last = out["last"]

    targets = {
        "traj_target": torch.zeros(B, last["traj"].shape[2], 4),
        "teacher_cost": torch.randn(B, cfg.planner.proposals).abs(),
        "curvature_target": torch.zeros_like(last["curvature"]),
        "speed_target": torch.zeros_like(last["speed"]),
        "ego_target": torch.zeros(B, 7),
    }
    if "occ_state" in last:
        targets["occ_state_target"] = torch.zeros(
            B,
            cfg.occupancy.out_z,
            cfg.occupancy.out_y,
            cfg.occupancy.out_x,
            dtype=torch.long,
        )
        targets["occ_sem_target"] = torch.zeros(
            B,
            cfg.occupancy.out_z,
            cfg.occupancy.out_y,
            cfg.occupancy.out_x,
            dtype=torch.long,
        )
    if "bev_lite" in last:
        targets["bev_target"] = torch.zeros_like(last["bev_lite"])
        targets["provenance_target"] = torch.zeros(
            B,
            cfg.bev_lite.out_h,
            cfg.bev_lite.out_w,
            dtype=torch.long,
        )
    if "centerline" in last:
        targets["lane_centerline_target"] = torch.zeros_like(last["centerline"])
    if "map_poly" in last:
        targets["map_poly_target"] = torch.zeros_like(last["map_poly"])
    if "actor_box" in last:
        targets["actor_box_target"] = torch.zeros_like(last["actor_box"])

    losses = compute_stage_losses("stage4", last, targets)
    print("Available output keys:", sorted(last.keys()))
    print("Losses:")
    for k, v in losses.items():
        print(f"  {k}: {float(v):.6f}")


if __name__ == "__main__":
    main()
