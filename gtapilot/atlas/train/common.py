from __future__ import annotations

from time import perf_counter
from typing import Any

import torch

from .. import Atlas, atlas_smoke_config, atlas_t1080_priv_config
from ..losses import compute_stage_losses
from ..teacher.privileged_teacher_model import PrivilegedTeacherModel


def make_synthetic_batch(cfg, batch_size: int = 2) -> dict[str, torch.Tensor]:
    steps = cfg.temporal.num_frames
    rgb = torch.randn(
        batch_size,
        steps,
        cfg.image.channels,
        cfg.image.raw_height,
        cfg.image.raw_width,
    )
    actions = torch.randn(batch_size, steps, cfg.action.action_dim)
    dt = torch.full((batch_size, steps, 1), 0.1)
    route = torch.randn(batch_size, cfg.route_adapter.route_points, 3)
    nav_cmd = torch.randn(batch_size, cfg.route_adapter.nav_cmd_dim)
    reasoner = torch.randn(batch_size, steps, cfg.reasoner_adapter.output_tokens, cfg.hidden_dim)
    return {
        "rgb_past": rgb,
        "actions_hist": actions,
        "dt_hist": dt,
        "route_polyline": route,
        "nav_cmd": nav_cmd,
        "reasoner_tok": reasoner,
    }


def make_synthetic_targets(outputs: dict[str, Any]) -> dict[str, torch.Tensor]:
    last = outputs["last"]
    batch_size = last["best_traj"].shape[0]
    targets: dict[str, torch.Tensor] = {
        "traj_target": last["best_traj"].detach().clone(),
        "teacher_cost": torch.randn(batch_size, last["score"].shape[1]),
        "ego_target": last["ego_out"].detach().clone() if "ego_out" in last else torch.randn(batch_size, 7),
        "depth_target": last["depth_mean"].detach().clone(),
        "future_dyn_target": last["future_dyn"].detach().clone(),
    }
    if "occ_state" in last:
        targets["occ_state_target"] = torch.zeros(
            last["occ_state"].shape[0],
            last["occ_state"].shape[2],
            last["occ_state"].shape[3],
            last["occ_state"].shape[4],
            dtype=torch.long,
        )
    if "occ_sem" in last:
        targets["occ_sem_target"] = torch.zeros(
            last["occ_sem"].shape[0],
            last["occ_sem"].shape[2],
            last["occ_sem"].shape[3],
            last["occ_sem"].shape[4],
            dtype=torch.long,
        )
    if "bev_lite" in last:
        targets["bev_target"] = torch.zeros_like(last["bev_lite"])
    if "provenance" in last:
        targets["provenance_target"] = torch.zeros(
            last["provenance"].shape[0],
            last["provenance"].shape[2],
            last["provenance"].shape[3],
            dtype=torch.long,
        )
    if "centerline" in last:
        targets["centerline"] = last["centerline"].detach().clone()
        targets["left_boundary"] = last["left_boundary"].detach().clone()
        targets["right_boundary"] = last["right_boundary"].detach().clone()
        targets["lane_sem_cls"] = torch.zeros(
            last["lane_sem_cls"].shape[0],
            last["lane_sem_cls"].shape[1],
            dtype=torch.long,
        )
    if "map_poly" in last:
        targets["map_poly"] = last["map_poly"].detach().clone()
    if "actor_box" in last:
        targets["actor_box"] = last["actor_box"].detach().clone()
        targets["actor_vel"] = last["actor_vel"].detach().clone()
        targets["actor_cls"] = torch.zeros(
            last["actor_cls"].shape[0],
            last["actor_cls"].shape[1],
            dtype=torch.long,
        )
    if "curvature" in last:
        targets["curvature_target"] = last["curvature"].detach().clone()
        targets["speed_target"] = last["speed"].detach().clone()
    return targets


def build_model(stage: str):
    if stage == "teacher":
        cfg = atlas_t1080_priv_config()
        cfg.image.raw_height = 128
        cfg.image.raw_width = 192
        cfg.image.padded_height = 128
        cfg.image.padded_width = 192
        cfg.vision.cam_tokens_per_frame = 16
        cfg.temporal.num_frames = 4
        cfg.action.history_len = 4
        cfg.world.static_grid_h = 6
        cfg.world.static_grid_w = 4
        cfg.world.dynamic_slots = 8
        cfg.world.lane_slots = 6
        cfg.world.map_elem_slots = 4
        cfg.planner.control_steps = 6
        cfg.planner.proposals = 4
        return PrivilegedTeacherModel(cfg), cfg
    cfg = atlas_smoke_config()
    return Atlas(cfg), cfg


def run_stage_smoke(stage: str) -> dict[str, Any]:
    stage_key = "teacher" if stage == "teacher" else stage
    model, cfg = build_model(stage_key)
    batch = make_synthetic_batch(cfg)
    privileged = {
        "lidar_tokens": torch.randn(batch["rgb_past"].shape[0], 12, cfg.hidden_dim)
    } if stage == "teacher" else None
    start = perf_counter()
    outputs = model.forward_train(
        batch["rgb_past"],
        batch["actions_hist"],
        batch["dt_hist"],
        route_polyline=batch["route_polyline"],
        nav_cmd=batch["nav_cmd"],
        reasoner_tok=batch["reasoner_tok"],
        privileged=privileged,
        stage="stage2" if stage == "teacher" else stage,
    )
    if stage == "stage1a":
        outputs["last"]["cam_projector_pred"] = outputs["last"]["cam_now"]
    if stage == "stage1c":
        outputs["last"]["world_projector_pred"] = outputs["last"]["persistent_tokens"]
    targets = make_synthetic_targets(outputs)
    if stage == "stage1a":
        targets["cam_projector_target"] = outputs["last"]["cam_now"].detach().clone()
        targets["cam_mask"] = torch.ones(
            outputs["last"]["cam_now"].shape[:2], dtype=torch.float32
        )
    if stage == "stage1c":
        targets["world_projector_target"] = outputs["last"]["persistent_tokens"].detach().clone()
        targets["world_mask"] = torch.ones(
            outputs["last"]["persistent_tokens"].shape[:2], dtype=torch.float32
        )
    losses = compute_stage_losses("stage2" if stage == "teacher" else stage, outputs["last"], targets)
    total = losses["total"]
    if not total.requires_grad:
        total = total + outputs["last"]["best_traj"].mean() * 0.0
        losses["total"] = total
    losses["total"].backward()
    elapsed = perf_counter() - start
    return {"outputs": outputs, "losses": losses, "elapsed_s": elapsed}
