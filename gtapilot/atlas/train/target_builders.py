from __future__ import annotations

from typing import Any

import torch


def gather_future_sequence_targets(
    sequence: torch.Tensor,
    dt_sequence: torch.Tensor,
    horizons_s: list[float] | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if sequence.ndim != 4:
        raise ValueError("sequence must have shape [B, T, N, D].")
    if dt_sequence.ndim != 3 or dt_sequence.shape[:2] != sequence.shape[:2]:
        raise ValueError("dt_sequence must have shape [B, T, 1] aligned with sequence.")

    batch, steps, tokens, dim = sequence.shape
    if not isinstance(horizons_s, torch.Tensor):
        horizons_s = torch.tensor(horizons_s, device=sequence.device, dtype=sequence.dtype)
    horizons_s = horizons_s.to(device=sequence.device, dtype=dt_sequence.dtype).flatten()
    dt_values = dt_sequence.squeeze(-1)
    target = sequence.new_zeros(batch, steps, horizons_s.numel(), tokens, dim)
    valid = torch.zeros(batch, steps, horizons_s.numel(), device=sequence.device, dtype=torch.bool)

    for batch_idx in range(batch):
        for step_idx in range(steps):
            future_elapsed_s = torch.cumsum(dt_values[batch_idx, step_idx + 1 :], dim=0)
            for horizon_idx, horizon_s in enumerate(horizons_s):
                if horizon_s <= 0:
                    target[batch_idx, step_idx, horizon_idx] = sequence[batch_idx, step_idx]
                    valid[batch_idx, step_idx, horizon_idx] = True
                    continue
                future_mask = future_elapsed_s >= horizon_s
                if not bool(future_mask.any()):
                    continue
                target_offset = int(torch.nonzero(future_mask, as_tuple=False)[0].item()) + 1
                target[batch_idx, step_idx, horizon_idx] = sequence[
                    batch_idx,
                    step_idx + target_offset,
                ]
                valid[batch_idx, step_idx, horizon_idx] = True
    return target, valid


def build_stage1a_targets(
    *,
    teacher_cam_seq: torch.Tensor,
    teacher_summary_seq: torch.Tensor,
    dt_recent: torch.Tensor,
    dt_summary: torch.Tensor,
    cam_horizons_s: list[float] | torch.Tensor,
    summary_horizons_s: list[float] | torch.Tensor,
) -> dict[str, torch.Tensor]:
    cam_target, cam_valid = gather_future_sequence_targets(
        teacher_cam_seq,
        dt_recent,
        cam_horizons_s,
    )
    summary_target, summary_valid = gather_future_sequence_targets(
        teacher_summary_seq,
        dt_summary,
        summary_horizons_s,
    )
    return {
        "cam_projector_target": cam_target,
        "cam_mask": cam_valid,
        "summary_projector_target": summary_target,
        "summary_mask": summary_valid,
    }


def build_stage1c_targets(
    *,
    teacher_world_seq: torch.Tensor,
    teacher_static_grid_seq: torch.Tensor,
    dt_recent: torch.Tensor,
    world_horizons_s: list[float] | torch.Tensor,
) -> dict[str, torch.Tensor]:
    world_target, world_valid = gather_future_sequence_targets(
        teacher_world_seq,
        dt_recent,
        world_horizons_s,
    )
    return {
        "world_projector_target": world_target,
        "world_mask": world_valid,
        "teacher_static_grid_seq": teacher_static_grid_seq,
    }


def stack_sequence_dict(per_step: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    if not per_step:
        return {}
    keys = set.intersection(*(set(step.keys()) for step in per_step))
    stacked: dict[str, torch.Tensor] = {}
    for key in keys:
        values = [step[key] for step in per_step]
        if all(isinstance(value, torch.Tensor) for value in values):
            stacked[key] = torch.stack(values, dim=1)
    return stacked
