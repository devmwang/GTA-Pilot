from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .schema import LoggedStep


def collate_optional_tensor(values: list[Any], dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor | None, torch.Tensor]:
    if all(value is None for value in values):
        mask = torch.zeros(len(values), dtype=torch.bool)
        return None, mask
    reference = next(value for value in values if value is not None)
    ref_array = torch.as_tensor(reference, dtype=dtype)
    stacked = torch.zeros((len(values), *ref_array.shape), dtype=dtype)
    mask = torch.zeros(len(values), dtype=torch.bool)
    for idx, value in enumerate(values):
        if value is None:
            continue
        stacked[idx] = torch.as_tensor(value, dtype=dtype)
        mask[idx] = True
    return stacked, mask


def collate_logged_steps(steps: list[LoggedStep]) -> dict[str, Any]:
    batch: dict[str, Any] = {
        "episode_id": [step.episode_id for step in steps],
        "frame_idx": torch.tensor([step.frame_idx for step in steps], dtype=torch.long),
        "timestamp_ms": torch.tensor([step.timestamp_ms for step in steps], dtype=torch.long),
        "rgb_front_path": [step.rgb_front_path for step in steps],
        "action": torch.as_tensor(np.stack([step.action for step in steps]), dtype=torch.float32),
        "dt_s": torch.tensor([step.dt_s for step in steps], dtype=torch.float32).unsqueeze(-1),
    }

    for field_name in (
        "route_polyline",
        "nav_cmd",
        "gt_pose_local",
        "gt_kinematics",
        "gt_occ_state",
        "gt_occ_sem",
        "gt_bev_lite",
        "gt_provenance",
        "gt_teacher_trajs",
        "gt_teacher_costs",
    ):
        stacked, mask = collate_optional_tensor(
            [getattr(step, field_name) for step in steps],
            dtype=torch.float32,
        )
        batch[field_name] = stacked
        batch[f"{field_name}_valid"] = mask

    batch["gt_teacher_best"] = torch.tensor(
        [(-1 if step.gt_teacher_best is None else step.gt_teacher_best) for step in steps],
        dtype=torch.long,
    )
    batch["gt_lane_segments"] = [step.gt_lane_segments for step in steps]
    batch["gt_map_elements"] = [step.gt_map_elements for step in steps]
    batch["gt_actors"] = [step.gt_actors for step in steps]
    batch["valid_mask"] = [step.valid_mask for step in steps]
    return batch
