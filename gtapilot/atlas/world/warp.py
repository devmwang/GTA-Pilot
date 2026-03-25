from __future__ import annotations

import torch

from ..utils import warp_bev


def warp_static_grid(
    grid: torch.Tensor,
    pose_delta: torch.Tensor,
    cell_x_m: float,
    cell_y_m: float,
) -> torch.Tensor:
    return warp_bev(
        grid,
        pose_delta=pose_delta,
        cell_x_m=cell_x_m,
        cell_y_m=cell_y_m,
    )
