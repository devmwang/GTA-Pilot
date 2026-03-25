from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TeacherTrajectorySet:
    trajectories: torch.Tensor
    costs: torch.Tensor
    best_index: torch.Tensor

    def as_targets(self) -> dict[str, torch.Tensor]:
        return {
            "gt_teacher_trajs": self.trajectories,
            "gt_teacher_costs": self.costs,
            "gt_teacher_best": self.best_index,
        }
