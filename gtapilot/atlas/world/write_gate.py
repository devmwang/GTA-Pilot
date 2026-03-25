from __future__ import annotations

import torch
import torch.nn as nn


class StaticWriteGate(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model * 3 + 3, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        static_tokens: torch.Tensor,
        updated_static: torch.Tensor,
        obs_summary: torch.Tensor,
        pose_uncertainty: torch.Tensor,
    ) -> torch.Tensor:
        pose_features = pose_uncertainty[:, None, :].expand(
            -1, static_tokens.shape[1], -1
        )
        gate_input = torch.cat(
            [static_tokens, updated_static, obs_summary, pose_features], dim=-1
        )
        return torch.sigmoid(self.gate(gate_input))
