from __future__ import annotations

import torch
import torch.nn as nn

from .config import AtlasCruiseConfig


def _mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(in_dim),
        nn.Linear(in_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, out_dim),
    )


class AtlasCruiseHeads(nn.Module):
    def __init__(self, cfg: AtlasCruiseConfig):
        super().__init__()
        self.cfg = cfg
        self.traj_delta = _mlp(cfg.hidden_dim, cfg.hidden_dim, cfg.traj_points * 4)
        self.ego = _mlp(cfg.hidden_dim, cfg.hidden_dim, 4)
        self.traj_conf = _mlp(cfg.hidden_dim, cfg.hidden_dim, 1)
        self.slow_or_brake = _mlp(cfg.hidden_dim, cfg.hidden_dim, 1)
        self.fallback = _mlp(cfg.hidden_dim, cfg.hidden_dim, 1)
        self.control_aux = (
            _mlp(cfg.hidden_dim, cfg.hidden_dim, cfg.control_horizon_steps * 3)
            if cfg.predict_control_aux
            else None
        )
        base = torch.zeros(cfg.traj_points, 4)
        base[:, 0] = torch.linspace(0.0, 48.0, cfg.traj_points)
        base[:, 3] = 15.0
        self.register_buffer("base_traj", base, persistent=False)

    def forward(self, policy_context: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = policy_context.shape[0]
        raw = self.traj_delta(policy_context).view(batch_size, self.cfg.traj_points, 4)
        delta = torch.empty_like(raw)
        delta[..., 0] = torch.tanh(raw[..., 0]) * 18.0
        delta[..., 1] = torch.tanh(raw[..., 1]) * 10.0
        delta[..., 2] = torch.tanh(raw[..., 2]) * 0.8
        delta[..., 3] = torch.tanh(raw[..., 3]) * 14.0
        traj = self.base_traj.to(device=policy_context.device, dtype=policy_context.dtype).unsqueeze(0) + delta
        traj[..., 3] = traj[..., 3].clamp_min(0.0)
        ego = self.ego(policy_context)
        ego = torch.stack(
            [
                ego[:, 0].clamp_min(0.0),
                ego[:, 1],
                ego[:, 2],
                ego[:, 3],
            ],
            dim=-1,
        )
        outputs: dict[str, torch.Tensor] = {
            "traj": traj,
            "ego_kinematics": ego,
            "traj_conf_logit": self.traj_conf(policy_context),
            "slow_or_brake_logit": self.slow_or_brake(policy_context),
            "fallback_logit": self.fallback(policy_context),
        }
        if self.control_aux is not None:
            control = self.control_aux(policy_context).view(
                batch_size,
                self.cfg.control_horizon_steps,
                3,
            )
            outputs["control_aux"] = torch.stack(
                [
                    torch.tanh(control[..., 0]),
                    torch.sigmoid(control[..., 1]),
                    torch.sigmoid(control[..., 2]),
                ],
                dim=-1,
            )
        return outputs
