from __future__ import annotations

import torch
import torch.nn as nn

from ..config import AtlasConfig
from ..utils import assert_rank


class EgoFilter(nn.Module):
    def __init__(self, cfg: AtlasConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.hidden_dim
        H = cfg.ego.hidden_size
        self.in_proj = nn.Linear(D * 3, H)
        self.gru = nn.GRU(
            input_size=H,
            hidden_size=H,
            num_layers=cfg.ego.num_layers,
            batch_first=True,
        )
        self.token_head = nn.Linear(H, cfg.ego.ego_tokens * D)
        self.pose_head = nn.Linear(H, 3)
        self.kin_head = nn.Linear(H, 4)
        self.logvar_head = nn.Linear(H, 3)

    def forward(
        self,
        cam_now: torch.Tensor,
        frame_sum: torch.Tensor,
        act_tokens: torch.Tensor,
        hidden: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        assert_rank(cam_now, 3, "cam_now")
        assert_rank(frame_sum, 3, "frame_sum")
        assert_rank(act_tokens, 3, "act_tokens")
        cam_summary = cam_now.mean(dim=1)
        frame_summary = frame_sum[:, -1]
        act_summary = act_tokens.mean(dim=1)
        x = torch.cat([cam_summary, frame_summary, act_summary], dim=-1)
        x = self.in_proj(x).unsqueeze(1)
        out, hidden_next = self.gru(x, hidden)
        h = out[:, 0]
        D = self.cfg.hidden_dim
        ego_tokens = self.token_head(h).reshape(h.shape[0], self.cfg.ego.ego_tokens, D)
        pose_delta = self.pose_head(h)
        kinematics = self.kin_head(h)
        logvar_pose = self.logvar_head(h).clamp(min=-6.0, max=4.0)
        return {
            "ego_tokens": ego_tokens,
            "pose_delta": pose_delta,
            "kinematics": kinematics,
            "logvar_pose": logvar_pose,
            "hidden_next": hidden_next,
        }


AtlasEgoFilter = EgoFilter
