from __future__ import annotations

import torch
import torch.nn as nn

from ..config import PLWTConfig
from .common import LearnedQueryPool, assert_rank


class ActionEncoder(nn.Module):
    def __init__(self, cfg: PLWTConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.hidden_dim
        self.in_proj = nn.Sequential(
            nn.Linear(cfg.action.action_dim + 1, cfg.action.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.action.hidden_dim, D),
        )
        self.query_pool = LearnedQueryPool(
            num_queries=cfg.action.act_tokens,
            dim=D,
            heads=cfg.action.num_heads,
            dropout=cfg.action.dropout,
        )

    def forward(self, actions_hist: torch.Tensor, dt_hist: torch.Tensor) -> torch.Tensor:
        """
        actions_hist: [B, L, 6]
        dt_hist:      [B, L, 1]
        returns:
          act_tokens: [B, A, D]
        """
        assert_rank(actions_hist, 3, "actions_hist")
        assert_rank(dt_hist, 3, "dt_hist")
        if actions_hist.shape[1] != self.cfg.action.history_len:
            raise ValueError(f"Expected history_len={self.cfg.action.history_len}, got {actions_hist.shape[1]}")
        x = torch.cat([actions_hist, dt_hist], dim=-1)
        seq = self.in_proj(x)
        return self.query_pool(seq)
