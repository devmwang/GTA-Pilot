from __future__ import annotations

import torch
import torch.nn as nn

from ..config import PLWTConfig
from .common import LearnedQueryPool, assert_rank


class ObservationPool(nn.Module):
    def __init__(self, cfg: PLWTConfig):
        super().__init__()
        self.cfg = cfg
        self.pool = LearnedQueryPool(
            num_queries=cfg.obs_pool.obs_tokens,
            dim=cfg.hidden_dim,
            heads=cfg.obs_pool.num_heads,
        )

    def forward(
        self,
        cam_now: torch.Tensor,
        frustum_tokens: torch.Tensor,
        ego_tokens: torch.Tensor,
        act_tokens: torch.Tensor,
    ) -> torch.Tensor:
        assert_rank(cam_now, 3, "cam_now")
        assert_rank(frustum_tokens, 3, "frustum_tokens")
        src = torch.cat([cam_now, frustum_tokens, ego_tokens, act_tokens], dim=1)
        return self.pool(src)
