from __future__ import annotations

import torch
import torch.nn as nn

from ..config import AtlasConfig
from ..utils import LearnedQueryPool, assert_rank


class RouteAdapter(nn.Module):
    def __init__(self, cfg: AtlasConfig):
        super().__init__()
        self.cfg = cfg
        d_model = cfg.hidden_dim
        self.point_mlp = nn.Sequential(
            nn.Linear(3, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.cmd_mlp = nn.Sequential(
            nn.Linear(cfg.route_adapter.nav_cmd_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.pool = LearnedQueryPool(
            cfg.route_adapter.route_tokens,
            d_model,
            heads=min(8, max(1, d_model // 64)),
        )

    def forward(
        self,
        route_polyline: torch.Tensor | None,
        nav_cmd: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if route_polyline is None:
            return torch.zeros(
                batch_size,
                self.cfg.route_adapter.route_tokens,
                self.cfg.hidden_dim,
                device=device,
                dtype=dtype,
            )
        assert_rank(route_polyline, 3, "route_polyline")
        route_embed = self.point_mlp(route_polyline)
        if nav_cmd is not None:
            assert_rank(nav_cmd, 2, "nav_cmd")
            route_embed = route_embed + self.cmd_mlp(nav_cmd)[:, None, :]
        return self.pool(route_embed)
