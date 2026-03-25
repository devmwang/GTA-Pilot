from __future__ import annotations

import torch
import torch.nn as nn

from ..config import PLWTConfig
from .common import LearnedQueryPool, assert_rank


class RouteAdapter(nn.Module):
    def __init__(self, cfg: PLWTConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.hidden_dim
        self.point_mlp = nn.Sequential(
            nn.Linear(3, D),
            nn.GELU(),
            nn.Linear(D, D),
        )
        self.cmd_mlp = nn.Sequential(
            nn.Linear(cfg.route_adapter.nav_cmd_dim, D),
            nn.GELU(),
            nn.Linear(D, D),
        )
        self.pool = LearnedQueryPool(cfg.route_adapter.route_tokens, D, heads=min(8, max(1, D // 64)))

    def forward(self, route_polyline: torch.Tensor | None, nav_cmd: torch.Tensor | None, batch_size: int, device, dtype) -> torch.Tensor:
        if route_polyline is None:
            return torch.zeros(batch_size, self.cfg.route_adapter.route_tokens, self.cfg.hidden_dim, device=device, dtype=dtype)
        assert_rank(route_polyline, 3, "route_polyline")
        route_embed = self.point_mlp(route_polyline)
        if nav_cmd is not None:
            assert_rank(nav_cmd, 2, "nav_cmd")
            cmd = self.cmd_mlp(nav_cmd)[:, None, :]
            route_embed = route_embed + cmd
        return self.pool(route_embed)


class ReasonerBridge(nn.Module):
    def __init__(self, cfg: PLWTConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.hidden_dim
        self.proj = nn.Sequential(
            nn.Linear(cfg.reasoner_adapter.input_dim, D),
            nn.GELU(),
            nn.Linear(D, D),
        )

    def forward(
        self,
        external_tokens: torch.Tensor | None,
        prev_tokens: torch.Tensor,
        prev_ttl: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if external_tokens is None:
            ttl = torch.clamp(prev_ttl - 1.0, min=0.0)
            keep = (ttl > 0).to(prev_tokens.dtype)
            return prev_tokens * keep[:, :, None], ttl

        assert_rank(external_tokens, 3, "reasoner_tokens")
        x = self.proj(external_tokens)
        target_n = self.cfg.reasoner_adapter.output_tokens
        if x.shape[1] >= target_n:
            x = x[:, :target_n]
        else:
            pad = torch.zeros(x.shape[0], target_n - x.shape[1], x.shape[2], device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)
        ttl = torch.full((x.shape[0], 1), float(self.cfg.reasoner_adapter.ttl_steps), device=x.device, dtype=x.dtype)
        return x, ttl
