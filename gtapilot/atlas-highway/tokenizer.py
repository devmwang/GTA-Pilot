from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import AtlasHAConfig
from .temporal import CrossAttentionBlock


class FixedGridVisualTokenizer(nn.Module):
    level_grids: tuple[tuple[int, int], ...] = ((24, 42), (12, 21), (6, 11), (3, 6))

    def __init__(self, cfg: AtlasHAConfig):
        super().__init__()
        self.cfg = cfg
        self.source_proj = nn.Linear(cfg.fpn_dim, cfg.hidden_dim)
        self.level_embed = nn.Parameter(torch.randn(len(self.level_grids), cfg.hidden_dim) * 0.02)
        self.frame_queries = nn.Parameter(
            torch.randn(cfg.tokens_per_frame, cfg.hidden_dim) * 0.02
        )
        self.cross_attn = CrossAttentionBlock(cfg.hidden_dim, num_heads=8, mlp_ratio=2)
        self.summary = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )

    def _pool_level(
        self,
        x: torch.Tensor,
        grid: tuple[int, int],
        level_idx: int,
    ) -> torch.Tensor:
        pooled = F.adaptive_avg_pool2d(x, grid)
        tokens = pooled.flatten(2).transpose(1, 2)
        tokens = self.source_proj(tokens)
        return tokens + self.level_embed[level_idx].to(device=x.device, dtype=tokens.dtype)

    def forward(
        self,
        pyramid: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        source_tokens = torch.cat(
            [
                self._pool_level(pyramid[name], grid, idx)
                for idx, (name, grid) in enumerate(
                    zip(("p2", "p3", "p4", "p5"), self.level_grids)
                )
            ],
            dim=1,
        )
        queries = self.frame_queries.to(device=source_tokens.device, dtype=source_tokens.dtype)
        queries = queries.unsqueeze(0).expand(source_tokens.shape[0], -1, -1)
        frame_tokens = self.cross_attn(queries, source_tokens)
        frame_summary = self.summary(frame_tokens.mean(dim=1))
        return source_tokens, frame_tokens, frame_summary
