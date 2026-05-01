from __future__ import annotations

import torch
import torch.nn as nn

from .config import AtlasHAConfig
from .temporal import CrossAttentionBlock


class HighwayActionEncoder(nn.Module):
    def __init__(self, cfg: AtlasHAConfig):
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Linear(cfg.action_dim + 1, cfg.hidden_dim)
        self.age_embed = nn.Parameter(
            torch.randn(cfg.action_history_steps, cfg.hidden_dim) * 0.02
        )
        self.action_queries = nn.Parameter(
            torch.randn(cfg.action_summary_tokens, cfg.hidden_dim) * 0.02
        )
        self.reader = CrossAttentionBlock(cfg.hidden_dim, num_heads=8, mlp_ratio=2)
        self.summary = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )

    def forward(
        self,
        actions_hist: torch.Tensor,
        dt_hist: torch.Tensor,
        valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if actions_hist.ndim != 3 or actions_hist.shape[-1] != self.cfg.action_dim:
            raise ValueError("actions_hist must have shape [B, M, 6].")
        if dt_hist.ndim != 3 or dt_hist.shape[-1] != 1:
            raise ValueError("dt_hist must have shape [B, M, 1].")
        if actions_hist.shape[:2] != dt_hist.shape[:2]:
            raise ValueError("actions_hist and dt_hist must align.")
        action_dt = torch.cat([actions_hist, dt_hist], dim=-1)
        context = self.input_proj(action_dt)
        age_embed = self.age_embed[-context.shape[1] :].to(
            device=context.device,
            dtype=context.dtype,
        )
        context = context + age_embed.unsqueeze(0)
        if self.training and self.cfg.action_dropout_prob > 0.0:
            keep = torch.rand(context.shape[:2], device=context.device) >= self.cfg.action_dropout_prob
            context = context * keep.unsqueeze(-1).to(context.dtype)
        key_padding_mask = None if valid is None else ~valid.bool()
        if key_padding_mask is not None and bool(key_padding_mask.all().item()):
            key_padding_mask = None
        queries = self.action_queries.to(device=context.device, dtype=context.dtype)
        queries = queries.unsqueeze(0).expand(context.shape[0], -1, -1)
        action_tokens = self.reader(queries, context, key_padding_mask)
        action_summary = self.summary(action_tokens.mean(dim=1))
        return action_tokens, action_summary
