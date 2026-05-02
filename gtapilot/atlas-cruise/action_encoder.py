from __future__ import annotations

import torch
import torch.nn as nn

from .config import AtlasCruiseConfig


class AtlasCruiseActionEncoder(nn.Module):
    def __init__(self, cfg: AtlasCruiseConfig):
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Linear(cfg.action_dim + 1, cfg.hidden_dim)
        self.age_embed = nn.Parameter(torch.randn(cfg.action_history_steps, cfg.hidden_dim) * 0.02)
        self.queries = nn.Parameter(torch.randn(cfg.action_tokens, cfg.hidden_dim) * 0.02)
        self.query_norm = nn.LayerNorm(cfg.hidden_dim)
        self.context_norm = nn.LayerNorm(cfg.hidden_dim)
        self.attn = nn.MultiheadAttention(
            cfg.hidden_dim,
            cfg.temporal_heads,
            dropout=cfg.temporal_dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(cfg.hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.hidden_dim, int(round(cfg.hidden_dim * cfg.temporal_mlp_ratio))),
            nn.GELU(),
            nn.Linear(int(round(cfg.hidden_dim * cfg.temporal_mlp_ratio)), cfg.hidden_dim),
        )

    def forward(
        self,
        actions_hist: torch.Tensor,
        dt_hist: torch.Tensor,
        valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if actions_hist.ndim != 3 or actions_hist.shape[-1] != self.cfg.action_dim:
            raise ValueError("actions_hist must have shape [B, M, 6].")
        if dt_hist.ndim != 3 or dt_hist.shape[-1] != 1:
            raise ValueError("dt_hist must have shape [B, M, 1].")
        if actions_hist.shape[:2] != dt_hist.shape[:2]:
            raise ValueError("actions_hist and dt_hist must align.")
        context = self.input_proj(torch.cat([actions_hist, dt_hist], dim=-1))
        context = context + self.age_embed[-context.shape[1] :].to(device=context.device, dtype=context.dtype).unsqueeze(0)
        if self.training and self.cfg.action_dropout_prob > 0.0:
            keep = torch.rand(context.shape[:2], device=context.device) >= self.cfg.action_dropout_prob
            context = context * keep.unsqueeze(-1).to(context.dtype)
        key_padding_mask = None
        if valid is not None:
            key_padding_mask = ~valid.bool()
            if bool(key_padding_mask.all().item()):
                key_padding_mask = None
        queries = self.queries.to(device=context.device, dtype=context.dtype)
        queries = queries.unsqueeze(0).expand(context.shape[0], -1, -1)
        attended, _ = self.attn(
            self.query_norm(queries),
            self.context_norm(context),
            self.context_norm(context),
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        out = queries + attended
        return out + self.ffn(self.ffn_norm(out))
