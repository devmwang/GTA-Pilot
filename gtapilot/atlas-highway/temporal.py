from __future__ import annotations

import torch
import torch.nn as nn

from .config import AtlasHAConfig


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        *,
        num_heads: int = 8,
        mlp_ratio: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(hidden_dim * mlp_ratio, hidden_dim),
        )

    def forward(
        self,
        queries: torch.Tensor,
        context: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attn_out, _ = self.attn(
            self.query_norm(queries),
            self.context_norm(context),
            self.context_norm(context),
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        queries = queries + attn_out
        queries = queries + self.ffn(self.ffn_norm(queries))
        return queries


def _all_invalid(mask: torch.Tensor | None) -> bool:
    if mask is None:
        return False
    return bool(mask.all().item()) if mask.numel() > 0 else False


class FullContextReader(nn.Module):
    def __init__(self, cfg: AtlasHAConfig):
        super().__init__()
        self.cfg = cfg
        self.queries = nn.Parameter(torch.randn(cfg.full_context_queries, cfg.hidden_dim) * 0.02)
        self.blocks = nn.ModuleList(
            [
                CrossAttentionBlock(cfg.hidden_dim, num_heads=8, mlp_ratio=2)
                for _ in range(cfg.full_context_blocks)
            ]
        )

    def forward(
        self,
        full_token_cache: torch.Tensor,
        full_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if full_token_cache.ndim != 4:
            raise ValueError("full_token_cache must have shape [B, S_full, F, D].")
        batch_size, _, tokens_per_frame, hidden_dim = full_token_cache.shape
        context = full_token_cache.reshape(batch_size, -1, hidden_dim)
        key_padding_mask = None
        if full_valid is not None:
            key_padding_mask = ~full_valid.bool().repeat_interleave(tokens_per_frame, dim=1)
            if _all_invalid(key_padding_mask):
                key_padding_mask = None
        queries = self.queries.to(device=context.device, dtype=context.dtype)
        queries = queries.unsqueeze(0).expand(batch_size, -1, -1)
        for block in self.blocks:
            queries = block(queries, context, key_padding_mask)
        return queries


class SummaryContextReader(nn.Module):
    def __init__(self, cfg: AtlasHAConfig):
        super().__init__()
        self.queries = nn.Parameter(
            torch.randn(cfg.summary_context_queries, cfg.hidden_dim) * 0.02
        )
        self.blocks = nn.ModuleList(
            [
                CrossAttentionBlock(cfg.hidden_dim, num_heads=8, mlp_ratio=2)
                for _ in range(cfg.summary_context_blocks)
            ]
        )

    def forward(
        self,
        summary_cache: torch.Tensor,
        summary_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if summary_cache.ndim != 3:
            raise ValueError("summary_cache must have shape [B, S_sum, D].")
        key_padding_mask = None
        if summary_valid is not None:
            key_padding_mask = ~summary_valid.bool()
            if _all_invalid(key_padding_mask):
                key_padding_mask = None
        queries = self.queries.to(device=summary_cache.device, dtype=summary_cache.dtype)
        queries = queries.unsqueeze(0).expand(summary_cache.shape[0], -1, -1)
        for block in self.blocks:
            queries = block(queries, summary_cache, key_padding_mask)
        return queries


class HeadFusionReader(nn.Module):
    def __init__(self, cfg: AtlasHAConfig, query_count: int = 8):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(query_count, cfg.hidden_dim) * 0.02)
        self.block = CrossAttentionBlock(cfg.hidden_dim, num_heads=8, mlp_ratio=2)
        self.summary = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )

    def forward(self, context_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        queries = self.queries.to(device=context_tokens.device, dtype=context_tokens.dtype)
        queries = queries.unsqueeze(0).expand(context_tokens.shape[0], -1, -1)
        head_tokens = self.block(queries, context_tokens)
        head_context = self.summary(head_tokens.mean(dim=1))
        return head_tokens, head_context
