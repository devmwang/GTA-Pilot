from __future__ import annotations

import torch
import torch.nn as nn

from ..config import AtlasConfig
from ..utils import (
    CrossAttentionBlock,
    LearnedQueryPool,
    SelfAttentionBlock,
    assert_rank,
    sinusoidal_embedding,
)


def _history_age_embedding(dt_hist: torch.Tensor, dim: int) -> torch.Tensor:
    if dt_hist.numel() == 0:
        return torch.zeros(
            *dt_hist.shape[:-1],
            dim,
            device=dt_hist.device,
            dtype=dt_hist.dtype,
        )
    age = torch.flip(torch.cumsum(torch.flip(dt_hist, dims=[1]), dim=1), dims=[1])
    return sinusoidal_embedding(age, dim)


class StreamingCrossBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.self_block = SelfAttentionBlock(dim, heads, dropout)
        self.cross_block = CrossAttentionBlock(dim, heads, dropout)

    def forward(
        self,
        query: torch.Tensor,
        src: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query = self.self_block(query)
        if src.numel() == 0:
            return query
        return self.cross_block(query, src, key_padding_mask=src_key_padding_mask)


class RecentVisualMixer(nn.Module):
    def __init__(self, cfg: AtlasConfig):
        super().__init__()
        self.cfg = cfg
        d_model = cfg.hidden_dim
        heads = cfg.temporal.num_heads
        dropout = cfg.temporal.dropout
        self.current_pos = nn.Parameter(
            torch.randn(1, cfg.temporal.recent_cam_tokens, d_model) * 0.02
        )
        self.recent_pos = nn.Parameter(
            torch.randn(
                1,
                cfg.temporal.recent_cache_frames,
                cfg.temporal.recent_cam_tokens,
                d_model,
            )
            * 0.02
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.blocks = nn.ModuleList(
            [
                StreamingCrossBlock(d_model, heads, dropout)
                for _ in range(cfg.temporal.recent_mixer_blocks)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.short_pool = LearnedQueryPool(
            cfg.temporal.short_context_tokens,
            d_model,
            heads=heads,
            dropout=dropout,
        )

    def forward(
        self,
        current_cam_tokens: torch.Tensor,
        recent_cam_cache: torch.Tensor,
        recent_dt_cache: torch.Tensor,
        recent_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert_rank(current_cam_tokens, 3, "current_cam_tokens")
        assert_rank(recent_cam_cache, 4, "recent_cam_cache")
        assert_rank(recent_dt_cache, 3, "recent_dt_cache")
        assert_rank(recent_valid, 2, "recent_valid")

        b, n, d = current_cam_tokens.shape
        if n != self.cfg.temporal.recent_cam_tokens:
            raise ValueError(
                f"Expected {self.cfg.temporal.recent_cam_tokens} current camera tokens, got {n}"
            )

        query = current_cam_tokens + self.current_pos[:, :n]
        history_mask: torch.Tensor | None = None
        if recent_cam_cache.shape[1] == 0:
            history = recent_cam_cache.new_zeros(b, 0, d)
        else:
            time_emb = self.time_mlp(
                _history_age_embedding(recent_dt_cache, d).reshape(
                    b, recent_dt_cache.shape[1], 1, d
                )
            )
            history = (
                recent_cam_cache
                + self.recent_pos[:, : recent_cam_cache.shape[1], :n]
                + time_emb
            ).reshape(b, recent_cam_cache.shape[1] * n, d)
            history_mask = (~recent_valid.to(dtype=torch.bool, device=history.device))[
                :, :, None
            ].expand(-1, -1, n).reshape(b, recent_cam_cache.shape[1] * n)

        for block in self.blocks:
            query = block(query, history, src_key_padding_mask=history_mask)
        query = self.norm(query)
        pooled_src = torch.cat([query, history], dim=1)
        pooled_mask = None
        if history_mask is not None:
            pooled_mask = torch.cat(
                [
                    torch.zeros(
                        b,
                        query.shape[1],
                        device=query.device,
                        dtype=torch.bool,
                    ),
                    history_mask,
                ],
                dim=1,
            )
        short_ctx = self.short_pool(pooled_src, key_padding_mask=pooled_mask)
        return query, short_ctx


class HistoryCompressor(nn.Module):
    def __init__(self, cfg: AtlasConfig):
        super().__init__()
        self.cfg = cfg
        self.pool = LearnedQueryPool(
            cfg.temporal.older_compressed_tokens,
            cfg.hidden_dim,
            heads=cfg.temporal.num_heads,
            dropout=cfg.temporal.dropout,
        )

    def forward(self, frame_tokens: torch.Tensor) -> torch.Tensor:
        assert_rank(frame_tokens, 3, "frame_tokens")
        return self.pool(frame_tokens)

    def forward_frames(
        self,
        frame_tokens: torch.Tensor,
        frame_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert_rank(frame_tokens, 4, "frame_tokens")
        batch, steps, tokens, d_model = frame_tokens.shape
        out = frame_tokens.new_zeros(
            batch,
            steps,
            self.cfg.temporal.older_compressed_tokens,
            d_model,
        )
        if frame_valid is None:
            flat = frame_tokens.reshape(batch * steps, tokens, d_model)
            compressed = self.pool(flat)
            return compressed.reshape(
                batch,
                steps,
                self.cfg.temporal.older_compressed_tokens,
                d_model,
            )
        assert_rank(frame_valid, 2, "frame_valid")
        for step_idx in range(steps):
            valid = frame_valid[:, step_idx]
            if not valid.any():
                continue
            out[valid, step_idx] = self.pool(frame_tokens[valid, step_idx])
        return out


class SummaryBankMixer(nn.Module):
    def __init__(
        self,
        *,
        queries: int,
        cfg: AtlasConfig,
        max_steps: int,
        num_blocks: int,
    ):
        super().__init__()
        self.cfg = cfg
        d_model = cfg.hidden_dim
        heads = cfg.temporal.num_heads
        dropout = cfg.temporal.dropout
        self.pos = nn.Parameter(torch.randn(1, max_steps, 1, d_model) * 0.02)
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.query_pool = LearnedQueryPool(queries, d_model, heads=heads, dropout=dropout)
        self.blocks = nn.ModuleList(
            [
                StreamingCrossBlock(d_model, heads, dropout)
                for _ in range(num_blocks)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        bank_tokens: torch.Tensor,
        bank_dt: torch.Tensor,
        bank_valid: torch.Tensor,
    ) -> torch.Tensor:
        assert_rank(bank_tokens, 4, "bank_tokens")
        assert_rank(bank_dt, 3, "bank_dt")
        assert_rank(bank_valid, 2, "bank_valid")
        b, t, k, d = bank_tokens.shape
        if t == 0:
            return bank_tokens.new_zeros(b, self.query_pool.queries.shape[1], d)

        time_emb = self.time_mlp(_history_age_embedding(bank_dt, d).reshape(b, t, 1, d))
        src = (bank_tokens + self.pos[:, :t, :k] + time_emb).reshape(b, t * k, d)
        src_mask = (~bank_valid.to(dtype=torch.bool, device=src.device))[:, :, None].expand(
            -1, -1, k
        ).reshape(b, t * k)
        query = self.query_pool(src, key_padding_mask=src_mask)
        for block in self.blocks:
            query = block(query, src, src_key_padding_mask=src_mask)
        query = self.norm(query)
        has_valid = bank_valid.any(dim=1, keepdim=True).to(query.dtype)
        return query * has_valid.unsqueeze(-1)


class FrameSummaryProjector(nn.Module):
    def __init__(self, cfg: AtlasConfig):
        super().__init__()
        self.cfg = cfg
        self.pool = LearnedQueryPool(
            cfg.temporal.mid_summary_tokens,
            cfg.hidden_dim,
            heads=cfg.temporal.num_heads,
            dropout=cfg.temporal.dropout,
        )

    def forward(
        self,
        current_cam_tokens: torch.Tensor,
        short_ctx: torch.Tensor,
        older_ctx: torch.Tensor,
        long_ctx: torch.Tensor,
    ) -> torch.Tensor:
        src = torch.cat([current_cam_tokens, short_ctx, older_ctx, long_ctx], dim=1)
        return self.pool(src)

    def forward_frames(
        self,
        frame_tokens: torch.Tensor,
        frame_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert_rank(frame_tokens, 4, "frame_tokens")
        batch, steps, tokens, d_model = frame_tokens.shape
        out = frame_tokens.new_zeros(
            batch,
            steps,
            self.cfg.temporal.mid_summary_tokens,
            d_model,
        )
        if frame_valid is None:
            flat = frame_tokens.reshape(batch * steps, tokens, d_model)
            summary = self.pool(flat)
            return summary.reshape(batch, steps, summary.shape[1], d_model)
        assert_rank(frame_valid, 2, "frame_valid")
        for step_idx in range(steps):
            valid = frame_valid[:, step_idx]
            if not valid.any():
                continue
            out[valid, step_idx] = self.pool(frame_tokens[valid, step_idx])
        return out


class TemporalMixer(nn.Module):
    def __init__(self, cfg: AtlasConfig):
        super().__init__()
        self.cfg = cfg
        self.recent_mixer = RecentVisualMixer(cfg)
        self.history_compressor = HistoryCompressor(cfg)
        self.older_mixer = SummaryBankMixer(
            queries=cfg.temporal.older_context_tokens,
            cfg=cfg,
            max_steps=cfg.temporal.older_compressed_frames,
            num_blocks=cfg.temporal.older_mixer_blocks,
        )
        self.mid_mixer = SummaryBankMixer(
            queries=cfg.temporal.long_context_tokens,
            cfg=cfg,
            max_steps=cfg.temporal.mid_summary_frames,
            num_blocks=cfg.temporal.mid_mixer_blocks,
        )
        self.frame_summary = FrameSummaryProjector(cfg)

    def compress_history_frame(self, frame_tokens: torch.Tensor) -> torch.Tensor:
        return self.history_compressor(frame_tokens)

    def compress_history_frames(
        self,
        frame_tokens: torch.Tensor,
        frame_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.history_compressor.forward_frames(frame_tokens, frame_valid)

    def summarize_mid_frame(self, frame_tokens: torch.Tensor) -> torch.Tensor:
        return self.frame_summary(
            frame_tokens,
            frame_tokens.new_zeros(
                frame_tokens.shape[0],
                self.cfg.temporal.short_context_tokens,
                frame_tokens.shape[-1],
            ),
            frame_tokens.new_zeros(
                frame_tokens.shape[0],
                self.cfg.temporal.older_context_tokens,
                frame_tokens.shape[-1],
            ),
            frame_tokens.new_zeros(
                frame_tokens.shape[0],
                self.cfg.temporal.long_context_tokens,
                frame_tokens.shape[-1],
            ),
        )

    def summarize_mid_frames(
        self,
        frame_tokens: torch.Tensor,
        frame_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.frame_summary.forward_frames(frame_tokens, frame_valid)

    def forward(
        self,
        current_cam_tokens: torch.Tensor,
        recent_cam_cache: torch.Tensor,
        recent_dt_cache: torch.Tensor,
        recent_valid: torch.Tensor,
        older_cam_cache: torch.Tensor,
        older_dt_cache: torch.Tensor,
        older_valid: torch.Tensor,
        mid_summary_cache: torch.Tensor,
        mid_dt_cache: torch.Tensor,
        mid_valid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        cam_now, short_ctx = self.recent_mixer(
            current_cam_tokens=current_cam_tokens,
            recent_cam_cache=recent_cam_cache,
            recent_dt_cache=recent_dt_cache,
            recent_valid=recent_valid,
        )
        older_ctx = self.older_mixer(older_cam_cache, older_dt_cache, older_valid)
        long_ctx = self.mid_mixer(mid_summary_cache, mid_dt_cache, mid_valid)
        frame_summary = self.frame_summary(cam_now, short_ctx, older_ctx, long_ctx)
        return {
            "cam_now": cam_now,
            "short_ctx": short_ctx,
            "older_ctx": older_ctx,
            "long_ctx": long_ctx,
            "frame_summary": frame_summary,
        }
