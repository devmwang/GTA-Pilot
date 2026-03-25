from __future__ import annotations

import torch
import torch.nn as nn

from ..config import AtlasConfig
from ..utils import assert_rank


class ReasonerBridge(nn.Module):
    def __init__(self, cfg: AtlasConfig):
        super().__init__()
        self.cfg = cfg
        d_model = cfg.hidden_dim
        self.proj = nn.Sequential(
            nn.Linear(cfg.reasoner_adapter.input_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
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
        tokens = self.proj(external_tokens)
        target_n = self.cfg.reasoner_adapter.output_tokens
        if tokens.shape[1] >= target_n:
            tokens = tokens[:, :target_n]
        else:
            pad = torch.zeros(
                tokens.shape[0],
                target_n - tokens.shape[1],
                tokens.shape[2],
                device=tokens.device,
                dtype=tokens.dtype,
            )
            tokens = torch.cat([tokens, pad], dim=1)
        ttl = torch.full(
            (tokens.shape[0], 1),
            float(self.cfg.reasoner_adapter.ttl_steps),
            device=tokens.device,
            dtype=tokens.dtype,
        )
        return tokens, ttl
