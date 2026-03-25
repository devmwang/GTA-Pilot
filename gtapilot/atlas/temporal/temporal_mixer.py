from __future__ import annotations

import torch
import torch.nn as nn

from ..config import AtlasConfig
from ..utils import SelfAttentionBlock, sinusoidal_embedding, assert_rank


class TemporalMixer(nn.Module):
    def __init__(self, cfg: AtlasConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.hidden_dim
        self.frame_embed = nn.Parameter(torch.randn(1, cfg.temporal.num_frames, 1, D) * 0.02)
        self.time_mlp = nn.Sequential(
            nn.Linear(D, D),
            nn.GELU(),
            nn.Linear(D, D),
        )
        self.blocks = nn.ModuleList(
            [
                SelfAttentionBlock(D, cfg.temporal.num_heads, cfg.temporal.dropout)
                for _ in range(cfg.temporal.mixer_blocks)
            ]
        )
        self.norm = nn.LayerNorm(D)

    def forward(self, cam_tokens: torch.Tensor, dt_hist: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        cam_tokens: [B, T, N, D]
        dt_hist:    [B, T, 1]
        returns:
          cam_now:  [B, N, D]
          frame_sum:[B, T, D]
        """
        assert_rank(cam_tokens, 4, "cam_tokens")
        assert_rank(dt_hist, 3, "dt_hist")
        b, t, n, d = cam_tokens.shape
        if t != self.cfg.temporal.num_frames:
            raise ValueError(f"Expected {self.cfg.temporal.num_frames} frames, got {t}")
        if n != self.cfg.vision.cam_tokens_per_frame:
            raise ValueError(f"Expected {self.cfg.vision.cam_tokens_per_frame} camera tokens, got {n}")

        dt_emb = sinusoidal_embedding(dt_hist, d).reshape(b, t, 1, d)
        dt_emb = self.time_mlp(dt_emb)
        x = cam_tokens + self.frame_embed[:, :t] + dt_emb
        x = x.reshape(b, t * n, d)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x).reshape(b, t, n, d)
        frame_sum = x.mean(dim=2)
        cam_now = x[:, -1]
        return cam_now, frame_sum


AtlasTemporalMixer = TemporalMixer
