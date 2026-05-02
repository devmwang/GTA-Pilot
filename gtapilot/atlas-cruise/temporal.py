from __future__ import annotations

import torch
import torch.nn as nn

from .config import AtlasCruiseConfig


class AtlasCruiseTemporalEncoder(nn.Module):
    def __init__(self, cfg: AtlasCruiseConfig):
        super().__init__()
        self.cfg = cfg
        self.cls_token = nn.Parameter(torch.randn(1, cfg.hidden_dim) * 0.02)
        self.frame_pos = nn.Parameter(torch.randn(cfg.num_visual_frames, cfg.hidden_dim) * 0.02)
        self.action_pos = nn.Parameter(torch.randn(cfg.action_tokens, cfg.hidden_dim) * 0.02)
        self.freshness_proj = nn.Sequential(
            nn.Linear(2, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.temporal_heads,
            dim_feedforward=int(round(cfg.hidden_dim * cfg.temporal_mlp_ratio)),
            dropout=cfg.temporal_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.temporal_layers)
        self.out = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )

    def forward(
        self,
        frame_summary: torch.Tensor,
        action_tokens: torch.Tensor,
        frame_freshness: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if frame_summary.ndim != 3 or frame_summary.shape[-1] != self.cfg.hidden_dim:
            raise ValueError("frame_summary must have shape [B, T, D].")
        if action_tokens.ndim != 3 or action_tokens.shape[-1] != self.cfg.hidden_dim:
            raise ValueError("action_tokens must have shape [B, R, D].")
        if frame_summary.shape[1] != self.cfg.num_visual_frames:
            raise ValueError(f"frame_summary must have T={self.cfg.num_visual_frames}.")
        if action_tokens.shape[1] != self.cfg.action_tokens:
            raise ValueError(f"action_tokens must have R={self.cfg.action_tokens}.")
        x_frames = frame_summary + self.frame_pos.to(device=frame_summary.device, dtype=frame_summary.dtype).unsqueeze(0)
        if frame_freshness is not None:
            if frame_freshness.shape[:2] != frame_summary.shape[:2] or frame_freshness.shape[-1] != 2:
                raise ValueError("frame_freshness must have shape [B, T, 2].")
            x_frames = x_frames + self.freshness_proj(frame_freshness.to(device=frame_summary.device, dtype=frame_summary.dtype))
        x_actions = action_tokens + self.action_pos.to(device=action_tokens.device, dtype=action_tokens.dtype).unsqueeze(0)
        cls = self.cls_token.to(device=frame_summary.device, dtype=frame_summary.dtype)
        cls = cls.unsqueeze(0).expand(frame_summary.shape[0], -1, -1)
        tokens = torch.cat([cls, x_frames, x_actions], dim=1)
        encoded = self.encoder(tokens)
        return self.out(encoded[:, 0]), encoded
