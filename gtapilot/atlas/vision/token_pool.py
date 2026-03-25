from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from ..config import AtlasConfig
from ..utils import MLP, assert_rank, flatten_hw
from .positional import build_2d_sincos_position_embedding


class MultiscaleSceneTokenPool(nn.Module):
    def __init__(self, cfg: AtlasConfig):
        super().__init__()
        vc = cfg.vision
        D = cfg.hidden_dim
        C = vc.fusion_dim
        self.cfg = cfg
        self.f8_to_d = nn.Conv2d(C, D, kernel_size=1)
        self.f16_to_d = nn.Conv2d(C, D, kernel_size=1)
        self.f32_to_d = nn.Conv2d(C, D, kernel_size=1)
        self.f64_to_d = nn.Conv2d(C, D, kernel_size=1)
        self.detail_summary = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(vc.detail_dim, D, kernel_size=1),
            nn.GELU(),
        )
        self.scale_embeddings = nn.Parameter(torch.randn(4, D) * 0.02)
        self.query_tokens = nn.Parameter(
            torch.randn(1, vc.cam_tokens_per_frame, D) * 0.02
        )
        self.cross_attn = nn.MultiheadAttention(
            D,
            vc.query_pool_heads,
            dropout=vc.dropout,
            batch_first=True,
        )
        self.norm_q = nn.LayerNorm(D)
        self.norm_kv = nn.LayerNorm(D)
        self.post = MLP(D, D * 4, D, dropout=vc.dropout)

    def _project_scale(
        self,
        feature: torch.Tensor,
        projector: nn.Module,
        scale_index: int,
    ) -> torch.Tensor:
        assert_rank(feature, 4, "feature")
        projected = projector(feature)
        tokens = flatten_hw(projected)
        pos = build_2d_sincos_position_embedding(
            projected.shape[-2],
            projected.shape[-1],
            projected.shape[1],
            projected.device,
            projected.dtype,
        )
        return tokens + pos + self.scale_embeddings[scale_index].view(1, 1, -1)

    def forward(
        self,
        detail_4x: torch.Tensor,
        fused_features: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        src = [
            self._project_scale(fused_features["f8"], self.f8_to_d, 0),
            self._project_scale(fused_features["f16"], self.f16_to_d, 1),
            self._project_scale(fused_features["f32"], self.f32_to_d, 2),
            self._project_scale(fused_features["f64"], self.f64_to_d, 3),
        ]
        if self.cfg.vision.include_detail_summary:
            detail_summary = self.detail_summary(detail_4x).flatten(2).transpose(1, 2)
            src.append(detail_summary)
        src_tokens = torch.cat(src, dim=1)

        queries = self.query_tokens.expand(src_tokens.shape[0], -1, -1)
        pooled, _ = self.cross_attn(
            self.norm_q(queries),
            self.norm_kv(src_tokens),
            self.norm_kv(src_tokens),
            need_weights=False,
        )
        return pooled + self.post(pooled)
