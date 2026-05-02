from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

from .config import AtlasCruiseConfig


_WEIGHT_ENUM_NAMES: dict[str, str] = {
    "regnet_y_400mf": "RegNet_Y_400MF_Weights",
    "regnet_y_800mf": "RegNet_Y_800MF_Weights",
    "regnet_y_1_6gf": "RegNet_Y_1_6GF_Weights",
}


def _resolve_regnet_weights(backbone_name: str, pretrained: bool):
    if not pretrained:
        return None
    enum_name = _WEIGHT_ENUM_NAMES.get(backbone_name)
    if enum_name is None:
        raise ValueError(f"No TorchVision weights are registered for {backbone_name!r}.")
    weights_enum = getattr(tv_models, enum_name)
    return weights_enum.DEFAULT


def _grid_for_token_count(token_count: int) -> tuple[int, int]:
    if token_count <= 0:
        raise ValueError("token_count must be positive.")
    best_h = 1
    best_w = token_count
    best_score = float("inf")
    target_ratio = 16.0 / 9.0
    for h in range(1, int(math.sqrt(token_count)) + 2):
        w = int(math.ceil(token_count / h))
        score = abs((w / h) - target_ratio) + (w * h - token_count) * 0.01
        if score < best_score:
            best_h, best_w, best_score = h, w, score
    return best_h, best_w


class AtlasCruiseBackbone(nn.Module):
    def __init__(self, cfg: AtlasCruiseConfig):
        super().__init__()
        if not hasattr(tv_models, cfg.backbone_name):
            raise ValueError(f"Unsupported TorchVision backbone {cfg.backbone_name!r}.")
        if cfg.backbone_name not in _WEIGHT_ENUM_NAMES and cfg.pretrained_backbone:
            raise ValueError(f"No pretrained RegNet weights are registered for {cfg.backbone_name!r}.")
        model_fn = getattr(tv_models, cfg.backbone_name)
        weights = _resolve_regnet_weights(cfg.backbone_name, cfg.pretrained_backbone)
        model = model_fn(weights=weights)
        self.stem = model.stem
        self.trunk = model.trunk_output
        self.out_channels = int(model.fc.in_features)
        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, scene_rgb: torch.Tensor) -> torch.Tensor:
        if scene_rgb.ndim != 4 or scene_rgb.shape[1] != 3:
            raise ValueError("scene_rgb must have shape [B, 3, H, W].")
        x = scene_rgb
        if x.dtype == torch.uint8:
            x = x.float().mul_(1.0 / 255.0)
        elif not torch.is_floating_point(x):
            x = x.float()
        x = (x - self.imagenet_mean.to(device=x.device, dtype=x.dtype)) / self.imagenet_std.to(
            device=x.device,
            dtype=x.dtype,
        )
        return self.trunk(self.stem(x))


class CruiseVisualTokenizer(nn.Module):
    def __init__(self, cfg: AtlasCruiseConfig, in_channels: int):
        super().__init__()
        self.cfg = cfg
        self.source_grid = _grid_for_token_count(cfg.source_tokens_per_frame)
        self.source_count = self.source_grid[0] * self.source_grid[1]
        self.source_proj = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.queries = nn.Parameter(torch.randn(cfg.tokens_per_frame, cfg.hidden_dim) * 0.02)
        self.query_norm = nn.LayerNorm(cfg.hidden_dim)
        self.context_norm = nn.LayerNorm(cfg.hidden_dim)
        self.attn = nn.MultiheadAttention(
            cfg.hidden_dim,
            cfg.temporal_heads,
            dropout=cfg.temporal_dropout,
            batch_first=True,
        )
        self.summary = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )

    def forward(self, feature_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if feature_map.ndim != 4:
            raise ValueError("feature_map must have shape [B, C, H, W].")
        pooled = F.adaptive_avg_pool2d(feature_map, self.source_grid)
        source = pooled.flatten(2).transpose(1, 2).contiguous()
        source = self.source_proj(source)
        queries = self.queries.to(device=source.device, dtype=source.dtype)
        queries = queries.unsqueeze(0).expand(source.shape[0], -1, -1)
        tokens, _ = self.attn(
            self.query_norm(queries),
            self.context_norm(source),
            self.context_norm(source),
            need_weights=False,
        )
        tokens = queries + tokens
        frame_summary = self.summary(tokens.mean(dim=1))
        return source, tokens, frame_summary


def build_cruise_backbone(cfg: AtlasCruiseConfig) -> AtlasCruiseBackbone:
    return AtlasCruiseBackbone(cfg)
