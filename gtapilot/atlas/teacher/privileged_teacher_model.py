from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ..config import AtlasConfig
from ..model import Atlas
from ..utils import LearnedQueryPool


class _PrivilegedTokenAdapter(nn.Module):
    def __init__(self, input_dim: int, output_tokens: int, d_model: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.pool = LearnedQueryPool(output_tokens, d_model, heads=max(1, d_model // 64))

    def forward(self, tensor: torch.Tensor | None, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if tensor is None:
            return torch.zeros(batch_size, 0, self.proj[-1].out_features, device=device, dtype=dtype)
        if tensor.ndim == 2:
            tensor = tensor[:, None, :]
        embedded = self.proj(tensor.to(dtype))
        return self.pool(embedded)


class PrivilegedTeacherModel(Atlas):
    def __init__(self, cfg: AtlasConfig):
        super().__init__(cfg)
        d_model = cfg.hidden_dim
        self.lidar_adapter = _PrivilegedTokenAdapter(d_model, 96, d_model)
        self.pose_adapter = _PrivilegedTokenAdapter(3, 4, d_model)
        self.actor_adapter = _PrivilegedTokenAdapter(d_model, 32, d_model)
        self.map_adapter = _PrivilegedTokenAdapter(d_model, 16, d_model)

    def encode_privileged_tokens(
        self,
        privileged: dict[str, Any] | None,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        privileged = privileged or {}
        return {
            "priv_lidar_tokens": self.lidar_adapter(
                privileged.get("lidar_tokens"), batch_size, device, dtype
            ),
            "priv_pose_tokens": self.pose_adapter(
                privileged.get("pose_tokens"), batch_size, device, dtype
            ),
            "priv_actor_tokens": self.actor_adapter(
                privileged.get("actor_tokens"), batch_size, device, dtype
            ),
            "priv_map_tokens": self.map_adapter(
                privileged.get("map_tokens"), batch_size, device, dtype
            ),
        }

    def forward_train(self, *args: Any, privileged: dict[str, Any] | None = None, **kwargs: Any) -> dict[str, Any]:
        outputs = super().forward_train(*args, privileged=privileged, **kwargs)
        batch_size = outputs["final_state"].static_grid.shape[0]
        device = outputs["final_state"].static_grid.device
        dtype = outputs["final_state"].static_grid.dtype
        outputs["teacher_privileged_tokens"] = self.encode_privileged_tokens(
            privileged, batch_size=batch_size, device=device, dtype=dtype
        )
        return outputs
