from __future__ import annotations

import math

import torch
import torch.nn as nn


def build_2d_sincos_position_embedding(
    height: int,
    width: int,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if dim % 4 != 0:
        raise ValueError("2D sin/cos position embedding dim must be divisible by 4")

    quarter = dim // 4
    y, x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, steps=height, device=device, dtype=dtype),
        torch.linspace(-1.0, 1.0, steps=width, device=device, dtype=dtype),
        indexing="ij",
    )
    omega = torch.exp(
        torch.arange(quarter, device=device, dtype=dtype)
        * (-math.log(10000.0) / max(quarter - 1, 1))
    )
    x = x.reshape(-1, 1) * omega.reshape(1, -1)
    y = y.reshape(-1, 1) * omega.reshape(1, -1)
    embedding = torch.cat([torch.sin(x), torch.cos(x), torch.sin(y), torch.cos(y)], dim=-1)
    return embedding.unsqueeze(0)


class RelativePositionBias(nn.Module):
    def __init__(self, window_size: int, num_heads: int):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        num_relative_positions = (2 * window_size - 1) * (2 * window_size - 1)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(num_relative_positions, num_heads)
        )

        coords = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords, coords, indexing="ij"))
        coords_flatten = coords.flatten(1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer(
            "relative_position_index",
            relative_position_index,
            persistent=False,
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self) -> torch.Tensor:
        window_area = self.window_size * self.window_size
        bias = self.relative_position_bias_table[
            self.relative_position_index.reshape(-1)
        ]
        bias = bias.reshape(window_area, window_area, self.num_heads)
        return bias.permute(2, 0, 1).contiguous()
