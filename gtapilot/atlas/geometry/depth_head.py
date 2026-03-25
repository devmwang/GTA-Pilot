from __future__ import annotations

import torch.nn as nn


class DepthHead(nn.Conv2d):
    def __init__(self, in_channels: int, depth_bins: int):
        super().__init__(in_channels, depth_bins, kernel_size=1)


__all__ = ["DepthHead"]
