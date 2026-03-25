from __future__ import annotations

import torch.nn as nn


class TrackHead(nn.Conv2d):
    def __init__(self, in_channels: int, track_history: int):
        super().__init__(in_channels, track_history * 2, kernel_size=1)


__all__ = ["TrackHead"]
