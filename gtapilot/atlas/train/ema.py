from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn


class EMAModel:
    def __init__(self, module: nn.Module, decay: float):
        self.decay = float(decay)
        self.module = deepcopy(module).eval()
        for param in self.module.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, module: nn.Module) -> None:
        ema_state = dict(self.module.named_parameters())
        for name, param in module.named_parameters():
            if name not in ema_state:
                continue
            ema_state[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)
        ema_buffers = dict(self.module.named_buffers())
        for name, buffer in module.named_buffers():
            if name in ema_buffers:
                ema_buffers[name].copy_(buffer.detach())

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.module.state_dict()

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.module.load_state_dict(state_dict)
