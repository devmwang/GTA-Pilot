from __future__ import annotations

import time

import torch


def grad_norm(parameters) -> float:
    total = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        total += float(param.grad.detach().pow(2).sum().item())
    return total ** 0.5


class StepTimer:
    def __init__(self):
        self.last = time.perf_counter()

    def tick(self) -> float:
        now = time.perf_counter()
        dt = now - self.last
        self.last = now
        return dt


def tensor_memory_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
