from __future__ import annotations

import torch
import torch.nn as nn


class FutureEvaluator(nn.Module):
    """
    Placeholder evaluator block for future factorization work.

    The main planner currently owns rollout scoring end to end; this class exists so the
    package layout matches the Atlas architecture brief and can be expanded without moving
    public imports again.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim)

    def forward(self, proposal_tokens: torch.Tensor) -> torch.Tensor:
        return self.proj(proposal_tokens)


__all__ = ["FutureEvaluator"]
