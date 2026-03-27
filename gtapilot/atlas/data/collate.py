from __future__ import annotations

from typing import Any

import torch


def collate_temporal_clips(samples: list[dict[str, Any]]) -> dict[str, Any]:
    batch: dict[str, Any] = {}
    keys = set().union(*(sample.keys() for sample in samples))
    for key in keys:
        values = [sample.get(key) for sample in samples]
        first = values[0]
        if isinstance(first, torch.Tensor):
            batch[key] = torch.stack(values, dim=0)
        elif first is None:
            batch[key] = None
        elif isinstance(first, (int, float, str, bool)):
            if all(isinstance(value, (int, float, bool)) for value in values):
                batch[key] = torch.tensor(values)
            else:
                batch[key] = values
        else:
            batch[key] = values
    return batch
