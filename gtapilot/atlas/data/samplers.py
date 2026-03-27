from __future__ import annotations

import random
from collections import OrderedDict
from typing import Iterable

from torch.utils.data import Sampler


class ClipGroupedBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        *,
        clip_ids: list[str],
        batch_size: int,
        drop_last: bool,
        shuffle: bool,
        seed: int = 1337,
    ):
        self.clip_ids = clip_ids
        self.batch_size = max(1, int(batch_size))
        self.drop_last = bool(drop_last)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        grouped: OrderedDict[str, list[int]] = OrderedDict()
        for sample_index, clip_id in enumerate(clip_ids):
            grouped.setdefault(clip_id, []).append(sample_index)
        self._grouped_indices = grouped
        self._batches = self._build_batches(epoch=0)
        self._epoch = 0

    def _build_batches(self, *, epoch: int) -> list[list[int]]:
        rng = random.Random(self.seed + epoch)
        clip_order = list(self._grouped_indices.keys())
        if self.shuffle:
            rng.shuffle(clip_order)
        batches: list[list[int]] = []
        for clip_id in clip_order:
            clip_indices = list(self._grouped_indices[clip_id])
            if self.shuffle:
                clip_batches = [
                    clip_indices[start : start + self.batch_size]
                    for start in range(0, len(clip_indices), self.batch_size)
                ]
                rng.shuffle(clip_batches)
                for batch in clip_batches:
                    if len(batch) == self.batch_size or not self.drop_last:
                        batches.append(batch)
            else:
                for start in range(0, len(clip_indices), self.batch_size):
                    batch = clip_indices[start : start + self.batch_size]
                    if len(batch) == self.batch_size or not self.drop_last:
                        batches.append(batch)
        return batches

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)
        self._batches = self._build_batches(epoch=self._epoch)

    def __iter__(self) -> Iterable[list[int]]:
        self._batches = self._build_batches(epoch=self._epoch)
        self._epoch += 1
        return iter(self._batches)

    def __len__(self) -> int:
        return len(self._batches)
