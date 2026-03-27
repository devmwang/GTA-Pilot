from __future__ import annotations

from typing import Any

import torch


def collate_temporal_clips(samples: list[dict[str, Any]]) -> dict[str, Any]:
    batch: dict[str, Any] = {
        "clip_id": [sample["clip_id"] for sample in samples],
        "metadata_path": [sample["metadata_path"] for sample in samples],
        "video_path": [sample["video_path"] for sample in samples],
        "target_frame_index": torch.tensor(
            [sample["target_frame_index"] for sample in samples],
            dtype=torch.long,
        ),
        "frame_source": [sample["frame_source"] for sample in samples],
        "rgb_recent": torch.stack([sample["rgb_recent"] for sample in samples], dim=0),
        "dt_recent": torch.stack([sample["dt_recent"] for sample in samples], dim=0),
        "rgb_older": torch.stack([sample["rgb_older"] for sample in samples], dim=0),
        "dt_older": torch.stack([sample["dt_older"] for sample in samples], dim=0),
        "rgb_mid": torch.stack([sample["rgb_mid"] for sample in samples], dim=0),
        "dt_mid": torch.stack([sample["dt_mid"] for sample in samples], dim=0),
        "actions_hist": torch.stack(
            [sample["actions_hist"] for sample in samples],
            dim=0,
        ),
        "dt_hist": torch.stack([sample["dt_hist"] for sample in samples], dim=0),
        "route_polyline": None,
        "nav_cmd": None,
        "reasoner_tok": None,
    }
    return batch
