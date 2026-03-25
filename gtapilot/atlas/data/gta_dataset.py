from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .schema import LoggedStep


class AtlasLoggedStepDataset(Dataset[dict[str, Any]]):
    """
    Lightweight manifest-backed dataset for Atlas experiments.

    The manifest is expected to be JSON or JSONL in the `LoggedStep.to_dict()` shape.
    Heavy GTA-specific sequence building can be layered on top later without changing the
    canonical sample schema.
    """

    def __init__(self, manifest_path: str | Path):
        self.manifest_path = Path(manifest_path)
        self.records = self._load_manifest(self.manifest_path)

    @staticmethod
    def _load_manifest(manifest_path: Path) -> list[dict[str, Any]]:
        text = manifest_path.read_text(encoding="utf-8").strip()
        if not text:
            return []
        if text.startswith("["):
            return json.loads(text)
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    @staticmethod
    def _load_rgb(path: str) -> torch.Tensor:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Unable to load frame: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        sample = LoggedStep(
            episode_id=record["episode_id"],
            frame_idx=record["frame_idx"],
            timestamp_ms=record["timestamp_ms"],
            rgb_front_path=record["rgb_front_path"],
            action=np.asarray(record["action"], dtype=np.float32),
            dt_s=float(record["dt_s"]),
            route_polyline=None if record.get("route_polyline") is None else np.asarray(record["route_polyline"], dtype=np.float32),
            nav_cmd=None if record.get("nav_cmd") is None else np.asarray(record["nav_cmd"], dtype=np.float32),
            gt_pose_local=None if record.get("gt_pose_local") is None else np.asarray(record["gt_pose_local"], dtype=np.float32),
            gt_kinematics=None if record.get("gt_kinematics") is None else np.asarray(record["gt_kinematics"], dtype=np.float32),
            gt_occ_state=None if record.get("gt_occ_state") is None else np.asarray(record["gt_occ_state"]),
            gt_occ_sem=None if record.get("gt_occ_sem") is None else np.asarray(record["gt_occ_sem"]),
            gt_bev_lite=None if record.get("gt_bev_lite") is None else np.asarray(record["gt_bev_lite"], dtype=np.float32),
            gt_provenance=None if record.get("gt_provenance") is None else np.asarray(record["gt_provenance"]),
            gt_teacher_trajs=None if record.get("gt_teacher_trajs") is None else np.asarray(record["gt_teacher_trajs"], dtype=np.float32),
            gt_teacher_costs=None if record.get("gt_teacher_costs") is None else np.asarray(record["gt_teacher_costs"], dtype=np.float32),
            gt_teacher_best=record.get("gt_teacher_best"),
            gt_actors=record.get("gt_actors"),
            valid_mask=record.get("valid_mask", {}),
        )
        payload = sample.to_dict()
        payload["rgb_front"] = self._load_rgb(sample.rgb_front_path)
        return payload
