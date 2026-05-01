from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .config import AtlasHAConfig


REQUIRED_TARGET_KEYS: tuple[str, ...] = (
    "target_traj",
    "target_candidate",
    "target_ego",
    "target_lead_present",
    "target_lead_state",
    "target_lane_lat",
    "target_lane_valid",
    "target_lane_conf",
)

RECOMMENDED_TARGET_KEYS: tuple[str, ...] = (
    "target_road_edge_lat",
    "target_road_edge_valid",
    "target_adjacent_left_visible",
    "target_adjacent_right_visible",
    "target_lane_change_teacher_ok",
    "target_soft_scene_type",
    "target_takeover_required",
    "target_legal_speed",
    "target_candidate_legal",
    "target_bev",
    "ui_mask",
)


@dataclass(slots=True)
class AtlasHASample:
    scene_rgb_current: torch.Tensor
    actions_hist: torch.Tensor
    dt_hist: torch.Tensor
    nav_cmd: torch.Tensor
    targets: dict[str, torch.Tensor]
    metadata: dict[str, Any]


def _tensor_from_payload(payload: Any, base_dir: Path) -> torch.Tensor:
    if isinstance(payload, str):
        path = (base_dir / payload).resolve()
        if path.suffix == ".pt":
            value = torch.load(path, map_location="cpu")
            if isinstance(value, torch.Tensor):
                return value
            raise ValueError(f"Expected tensor in {path}.")
        raise ValueError(f"Unsupported tensor path suffix {path.suffix!r}.")
    return torch.as_tensor(payload)


def validate_sample_shapes(sample: AtlasHASample, cfg: AtlasHAConfig) -> None:
    rgb = sample.scene_rgb_current
    if rgb.shape != (3, cfg.input_h, cfg.input_w):
        raise ValueError(f"scene_rgb_current must be [3,{cfg.input_h},{cfg.input_w}], got {tuple(rgb.shape)}.")
    if sample.actions_hist.shape != (cfg.action_history_steps, cfg.action_dim):
        raise ValueError("actions_hist must be [M,6].")
    if sample.dt_hist.shape != (cfg.action_history_steps, 1):
        raise ValueError("dt_hist must be [M,1].")
    if sample.nav_cmd.shape != (cfg.nav_cmd_dim,):
        raise ValueError("nav_cmd must be [6].")

    targets = sample.targets
    for key in REQUIRED_TARGET_KEYS:
        if key not in targets:
            raise ValueError(f"Missing required Atlas-HA target {key!r}.")
    _expect(targets["target_traj"], (cfg.num_traj_points, 4), "target_traj")
    if targets["target_candidate"].numel() != 1:
        raise ValueError("target_candidate must be scalar.")
    _expect(targets["target_ego"], (4,), "target_ego")
    _expect(targets["target_lead_present"], (1,), "target_lead_present")
    _expect(targets["target_lead_state"], (5,), "target_lead_state")
    _expect(targets["target_lane_lat"], (3, cfg.num_lane_points), "target_lane_lat")
    _expect(targets["target_lane_valid"], (3, cfg.num_lane_points), "target_lane_valid")
    _expect(targets["target_lane_conf"], (3,), "target_lane_conf")
    if "target_legal_speed" in targets:
        _expect(targets["target_legal_speed"], (cfg.num_traj_points,), "target_legal_speed")
    if "target_candidate_legal" in targets:
        _expect(
            targets["target_candidate_legal"],
            (cfg.num_traj_candidates,),
            "target_candidate_legal",
        )
    if "ui_mask" in targets:
        _expect(targets["ui_mask"], (1, cfg.input_h, cfg.input_w), "ui_mask")


def _expect(tensor: torch.Tensor, shape: tuple[int, ...], name: str) -> None:
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}.")


class AtlasHAJsonlDataset(Dataset[AtlasHASample]):
    def __init__(
        self,
        samples_jsonl: str | Path,
        cfg: AtlasHAConfig | None = None,
        *,
        validate: bool = True,
    ):
        self.cfg = cfg or AtlasHAConfig()
        self.path = Path(samples_jsonl)
        self.base_dir = self.path.parent
        self.rows = [
            json.loads(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.validate = validate

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> AtlasHASample:
        row = self.rows[index]
        targets = {
            key: _tensor_from_payload(value, self.base_dir).float()
            for key, value in dict(row.get("targets", {})).items()
        }
        sample = AtlasHASample(
            scene_rgb_current=_tensor_from_payload(row["scene_rgb_current"], self.base_dir).float(),
            actions_hist=_tensor_from_payload(row["actions_hist"], self.base_dir).float(),
            dt_hist=_tensor_from_payload(row["dt_hist"], self.base_dir).float(),
            nav_cmd=_tensor_from_payload(row.get("nav_cmd", [1, 0, 0, 0, 0, 0]), self.base_dir).float(),
            targets=targets,
            metadata=dict(row.get("metadata", {})),
        )
        if self.validate:
            validate_sample_shapes(sample, self.cfg)
        return sample


def collate_atlas_ha_samples(samples: list[AtlasHASample]) -> dict[str, Any]:
    if not samples:
        raise ValueError("Cannot collate an empty Atlas-HA batch.")
    target_keys = set.intersection(*(set(sample.targets) for sample in samples))
    return {
        "scene_rgb_current": torch.stack([sample.scene_rgb_current for sample in samples], dim=0),
        "actions_hist": torch.stack([sample.actions_hist for sample in samples], dim=0),
        "dt_hist": torch.stack([sample.dt_hist for sample in samples], dim=0),
        "nav_cmd": torch.stack([sample.nav_cmd for sample in samples], dim=0),
        "targets": {
            key: torch.stack([sample.targets[key] for sample in samples], dim=0)
            for key in sorted(target_keys)
        },
        "metadata": [sample.metadata for sample in samples],
    }
