from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..config import AtlasConfig
from .gta_dataset import AtlasBlackboxClipDataset
from .privileged_schema import PrivilegedClipManifest


class AtlasPrivilegedClipDataset(AtlasBlackboxClipDataset):
    def __init__(
        self,
        recordings_root: str | Path,
        cfg: AtlasConfig,
        **kwargs: Any,
    ):
        kwargs.setdefault("require_privileged", True)
        super().__init__(recordings_root, cfg, **kwargs)
        self._privileged_manifest_cache: dict[Path, PrivilegedClipManifest] = {}
        self._privileged_array_cache: dict[Path, dict[str, np.ndarray]] = {}

    def _privileged_dir_for_sample(self, sample) -> Path:
        if sample.privileged_path is None:
            raise FileNotFoundError(
                f"Privileged data missing for clip {sample.clip_id}: {sample.metadata_path}"
            )
        return sample.privileged_path

    def _load_privileged_manifest(self, sample) -> PrivilegedClipManifest:
        privileged_dir = self._privileged_dir_for_sample(sample)
        manifest_path = privileged_dir / "manifest.json"
        if manifest_path not in self._privileged_manifest_cache:
            self._privileged_manifest_cache[manifest_path] = PrivilegedClipManifest.load(
                manifest_path
            )
        return self._privileged_manifest_cache[manifest_path]

    def _load_privileged_arrays(self, sample) -> dict[str, np.ndarray]:
        privileged_dir = self._privileged_dir_for_sample(sample)
        if privileged_dir not in self._privileged_array_cache:
            manifest = self._load_privileged_manifest(sample)
            arrays: dict[str, np.ndarray] = {}
            for key, rel_path in manifest.files.items():
                arrays[key] = np.load(privileged_dir / rel_path, mmap_mode="r")
            self._privileged_array_cache[privileged_dir] = arrays
        return self._privileged_array_cache[privileged_dir]

    def __getitem__(self, index: int) -> dict[str, Any]:
        batch = super().__getitem__(index)
        sample = self.samples[index]
        privileged_dir = self._privileged_dir_for_sample(sample)
        arrays = self._load_privileged_arrays(sample)
        recent_idx = sample.recent_frame_indices
        track_history = self.cfg.geometry.track_history
        depth_target = torch.from_numpy(np.asarray(arrays["depth_8x_m"][recent_idx])).float().unsqueeze(1)
        depth_valid = torch.from_numpy(np.asarray(arrays["depth_valid_8x"][recent_idx])).bool()
        dynamic_mask = torch.from_numpy(np.asarray(arrays["dynamic_mask_8x"][recent_idx])).bool()
        pose_delta = torch.from_numpy(np.asarray(arrays["pose_delta_local"][recent_idx])).float()
        kinematics = torch.from_numpy(np.asarray(arrays["kinematics"][recent_idx])).float()
        pose_valid = torch.from_numpy(np.asarray(arrays["ego_valid"][recent_idx])).bool()
        sparse_track = np.asarray(arrays["track_target_sparse"][recent_idx])
        sparse_valid = np.asarray(arrays["track_valid_sparse"][recent_idx])
        lag_indices = np.asarray(arrays["track_lag_indices"]).astype(np.int64)
        dense_track = np.zeros(
            (
                len(recent_idx),
                track_history,
                2,
                sparse_track.shape[-2],
                sparse_track.shape[-1],
            ),
            dtype=np.float32,
        )
        dense_valid = np.zeros(
            (
                len(recent_idx),
                track_history,
                sparse_valid.shape[-2],
                sparse_valid.shape[-1],
            ),
            dtype=bool,
        )
        for sparse_idx, lag in enumerate(lag_indices.tolist()):
            if lag <= 0 or lag > track_history:
                continue
            dense_track[:, lag - 1] = sparse_track[:, sparse_idx]
            dense_valid[:, lag - 1] = sparse_valid[:, sparse_idx]

        batch.update(
            {
                "privileged_dir": str(privileged_dir),
                "pose_delta_recent": pose_delta,
                "kinematics_recent": kinematics,
                "pose_valid_recent": pose_valid,
                "depth_target_recent": depth_target,
                "depth_valid_recent": depth_valid,
                "dynamic_mask_recent": dynamic_mask,
                "track_target_recent": torch.from_numpy(dense_track).float(),
                "track_valid_recent": torch.from_numpy(dense_valid).bool(),
                "track_lag_indices": torch.from_numpy(lag_indices),
            }
        )
        return batch
