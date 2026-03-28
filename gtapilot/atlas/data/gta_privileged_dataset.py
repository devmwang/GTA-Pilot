from __future__ import annotations

import hashlib
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
            for attr_name, manifest_key in (
                ("frame_ids_file", "frame_ids"),
                ("capture_timestamps_ns_file", "capture_timestamps_ns"),
                ("video_frame_indices_file", "video_frame_indices"),
            ):
                rel_path = getattr(manifest, attr_name)
                if rel_path:
                    arrays[manifest_key] = np.load(privileged_dir / rel_path, mmap_mode="r")
            self._validate_privileged_alignment(sample, manifest, arrays)
            self._privileged_array_cache[privileged_dir] = arrays
        return self._privileged_array_cache[privileged_dir]

    def _validate_privileged_alignment(self, sample, manifest, arrays: dict[str, np.ndarray]) -> None:
        frames = self._load_frames(sample.metadata_file)
        if manifest.frame_count != len(frames):
            raise ValueError(
                f"Privileged frame_count mismatch for {sample.clip_id}: "
                f"manifest={manifest.frame_count} source={len(frames)}"
            )
        if manifest.source_metadata_file is not None:
            if Path(manifest.source_metadata_file).resolve() != sample.metadata_file.resolve():
                raise ValueError(
                    f"Privileged source metadata mismatch for {sample.clip_id}: "
                    f"{manifest.source_metadata_file} != {sample.metadata_file}"
                )
        if manifest.source_metadata_sha1 is not None:
            metadata_sha1 = hashlib.sha1(
                sample.metadata_file.read_bytes()
            ).hexdigest()
            if metadata_sha1 != manifest.source_metadata_sha1:
                raise ValueError(
                    f"Privileged source metadata hash mismatch for {sample.clip_id}."
                )
        if manifest.track_lag_indices != list(self.cfg.geometry.track_lag_indices):
            raise ValueError(
                f"Privileged lag indices mismatch for {sample.clip_id}: "
                f"{manifest.track_lag_indices} != {list(self.cfg.geometry.track_lag_indices)}"
            )
        source_frame_ids = np.asarray([frame.frame_id for frame in frames], dtype=np.int64)
        source_timestamps = np.asarray(
            [frame.capture_timestamp_ns for frame in frames],
            dtype=np.int64,
        )
        grid_h = int(manifest.grid_height_8x)
        grid_w = int(manifest.grid_width_8x)
        if (
            grid_h != int(self.cfg.geometry.grid_h_8x)
            or grid_w != int(self.cfg.geometry.grid_w_8x)
        ):
            raise ValueError(
                f"Privileged grid shape mismatch for {sample.clip_id}: "
                f"manifest=({grid_h}, {grid_w}) "
                f"model=({self.cfg.geometry.grid_h_8x}, {self.cfg.geometry.grid_w_8x})"
            )
        expected_lags = len(manifest.track_lag_indices)
        expected_shapes = {
            "pose_delta_local": (len(frames), 3),
            "kinematics": (len(frames), 4),
            "ego_valid": (len(frames),),
            "depth_8x_m": (len(frames), grid_h, grid_w),
            "depth_valid_8x": (len(frames), grid_h, grid_w),
            "dynamic_mask_8x": (len(frames), grid_h, grid_w),
            "track_target_sparse": (len(frames), expected_lags, 2, grid_h, grid_w),
            "track_valid_sparse": (len(frames), expected_lags, grid_h, grid_w),
            "track_lag_indices": (expected_lags,),
        }
        for key, expected_shape in expected_shapes.items():
            if key not in arrays:
                raise ValueError(f"Privileged package missing required array '{key}'.")
            if tuple(int(dim) for dim in arrays[key].shape) != expected_shape:
                raise ValueError(
                    f"Privileged array shape mismatch for {sample.clip_id}:{key} "
                    f"{tuple(int(dim) for dim in arrays[key].shape)} != {expected_shape}"
                )
        if np.any(np.diff(source_frame_ids) <= 0):
            raise ValueError(f"Source frame ids are not strictly increasing for {sample.clip_id}.")
        if np.any(np.diff(source_timestamps) <= 0):
            raise ValueError(
                f"Source capture timestamps are not strictly increasing for {sample.clip_id}."
            )
        if "frame_ids" in arrays:
            frame_ids = np.asarray(arrays["frame_ids"], dtype=np.int64)
            if np.any(np.diff(frame_ids) <= 0):
                raise ValueError(
                    f"Privileged frame ids are not strictly increasing for {sample.clip_id}."
                )
            if frame_ids.shape[0] != len(frames) or not np.array_equal(frame_ids, source_frame_ids):
                raise ValueError(f"Privileged frame-id alignment mismatch for {sample.clip_id}.")
        if "capture_timestamps_ns" in arrays:
            capture_timestamps_ns = np.asarray(arrays["capture_timestamps_ns"], dtype=np.int64)
            if np.any(np.diff(capture_timestamps_ns) <= 0):
                raise ValueError(
                    f"Privileged capture timestamps are not strictly increasing for {sample.clip_id}."
                )
            if capture_timestamps_ns.shape[0] != len(frames) or not np.array_equal(
                capture_timestamps_ns,
                source_timestamps,
            ):
                raise ValueError(
                    f"Privileged capture-timestamp alignment mismatch for {sample.clip_id}."
                )
        if "video_frame_indices" in arrays:
            source_video_frame_indices = np.asarray(
                [frame.video_frame_index for frame in frames],
                dtype=np.int64,
            )
            if arrays["video_frame_indices"].shape[0] != len(frames) or not np.array_equal(
                np.asarray(arrays["video_frame_indices"], dtype=np.int64),
                source_video_frame_indices,
            ):
                raise ValueError(
                    f"Privileged video-frame alignment mismatch for {sample.clip_id}."
                )
        for key in (
            "pose_delta_local",
            "kinematics",
            "ego_valid",
            "depth_8x_m",
            "depth_valid_8x",
            "dynamic_mask_8x",
            "track_target_sparse",
            "track_valid_sparse",
        ):
            if key not in arrays:
                raise ValueError(f"Privileged array '{key}' missing for {sample.clip_id}.")
            if int(np.asarray(arrays[key]).shape[0]) != len(frames):
                raise ValueError(
                    f"Privileged array '{key}' length mismatch for {sample.clip_id}: "
                    f"{np.asarray(arrays[key]).shape[0]} != {len(frames)}"
                )
        if "track_lag_indices" not in arrays:
            raise ValueError(f"Privileged array 'track_lag_indices' missing for {sample.clip_id}.")
        lag_count = int(np.asarray(arrays["track_lag_indices"]).shape[0])
        if np.asarray(arrays["track_target_sparse"]).shape[1] != lag_count:
            raise ValueError(
                f"Privileged sparse track target lag-count mismatch for {sample.clip_id}."
            )
        if np.asarray(arrays["track_valid_sparse"]).shape[1] != lag_count:
            raise ValueError(
                f"Privileged sparse track valid lag-count mismatch for {sample.clip_id}."
            )

    def __getitem__(self, index: int) -> dict[str, Any]:
        batch = super().__getitem__(index)
        sample = self.samples[index]
        privileged_dir = self._privileged_dir_for_sample(sample)
        arrays = self._load_privileged_arrays(sample)
        recent_idx = sample.recent_frame_indices
        depth_target = torch.from_numpy(np.asarray(arrays["depth_8x_m"][recent_idx])).float().unsqueeze(1)
        depth_valid = torch.from_numpy(np.asarray(arrays["depth_valid_8x"][recent_idx])).bool()
        dynamic_mask = torch.from_numpy(np.asarray(arrays["dynamic_mask_8x"][recent_idx])).bool()
        pose_delta = torch.from_numpy(np.asarray(arrays["pose_delta_local"][recent_idx])).float()
        kinematics = torch.from_numpy(np.asarray(arrays["kinematics"][recent_idx])).float()
        pose_valid = torch.from_numpy(np.asarray(arrays["ego_valid"][recent_idx])).bool()
        sparse_track = np.asarray(arrays["track_target_sparse"][recent_idx])
        sparse_valid = np.asarray(arrays["track_valid_sparse"][recent_idx])
        lag_indices = np.asarray(arrays["track_lag_indices"]).astype(np.int64)

        batch.update(
            {
                "privileged_dir": str(privileged_dir),
                "pose_delta_recent": pose_delta,
                "kinematics_recent": kinematics,
                "pose_valid_recent": pose_valid,
                "depth_target_recent": depth_target,
                "depth_valid_recent": depth_valid,
                "dynamic_mask_recent": dynamic_mask,
                "track_target_recent_sparse": torch.from_numpy(sparse_track).float(),
                "track_valid_recent_sparse": torch.from_numpy(sparse_valid).bool(),
                "track_lag_indices": torch.from_numpy(lag_indices),
            }
        )
        return batch
