from __future__ import annotations

import bisect
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import AtlasConfig
from .index_cache import load_cached_index, save_cached_index
from .schema import AtlasTemporalClipIndex, BlackboxActionRecord, BlackboxFrameRecord
from .video_decode import decode_rgb_frame_union

RAW_ACTION_SOURCE = "raw_action_stream"
LEGACY_FRAME_ACTION_SOURCE = "legacy_frame_aligned"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_metadata_paths(
    recordings_root: Path,
    *,
    metadata_paths: list[str | Path] | None = None,
    split_file: str | Path | None = None,
) -> list[Path]:
    def _resolve_entry(
        raw_path: str | Path,
        *,
        split_parent: Path | None = None,
    ) -> Path:
        path = Path(raw_path)
        if path.is_absolute():
            return path.resolve()
        candidates: list[Path] = []
        if split_parent is not None:
            candidates.append(split_parent / path)
        candidates.append(recordings_root / path)
        candidates.append(path)
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        return candidates[0].resolve()

    if metadata_paths is not None:
        return [_resolve_entry(path) for path in metadata_paths]
    if split_file is not None:
        split_path = Path(split_file).resolve()
        lines = split_path.read_text(encoding="utf-8").splitlines()
        return [
            _resolve_entry(line.strip(), split_parent=split_path.parent)
            for line in lines
            if line.strip()
        ]
    return sorted(recordings_root.glob("capture_*_metadata.json"))


def _interval_ns(hz: int | float) -> int:
    return max(1, int(round(1_000_000_000.0 / max(1.0, float(hz)))))


def _metadata_stat_payload(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "mtime_ns": int(stat.st_mtime_ns),
        "size": int(stat.st_size),
    }


def _optional_path_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": str(path.resolve()),
            "missing": True,
        }
    return _metadata_stat_payload(path)


def _source_dt_seconds(
    timestamps_ns: Sequence[int],
    index: int,
    *,
    default_dt_s: float,
) -> float:
    if index > 0:
        return max(
            0.0,
            float(timestamps_ns[index] - timestamps_ns[index - 1]) / 1_000_000_000.0,
        )
    return float(default_dt_s)


def _selected_dt_tensor(
    timestamps_ns: Sequence[int],
    indices: Sequence[int],
    *,
    default_dt_s: float,
    prev_index: int | None = None,
) -> torch.Tensor:
    values: list[float] = []
    for offset, source_index in enumerate(indices):
        if offset > 0:
            prev_index = indices[offset - 1]
            values.append(
                max(
                    0.0,
                    float(timestamps_ns[source_index] - timestamps_ns[prev_index])
                    / 1_000_000_000.0,
                )
            )
            continue
        values.append(
            max(
                0.0,
                float(timestamps_ns[source_index] - timestamps_ns[prev_index]) / 1_000_000_000.0,
            )
            if prev_index is not None and prev_index >= 0
            else _source_dt_seconds(
                timestamps_ns,
                source_index,
                default_dt_s=default_dt_s,
            )
        )
    return torch.tensor(values, dtype=torch.float32).unsqueeze(-1)


def _select_latest_indices(
    source_timestamps_ns: list[int],
    desired_timestamps_ns: list[int],
) -> list[int] | None:
    selected: list[int] = []
    for desired_ts in desired_timestamps_ns:
        source_index = bisect.bisect_right(
            source_timestamps_ns,
            desired_ts,
            0,
            len(source_timestamps_ns),
        ) - 1
        if source_index < 0:
            return None
        selected.append(source_index)
    return selected


def _frame_action_vectors(
    frames: list[BlackboxFrameRecord],
    indices: list[int],
) -> torch.Tensor:
    return torch.tensor(
        np.stack([frames[source_index].action_vector for source_index in indices]),
        dtype=torch.float32,
    )


def _stream_action_vectors(
    actions: list[BlackboxActionRecord],
    indices: list[int],
) -> torch.Tensor:
    return torch.tensor(
        np.stack([actions[source_index].action_vector for source_index in indices]),
        dtype=torch.float32,
    )


def _effective_action_timestamps_ns(actions: list[BlackboxActionRecord]) -> list[int]:
    return [
        action.message_timestamp_ns
        if action.message_timestamp_ns > 0
        else action.publish_timestamp_ns
        for action in actions
    ]


class AtlasBlackboxClipDataset(Dataset[dict[str, Any]]):
    """
    Builds Atlas temporal training samples from blackbox MKV+metadata recordings.

    Samples are resampled from the source capture timeline onto the student or
    teacher model grids defined by AtlasConfig.
    """

    def __init__(
        self,
        recordings_root: str | Path,
        cfg: AtlasConfig,
        *,
        metadata_paths: list[str | Path] | None = None,
        split_file: str | Path | None = None,
        index_cache_dir: str | Path | None = None,
        recent_steps: int | None = None,
        older_steps: int | None = None,
        mid_steps: int | None = None,
        action_steps: int | None = None,
        anchor_stride_steps: int = 1,
        max_samples_per_clip_per_epoch: int | None = None,
        require_privileged: bool = False,
    ):
        self.recordings_root = Path(recordings_root)
        self.cfg = cfg
        self.recent_steps = recent_steps or cfg.temporal.recent_full_frames
        self.older_steps = older_steps or cfg.temporal.older_compressed_frames
        self.mid_steps = mid_steps or cfg.temporal.mid_summary_frames
        self.action_steps = action_steps or cfg.action.history_len
        self.anchor_stride_steps = max(1, int(anchor_stride_steps))
        self.max_samples_per_clip_per_epoch = (
            None
            if max_samples_per_clip_per_epoch is None
            else max(1, int(max_samples_per_clip_per_epoch))
        )
        self.model_hz = int(cfg.temporal.fast_loop_hz)
        self.mid_hz = int(cfg.temporal.mid_summary_hz)
        self.index_cache_dir = (
            self.recordings_root / ".atlas-index-cache"
            if index_cache_dir is None
            else Path(index_cache_dir)
        )
        self.require_privileged = require_privileged
        self.metadata_paths = _resolve_metadata_paths(
            self.recordings_root,
            metadata_paths=metadata_paths,
            split_file=split_file,
        )
        self._manifest_cache: dict[Path, dict[str, Any]] = {}
        self._frame_cache: dict[Path, list[BlackboxFrameRecord]] = {}
        self._action_cache: dict[Path, list[BlackboxActionRecord]] = {}
        self.samples = self._build_indices()

    def _cache_key_payload(self) -> dict[str, Any]:
        payload = {
            "metadata": [_metadata_stat_payload(path) for path in self.metadata_paths],
            "cache_schema": "atlas_stage1_time_grid_v2",
            "recent_steps": self.recent_steps,
            "older_steps": self.older_steps,
            "mid_steps": self.mid_steps,
            "action_steps": self.action_steps,
            "anchor_stride_steps": self.anchor_stride_steps,
            "max_samples_per_clip_per_epoch": self.max_samples_per_clip_per_epoch,
            "model_hz": self.model_hz,
            "mid_hz": self.mid_hz,
            "action_source_policy": {
                str(path.resolve()): self._action_source_for_manifest(path)
                for path in self.metadata_paths
            },
        }
        if self.require_privileged:
            payload["privileged_manifests"] = [
                _optional_path_payload(
                    path.with_name(f"{path.stem.replace('_metadata', '')}_privileged")
                    / "manifest.json"
                )
                for path in self.metadata_paths
            ]
        return payload

    def _load_manifest(self, metadata_path: Path) -> dict[str, Any]:
        if metadata_path not in self._manifest_cache:
            self._manifest_cache[metadata_path] = _read_json(metadata_path)
        return self._manifest_cache[metadata_path]

    def _load_frames(self, metadata_path: Path) -> list[BlackboxFrameRecord]:
        if metadata_path not in self._frame_cache:
            manifest = self._load_manifest(metadata_path)
            self._frame_cache[metadata_path] = [
                BlackboxFrameRecord.from_dict(frame_payload)
                for frame_payload in manifest.get("frames", [])
            ]
        return self._frame_cache[metadata_path]

    def _load_actions(self, metadata_path: Path) -> list[BlackboxActionRecord]:
        if metadata_path not in self._action_cache:
            manifest = self._load_manifest(metadata_path)
            records: list[BlackboxActionRecord] = []
            for action_payload in manifest.get("actions", []):
                record = BlackboxActionRecord.from_dict(action_payload)
                if record.message_timestamp_ns > 0 or record.publish_timestamp_ns > 0:
                    records.append(record)
            records.sort(
                key=lambda action: action.message_timestamp_ns
                if action.message_timestamp_ns > 0
                else action.publish_timestamp_ns
            )
            self._action_cache[metadata_path] = records
        return self._action_cache[metadata_path]

    def _action_source_for_manifest(self, metadata_path: Path) -> str:
        if self._load_actions(metadata_path):
            return RAW_ACTION_SOURCE
        return LEGACY_FRAME_ACTION_SOURCE

    def _sample_indices_for_clip(
        self,
        *,
        metadata_path: Path,
        frames: list[BlackboxFrameRecord],
        actions: list[BlackboxActionRecord],
        nominal_fps: float,
    ) -> list[AtlasTemporalClipIndex]:
        clip_id = metadata_path.stem.replace("_metadata", "")
        manifest = self._load_manifest(metadata_path)
        video_path = metadata_path.with_name(str(manifest["video_file_name"]))
        frame_timestamps_ns = [frame.capture_timestamp_ns for frame in frames]
        model_step_ns = _interval_ns(self.model_hz)
        mid_step_ns = _interval_ns(self.mid_hz)
        grid_origin_ns = frame_timestamps_ns[0]
        last_anchor_step = max(0, (frame_timestamps_ns[-1] - grid_origin_ns) // model_step_ns)
        privileged_dir = metadata_path.with_name(f"{clip_id}_privileged")
        privileged_dir_str = str(privileged_dir) if (privileged_dir / "manifest.json").exists() else None
        if self.require_privileged and privileged_dir_str is None:
            return []
        samples: list[AtlasTemporalClipIndex] = []

        for anchor_step in range(0, int(last_anchor_step) + 1, self.anchor_stride_steps):
            anchor_timestamp_ns = grid_origin_ns + anchor_step * model_step_ns
            target_frame_index = bisect.bisect_right(frame_timestamps_ns, anchor_timestamp_ns) - 1
            if target_frame_index < 0:
                continue

            recent_timestamps_ns = [
                anchor_timestamp_ns - (self.recent_steps - 1 - offset) * model_step_ns
                for offset in range(self.recent_steps)
            ]
            older_timestamps_ns = [
                anchor_timestamp_ns
                - (self.recent_steps + self.older_steps - 1 - offset) * model_step_ns
                for offset in range(self.older_steps)
            ]
            mid_timestamps_ns = [
                anchor_timestamp_ns
                - (self.recent_steps + self.older_steps - 1) * model_step_ns
                - (self.mid_steps - offset) * mid_step_ns
                for offset in range(self.mid_steps)
            ]
            action_grid_timestamps_ns = [
                anchor_timestamp_ns - (self.action_steps - 1 - offset) * model_step_ns
                for offset in range(self.action_steps)
            ]

            recent_frame_indices = _select_latest_indices(frame_timestamps_ns, recent_timestamps_ns)
            if recent_frame_indices is None:
                continue
            older_frame_indices = _select_latest_indices(
                frame_timestamps_ns,
                older_timestamps_ns,
            )
            if older_frame_indices is None:
                continue
            mid_frame_indices = _select_latest_indices(
                frame_timestamps_ns,
                mid_timestamps_ns,
            )
            if mid_frame_indices is None:
                continue

            if actions:
                action_timestamps_ns = _effective_action_timestamps_ns(actions)
                action_entry_indices = _select_latest_indices(
                    action_timestamps_ns,
                    action_grid_timestamps_ns,
                )
                action_source = RAW_ACTION_SOURCE
            else:
                action_timestamps_ns = frame_timestamps_ns
                action_entry_indices = _select_latest_indices(
                    action_timestamps_ns,
                    action_grid_timestamps_ns,
                )
                action_source = LEGACY_FRAME_ACTION_SOURCE
            if action_entry_indices is None:
                continue
            if action_source == RAW_ACTION_SOURCE:
                selected_action_timestamps_ns = [
                    action_timestamps_ns[action_index]
                    for action_index in action_entry_indices
                ]
                stale_limit_ns = max(model_step_ns * 3, 100_000_000)
                if any(
                    desired_ts < selected_ts
                    or (desired_ts - selected_ts) > stale_limit_ns
                    for desired_ts, selected_ts in zip(
                        action_grid_timestamps_ns,
                        selected_action_timestamps_ns,
                    )
                ):
                    continue

            samples.append(
                AtlasTemporalClipIndex(
                    clip_id=clip_id,
                    metadata_path=str(metadata_path),
                    video_path=str(video_path),
                    anchor_timestamp_ns=int(anchor_timestamp_ns),
                    target_frame_index=int(target_frame_index),
                    recent_frame_indices=recent_frame_indices,
                    older_frame_indices=older_frame_indices,
                    mid_frame_indices=mid_frame_indices,
                    action_entry_indices=action_entry_indices,
                    action_source=action_source,
                    nominal_fps=nominal_fps,
                    frame_source=frames[target_frame_index].frame_source,
                    privileged_dir=privileged_dir_str,
                )
            )
        if (
            self.max_samples_per_clip_per_epoch is not None
            and len(samples) > self.max_samples_per_clip_per_epoch
        ):
            select = np.linspace(
                0,
                len(samples) - 1,
                num=self.max_samples_per_clip_per_epoch,
                dtype=np.int64,
            )
            samples = [samples[int(sample_index)] for sample_index in select.tolist()]
        return samples

    def _build_indices(self) -> list[AtlasTemporalClipIndex]:
        cached = load_cached_index(self.index_cache_dir, self._cache_key_payload())
        if cached is not None:
            return [
                AtlasTemporalClipIndex.from_dict(sample_payload)
                for sample_payload in cached
            ]

        samples: list[AtlasTemporalClipIndex] = []
        for metadata_path in self.metadata_paths:
            frames = self._load_frames(metadata_path)
            if not frames:
                continue

            manifest = self._load_manifest(metadata_path)
            nominal_fps = float(manifest.get("video_nominal_fps", 0.0) or 0.0)
            if nominal_fps <= 0.0:
                nominal_fps = float(frames[0].frame_metadata.get("nominal_fps", 60.0))
            actions = self._load_actions(metadata_path)
            samples.extend(
                self._sample_indices_for_clip(
                    metadata_path=metadata_path,
                    frames=frames,
                    actions=actions,
                    nominal_fps=nominal_fps,
                )
            )

        save_cached_index(
            self.index_cache_dir,
            self._cache_key_payload(),
            [sample.to_dict() for sample in samples],
        )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _load_rgb_frames(
        self,
        frames: list[BlackboxFrameRecord],
        video_path: Path,
        recent_indices: list[int],
        older_indices: list[int],
        mid_indices: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        decoded = decode_rgb_frame_union(
            video_path,
            sorted(
                {
                    max(0, int(frames[source_index].video_frame_index) - 1)
                    for source_index in recent_indices + older_indices + mid_indices
                }
            ),
        )

        def _stack(indices: list[int]) -> torch.Tensor:
            return torch.stack(
                [
                    decoded[max(0, int(frames[source_index].video_frame_index) - 1)]
                    for source_index in indices
                ],
                dim=0,
            )

        return _stack(recent_indices), _stack(older_indices), _stack(mid_indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        frames = self._load_frames(sample.metadata_file)
        actions = self._load_actions(sample.metadata_file)
        frame_timestamps_ns = [frame.capture_timestamp_ns for frame in frames]
        model_period_s = 1.0 / max(1.0, float(self.model_hz))
        mid_period_s = 1.0 / max(1.0, float(self.mid_hz))
        rgb_recent, rgb_older, rgb_mid = self._load_rgb_frames(
            frames,
            sample.video_file,
            sample.recent_frame_indices,
            sample.older_frame_indices,
            sample.mid_frame_indices,
        )

        batch = {
            "clip_id": sample.clip_id,
            "metadata_path": sample.metadata_path,
            "video_path": sample.video_path,
            "privileged_dir": sample.privileged_dir,
            "anchor_timestamp_ns": sample.anchor_timestamp_ns,
            "target_frame_index": sample.target_frame_index,
            "frame_source": sample.frame_source,
            "action_source": sample.action_source,
            "rgb_recent": rgb_recent,
            "dt_recent": _selected_dt_tensor(
                frame_timestamps_ns,
                sample.recent_frame_indices,
                default_dt_s=model_period_s,
                prev_index=sample.older_frame_indices[-1] if sample.older_frame_indices else None,
            ),
            "rgb_older": rgb_older,
            "dt_older": _selected_dt_tensor(
                frame_timestamps_ns,
                sample.older_frame_indices,
                default_dt_s=model_period_s,
                prev_index=sample.mid_frame_indices[-1] if sample.mid_frame_indices else None,
            ),
            "rgb_mid": rgb_mid,
            "dt_mid": _selected_dt_tensor(
                frame_timestamps_ns,
                sample.mid_frame_indices,
                default_dt_s=mid_period_s,
            ),
            "route_polyline": None,
            "nav_cmd": None,
            "reasoner_tok": None,
        }
        if sample.action_source == RAW_ACTION_SOURCE:
            action_timestamps_ns = _effective_action_timestamps_ns(actions)
            batch["actions_hist"] = _stream_action_vectors(actions, sample.action_entry_indices)
            batch["dt_hist"] = _selected_dt_tensor(
                action_timestamps_ns,
                sample.action_entry_indices,
                default_dt_s=model_period_s,
            )
        else:
            batch["actions_hist"] = _frame_action_vectors(frames, sample.action_entry_indices)
            batch["dt_hist"] = _selected_dt_tensor(
                frame_timestamps_ns,
                sample.action_entry_indices,
                default_dt_s=model_period_s,
            )
        return batch
