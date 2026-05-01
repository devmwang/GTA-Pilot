from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import AtlasConfig
from .index_cache import load_cached_index, save_cached_index
from .schema import AtlasTemporalClipIndex
from .video_decode import decode_rgb_frame_union

RAW_ACTION_SOURCE = "raw_action_stream"


@dataclass(slots=True)
class _ClipTimeline:
    clip_id: str
    metadata_path: Path
    actions_path: Path
    video_path: Path
    privileged_dir: Path | None
    nominal_fps: float
    frame_source: str
    frame_timestamps_ns: np.ndarray
    frame_ids: np.ndarray
    video_frame_indices: np.ndarray
    action_timestamps_ns: np.ndarray
    action_vectors: np.ndarray


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
    return sorted(recordings_root.glob("capture_*/metadata.json"))


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
    source_timestamps_ns: Sequence[int] | np.ndarray,
    desired_timestamps_ns: Sequence[int] | np.ndarray,
) -> np.ndarray | None:
    desired = np.asarray(desired_timestamps_ns, dtype=np.int64)
    if desired.size == 0:
        return np.zeros((0,), dtype=np.int64)
    source = np.asarray(source_timestamps_ns, dtype=np.int64)
    indices = np.searchsorted(source, desired, side="right") - 1
    if np.any(indices < 0):
        return None
    return indices.astype(np.int64, copy=False)


def _stream_action_vectors(
    action_vectors: np.ndarray,
    indices: Sequence[int] | np.ndarray,
) -> torch.Tensor:
    return torch.from_numpy(np.asarray(action_vectors[np.asarray(indices, dtype=np.int64)])).float()


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
        self._clip_cache: dict[Path, _ClipTimeline] = {}
        self.samples = self._build_indices()

    def _cache_key_payload(self) -> dict[str, Any]:
        payload = {
            "metadata": [_metadata_stat_payload(path) for path in self.metadata_paths],
            "actions": [
                _metadata_stat_payload(path.parent / "actions.json")
                for path in self.metadata_paths
            ],
            "cache_schema": "atlas_stage1_time_grid_v3_anchor_only",
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
                _optional_path_payload(path.parent / "privileged" / "manifest.json")
                for path in self.metadata_paths
            ]
        return payload

    def _load_clip_timeline(self, metadata_path: Path) -> _ClipTimeline:
        metadata_path = metadata_path.resolve()
        timeline = self._clip_cache.get(metadata_path)
        if timeline is not None:
            return timeline

        manifest = _read_json(metadata_path)
        if int(manifest.get("schema_version", 0)) != 3:
            raise ValueError(
                f"Unsupported blackbox manifest schema for Atlas: {metadata_path}"
            )
        if str(manifest.get("kind")) != "video_metadata":
            raise ValueError(f"Unsupported video metadata kind for Atlas: {metadata_path}")
        capture_session = dict(manifest.get("capture_session", {}))
        video_session = dict(manifest.get("video_session", {}))
        frame_payloads = manifest.get("frames", [])
        frame_count = len(frame_payloads)
        clip_id = str(manifest.get("clip_id", metadata_path.parent.name))
        actions_path = metadata_path.parent / str(manifest.get("actions_file", "actions.json"))
        if not actions_path.exists():
            raise FileNotFoundError(f"Actions metadata missing for clip {clip_id}: {actions_path}")
        actions_manifest = _read_json(actions_path)
        if int(actions_manifest.get("schema_version", 0)) != 3:
            raise ValueError(f"Unsupported actions manifest schema for Atlas: {actions_path}")
        if str(actions_manifest.get("kind")) != "actions":
            raise ValueError(f"Unsupported actions metadata kind for Atlas: {actions_path}")
        if str(actions_manifest.get("clip_id", "")) != clip_id:
            raise ValueError(f"Actions metadata clip_id mismatch for {clip_id}: {actions_path}")
        video_file_name = str(video_session.get("file_name", ""))
        if not video_file_name:
            raise ValueError(f"Missing video_session.file_name in {metadata_path}")
        video_path = (metadata_path.parent / video_file_name).resolve()
        privileged_dir = (metadata_path.parent / "privileged").resolve()
        privileged_path = privileged_dir if (privileged_dir / "manifest.json").exists() else None

        frame_timestamps_ns = np.fromiter(
            (int(frame_payload["capture_timestamp_ns"]) for frame_payload in frame_payloads),
            dtype=np.int64,
            count=frame_count,
        )
        frame_ids = np.fromiter(
            (int(frame_payload.get("frame_id", -1)) for frame_payload in frame_payloads),
            dtype=np.int64,
            count=frame_count,
        )
        video_frame_indices = np.fromiter(
            (int(frame_payload["video_frame_index"]) for frame_payload in frame_payloads),
            dtype=np.int64,
            count=frame_count,
        )
        frame_source = str(capture_session.get("capture_source", ""))
        nominal_fps = float(
            video_session.get(
                "nominal_fps",
                capture_session.get("nominal_fps", 60.0),
            )
            or 60.0
        )

        action_payloads = actions_manifest.get("actions", [])
        raw_action_timestamps: list[int] = []
        raw_action_vectors: list[list[float]] = []
        for action_payload in action_payloads:
            timestamp_ns = int(
                action_payload.get("message_timestamp_ns", 0)
                or action_payload.get("publish_timestamp_ns", 0)
                or 0
            )
            if timestamp_ns <= 0:
                continue
            raw_payload = action_payload.get("payload") or {}
            raw_action_timestamps.append(timestamp_ns)
            raw_action_vectors.append(
                [
                    float(raw_payload.get("steer", 0.0)),
                    float(raw_payload.get("throttle", 0.0)),
                    float(raw_payload.get("brake", 0.0)),
                    float(raw_payload.get("handbrake", 0.0)),
                    float(raw_payload.get("reverse", 0.0)),
                    float(raw_payload.get("pilot_active", 0.0)),
                ]
            )
        if not raw_action_timestamps:
            raise ValueError(f"Actions metadata contains no usable actions for clip {clip_id}.")
        action_timestamps_ns = np.asarray(raw_action_timestamps, dtype=np.int64)
        action_vectors = np.asarray(raw_action_vectors, dtype=np.float32)
        order = np.argsort(action_timestamps_ns, kind="stable")
        action_timestamps_ns = action_timestamps_ns[order]
        action_vectors = action_vectors[order]

        timeline = _ClipTimeline(
            clip_id=clip_id,
            metadata_path=metadata_path,
            actions_path=actions_path.resolve(),
            video_path=video_path,
            privileged_dir=privileged_path,
            nominal_fps=nominal_fps,
            frame_source=frame_source,
            frame_timestamps_ns=frame_timestamps_ns,
            frame_ids=frame_ids,
            video_frame_indices=video_frame_indices,
            action_timestamps_ns=action_timestamps_ns,
            action_vectors=action_vectors,
        )
        self._clip_cache[metadata_path] = timeline
        return timeline

    def _action_source_for_manifest(self, metadata_path: Path) -> str:
        self._load_clip_timeline(metadata_path)
        return RAW_ACTION_SOURCE

    def _sample_indices_for_clip(
        self,
        *,
        timeline: _ClipTimeline,
    ) -> list[AtlasTemporalClipIndex]:
        frame_timestamps_ns = timeline.frame_timestamps_ns
        model_step_ns = _interval_ns(self.model_hz)
        mid_step_ns = _interval_ns(self.mid_hz)
        grid_origin_ns = int(frame_timestamps_ns[0])
        last_anchor_step = max(
            0,
            int((int(frame_timestamps_ns[-1]) - grid_origin_ns) // model_step_ns),
        )
        privileged_dir_str = (
            None if timeline.privileged_dir is None else str(timeline.privileged_dir)
        )
        if self.require_privileged and privileged_dir_str is None:
            return []
        samples: list[AtlasTemporalClipIndex] = []
        action_timestamps_ns = timeline.action_timestamps_ns

        for anchor_step in range(0, int(last_anchor_step) + 1, self.anchor_stride_steps):
            anchor_timestamp_ns = grid_origin_ns + anchor_step * model_step_ns

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

            action_entry_indices = _select_latest_indices(
                action_timestamps_ns,
                action_grid_timestamps_ns,
            )
            if action_entry_indices is None:
                continue
            selected_action_timestamps_ns = [
                int(action_timestamps_ns[action_index])
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
                    clip_id=timeline.clip_id,
                    metadata_path=str(timeline.metadata_path),
                    actions_path=str(timeline.actions_path),
                    video_path=str(timeline.video_path),
                    anchor_timestamp_ns=int(anchor_timestamp_ns),
                    action_source=RAW_ACTION_SOURCE,
                    nominal_fps=timeline.nominal_fps,
                    frame_source=timeline.frame_source,
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
            timeline = self._load_clip_timeline(metadata_path)
            if timeline.frame_timestamps_ns.size == 0:
                continue
            samples.extend(
                self._sample_indices_for_clip(
                    timeline=timeline,
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

    def _resolve_sample_indices(
        self,
        sample: AtlasTemporalClipIndex,
        timeline: _ClipTimeline,
    ) -> dict[str, np.ndarray | int]:
        frame_timestamps_ns = timeline.frame_timestamps_ns
        model_step_ns = _interval_ns(self.model_hz)
        mid_step_ns = _interval_ns(self.mid_hz)
        anchor_timestamp_ns = int(sample.anchor_timestamp_ns)

        recent_timestamps_ns = np.asarray(
            [
                anchor_timestamp_ns - (self.recent_steps - 1 - offset) * model_step_ns
                for offset in range(self.recent_steps)
            ],
            dtype=np.int64,
        )
        older_timestamps_ns = np.asarray(
            [
                anchor_timestamp_ns
                - (self.recent_steps + self.older_steps - 1 - offset) * model_step_ns
                for offset in range(self.older_steps)
            ],
            dtype=np.int64,
        )
        mid_timestamps_ns = np.asarray(
            [
                anchor_timestamp_ns
                - (self.recent_steps + self.older_steps - 1) * model_step_ns
                - (self.mid_steps - offset) * mid_step_ns
                for offset in range(self.mid_steps)
            ],
            dtype=np.int64,
        )
        action_grid_timestamps_ns = np.asarray(
            [
                anchor_timestamp_ns - (self.action_steps - 1 - offset) * model_step_ns
                for offset in range(self.action_steps)
            ],
            dtype=np.int64,
        )

        target_frame_index = int(
            np.searchsorted(frame_timestamps_ns, anchor_timestamp_ns, side="right") - 1
        )
        if target_frame_index < 0:
            raise IndexError(f"Invalid Atlas sample anchor for clip {sample.clip_id}.")

        recent_frame_indices = _select_latest_indices(frame_timestamps_ns, recent_timestamps_ns)
        older_frame_indices = _select_latest_indices(frame_timestamps_ns, older_timestamps_ns)
        mid_frame_indices = _select_latest_indices(frame_timestamps_ns, mid_timestamps_ns)
        action_timestamps_ns: Sequence[int] | np.ndarray = timeline.action_timestamps_ns
        action_entry_indices = _select_latest_indices(action_timestamps_ns, action_grid_timestamps_ns)
        if (
            recent_frame_indices is None
            or older_frame_indices is None
            or mid_frame_indices is None
            or action_entry_indices is None
        ):
            raise IndexError(f"Atlas sample became invalid for clip {sample.clip_id}.")

        return {
            "target_frame_index": target_frame_index,
            "recent_frame_indices": recent_frame_indices,
            "older_frame_indices": older_frame_indices,
            "mid_frame_indices": mid_frame_indices,
            "action_entry_indices": action_entry_indices,
        }

    def _load_rgb_frames(
        self,
        video_path: Path,
        video_frame_indices: np.ndarray,
        recent_indices: Sequence[int] | np.ndarray,
        older_indices: Sequence[int] | np.ndarray,
        mid_indices: Sequence[int] | np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        recent_idx = np.asarray(recent_indices, dtype=np.int64)
        older_idx = np.asarray(older_indices, dtype=np.int64)
        mid_idx = np.asarray(mid_indices, dtype=np.int64)
        decoded = decode_rgb_frame_union(
            video_path,
            sorted(
                {
                    max(0, int(video_frame_indices[source_index]) - 1)
                    for source_index in np.concatenate((recent_idx, older_idx, mid_idx))
                }
            ),
        )

        def _stack(indices: np.ndarray) -> torch.Tensor:
            return torch.stack(
                [
                    decoded[max(0, int(video_frame_indices[source_index]) - 1)]
                    for source_index in indices
                ],
                dim=0,
            )

        return _stack(recent_idx), _stack(older_idx), _stack(mid_idx)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        timeline = self._load_clip_timeline(sample.metadata_file)
        selection = self._resolve_sample_indices(sample, timeline)
        recent_frame_indices = np.asarray(selection["recent_frame_indices"], dtype=np.int64)
        older_frame_indices = np.asarray(selection["older_frame_indices"], dtype=np.int64)
        mid_frame_indices = np.asarray(selection["mid_frame_indices"], dtype=np.int64)
        action_entry_indices = np.asarray(selection["action_entry_indices"], dtype=np.int64)
        target_frame_index = int(selection["target_frame_index"])
        frame_timestamps_ns = timeline.frame_timestamps_ns
        model_period_s = 1.0 / max(1.0, float(self.model_hz))
        mid_period_s = 1.0 / max(1.0, float(self.mid_hz))
        rgb_recent, rgb_older, rgb_mid = self._load_rgb_frames(
            timeline.video_path,
            timeline.video_frame_indices,
            recent_frame_indices,
            older_frame_indices,
            mid_frame_indices,
        )

        batch = {
            "clip_id": sample.clip_id,
            "metadata_path": sample.metadata_path,
            "actions_path": sample.actions_path,
            "video_path": sample.video_path,
            "privileged_dir": sample.privileged_dir,
            "anchor_timestamp_ns": sample.anchor_timestamp_ns,
            "target_frame_index": target_frame_index,
            "frame_source": sample.frame_source,
            "action_source": sample.action_source,
            "rgb_recent": rgb_recent,
            "dt_recent": _selected_dt_tensor(
                frame_timestamps_ns,
                recent_frame_indices,
                default_dt_s=model_period_s,
                prev_index=int(older_frame_indices[-1]) if older_frame_indices.size > 0 else None,
            ),
            "rgb_older": rgb_older,
            "dt_older": _selected_dt_tensor(
                frame_timestamps_ns,
                older_frame_indices,
                default_dt_s=model_period_s,
                prev_index=int(mid_frame_indices[-1]) if mid_frame_indices.size > 0 else None,
            ),
            "rgb_mid": rgb_mid,
            "dt_mid": _selected_dt_tensor(
                frame_timestamps_ns,
                mid_frame_indices,
                default_dt_s=mid_period_s,
            ),
            "route_polyline": None,
            "nav_cmd": None,
            "reasoner_tok": None,
        }
        batch["actions_hist"] = _stream_action_vectors(
            timeline.action_vectors,
            action_entry_indices,
        )
        batch["dt_hist"] = _selected_dt_tensor(
            timeline.action_timestamps_ns,
            action_entry_indices,
            default_dt_s=model_period_s,
        )
        return batch
