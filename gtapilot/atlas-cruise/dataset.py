from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from gtapilot.atlas.data.video_decode import decode_rgb_frame_union

from .config import AtlasCruiseConfig
from .preprocess import preprocess_frame_sequence


REQUIRED_TARGET_KEYS: tuple[str, ...] = (
    "target_traj",
    "target_traj_valid",
    "target_ego",
    "target_slow_or_brake",
    "target_fallback",
)


@dataclass(slots=True)
class AtlasCruiseSample:
    rgb_recent: torch.Tensor
    actions_hist: torch.Tensor
    dt_hist: torch.Tensor
    ui_mask_recent: torch.Tensor
    frame_freshness: torch.Tensor
    targets: dict[str, torch.Tensor]
    metadata: dict[str, Any]


@dataclass(slots=True)
class _ClipTimeline:
    clip_id: str
    metadata_path: Path
    actions_path: Path
    video_path: Path
    nominal_fps: float
    frame_timestamps_ns: np.ndarray
    frame_ids: np.ndarray
    video_frame_indices: np.ndarray
    frame_is_repeat: np.ndarray
    action_timestamps_ns: np.ndarray
    action_vectors: np.ndarray


@dataclass(slots=True)
class _SampleIndex:
    metadata_path: Path
    anchor_frame_index: int
    visual_frame_indices: np.ndarray
    visual_desired_timestamps_ns: np.ndarray
    action_indices: np.ndarray
    action_desired_timestamps_ns: np.ndarray


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_metadata_paths(
    recordings_root: Path,
    *,
    metadata_paths: list[str | Path] | None = None,
    split_file: str | Path | None = None,
) -> list[Path]:
    def resolve(raw: str | Path, *, split_parent: Path | None = None) -> Path:
        path = Path(raw)
        if path.is_absolute():
            return path.resolve()
        candidates = []
        if split_parent is not None:
            candidates.append(split_parent / path)
        candidates.extend([recordings_root / path, path])
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        return candidates[0].resolve()

    if metadata_paths is not None:
        return [resolve(path) for path in metadata_paths]
    if split_file is not None:
        split_path = Path(split_file).resolve()
        return [
            resolve(line.strip(), split_parent=split_path.parent)
            for line in split_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    return sorted(recordings_root.glob("capture_*/metadata.json"))


def _select_latest_indices(source_timestamps_ns: Sequence[int] | np.ndarray, desired_timestamps_ns: Sequence[int] | np.ndarray) -> np.ndarray | None:
    source = np.asarray(source_timestamps_ns, dtype=np.int64)
    desired = np.asarray(desired_timestamps_ns, dtype=np.int64)
    indices = np.searchsorted(source, desired, side="right") - 1
    if np.any(indices < 0):
        return None
    return indices.astype(np.int64, copy=False)


def _action_vector_from_payload(payload: dict[str, Any]) -> list[float]:
    return [
        float(payload.get("steer", 0.0)),
        float(payload.get("throttle", 0.0)),
        float(payload.get("brake", 0.0)),
        float(payload.get("handbrake", 0.0)),
        float(payload.get("reverse", 0.0)),
        float(payload.get("pilot_active", 0.0)),
    ]


def _load_timeline(metadata_path: Path) -> _ClipTimeline:
    manifest = _read_json(metadata_path)
    if int(manifest.get("schema_version", 0)) != 3 or str(manifest.get("kind")) != "video_metadata":
        raise ValueError(f"Unsupported blackbox metadata manifest: {metadata_path}")
    clip_id = str(manifest.get("clip_id", metadata_path.parent.name))
    video_session = dict(manifest.get("video_session", {}))
    video_file = str(video_session.get("file_name", "video.mkv"))
    video_path = (metadata_path.parent / video_file).resolve()
    actions_path = metadata_path.parent / str(manifest.get("actions_file", "actions.json"))
    actions_manifest = _read_json(actions_path)
    if int(actions_manifest.get("schema_version", 0)) != 3 or str(actions_manifest.get("kind")) != "actions":
        raise ValueError(f"Unsupported blackbox actions manifest: {actions_path}")
    frames = list(manifest.get("frames", []))
    frame_timestamps = np.asarray([int(row["capture_timestamp_ns"]) for row in frames], dtype=np.int64)
    frame_ids = np.asarray([int(row.get("frame_id", -1)) for row in frames], dtype=np.int64)
    video_frame_indices = np.asarray([int(row["video_frame_index"]) for row in frames], dtype=np.int64)
    frame_is_repeat = np.asarray([bool(row.get("is_repeat", False)) for row in frames], dtype=bool)
    action_timestamps: list[int] = []
    action_vectors: list[list[float]] = []
    for row in actions_manifest.get("actions", []):
        timestamp_ns = int(row.get("message_timestamp_ns", row.get("publish_timestamp_ns", 0)) or 0)
        if timestamp_ns <= 0:
            continue
        action_timestamps.append(timestamp_ns)
        action_vectors.append(_action_vector_from_payload(dict(row.get("payload") or {})))
    if not action_timestamps:
        raise ValueError(f"Actions manifest has no usable raw actions: {actions_path}")
    action_ts = np.asarray(action_timestamps, dtype=np.int64)
    action_vec = np.asarray(action_vectors, dtype=np.float32)
    order = np.argsort(action_ts, kind="stable")
    nominal_fps = float(video_session.get("nominal_fps", (manifest.get("capture_session") or {}).get("nominal_fps", 60.0)) or 60.0)
    return _ClipTimeline(
        clip_id=clip_id,
        metadata_path=metadata_path.resolve(),
        actions_path=actions_path.resolve(),
        video_path=video_path,
        nominal_fps=nominal_fps,
        frame_timestamps_ns=frame_timestamps,
        frame_ids=frame_ids,
        video_frame_indices=video_frame_indices,
        frame_is_repeat=frame_is_repeat,
        action_timestamps_ns=action_ts[order],
        action_vectors=action_vec[order],
    )


def _tensor_from_payload(payload: Any, base_dir: Path) -> torch.Tensor:
    if isinstance(payload, str):
        path = (base_dir / payload).resolve()
        if path.suffix == ".pt":
            value = torch.load(path, map_location="cpu")
            if isinstance(value, torch.Tensor):
                return value
            raise ValueError(f"Expected tensor in {path}.")
        if path.suffix == ".npy":
            return torch.from_numpy(np.load(path))
        raise ValueError(f"Unsupported tensor path suffix {path.suffix!r}.")
    return torch.as_tensor(payload)


def _selected_dt_tensor(source_timestamps_ns: np.ndarray, indices: np.ndarray, desired_timestamps_ns: np.ndarray) -> torch.Tensor:
    values: list[float] = []
    for offset, source_index in enumerate(indices.tolist()):
        selected = int(source_timestamps_ns[source_index])
        if offset == 0:
            if indices.shape[0] > 1:
                default_dt = max(0.0, float(desired_timestamps_ns[1] - desired_timestamps_ns[0]) / 1_000_000_000.0)
            else:
                default_dt = 0.0
            values.append(default_dt)
        else:
            values.append(max(0.0, float(selected - int(source_timestamps_ns[int(indices[offset - 1])])) / 1_000_000_000.0))
    return torch.tensor(values, dtype=torch.float32).unsqueeze(-1)


def validate_sample_shapes(sample: AtlasCruiseSample, cfg: AtlasCruiseConfig) -> None:
    if sample.rgb_recent.shape != (cfg.num_visual_frames, 3, cfg.input_h, cfg.input_w):
        raise ValueError("rgb_recent has the wrong shape.")
    if sample.ui_mask_recent.shape != (cfg.num_visual_frames, 1, cfg.input_h, cfg.input_w):
        raise ValueError("ui_mask_recent has the wrong shape.")
    if sample.frame_freshness.shape != (cfg.num_visual_frames, 2):
        raise ValueError("frame_freshness has the wrong shape.")
    if sample.actions_hist.shape != (cfg.action_history_steps, cfg.action_dim):
        raise ValueError("actions_hist has the wrong shape.")
    if sample.dt_hist.shape != (cfg.action_history_steps, 1):
        raise ValueError("dt_hist has the wrong shape.")
    targets = sample.targets
    for key in REQUIRED_TARGET_KEYS:
        if key not in targets:
            raise ValueError(f"Missing required target {key!r}.")
    if targets["target_traj"].shape != (cfg.traj_points, 4):
        raise ValueError("target_traj has the wrong shape.")
    if targets["target_traj_valid"].shape != (cfg.traj_points,):
        raise ValueError("target_traj_valid has the wrong shape.")
    if targets["target_ego"].shape != (4,):
        raise ValueError("target_ego has the wrong shape.")
    if targets["target_slow_or_brake"].shape != (1,):
        raise ValueError("target_slow_or_brake has the wrong shape.")
    if targets["target_fallback"].shape != (1,):
        raise ValueError("target_fallback has the wrong shape.")


class AtlasCruiseJsonlDataset(Dataset[AtlasCruiseSample]):
    def __init__(self, samples_jsonl: str | Path, cfg: AtlasCruiseConfig | None = None, *, validate: bool = True):
        self.cfg = cfg or AtlasCruiseConfig()
        self.cfg.validate()
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

    def __getitem__(self, index: int) -> AtlasCruiseSample:
        row = self.rows[index]
        targets = {
            key: _tensor_from_payload(value, self.base_dir).float()
            for key, value in dict(row.get("targets", {})).items()
        }
        sample = AtlasCruiseSample(
            rgb_recent=_tensor_from_payload(row["rgb_recent"], self.base_dir).float(),
            actions_hist=_tensor_from_payload(row["actions_hist"], self.base_dir).float(),
            dt_hist=_tensor_from_payload(row["dt_hist"], self.base_dir).float(),
            ui_mask_recent=_tensor_from_payload(row.get("ui_mask_recent", torch.zeros(self.cfg.num_visual_frames, 1, self.cfg.input_h, self.cfg.input_w)), self.base_dir).float(),
            frame_freshness=_tensor_from_payload(row.get("frame_freshness", torch.zeros(self.cfg.num_visual_frames, 2)), self.base_dir).float(),
            targets=targets,
            metadata=dict(row.get("metadata", {})),
        )
        if self.validate:
            validate_sample_shapes(sample, self.cfg)
        return sample


class AtlasCruiseBlackboxDataset(Dataset[AtlasCruiseSample]):
    def __init__(
        self,
        recordings_root: str | Path,
        cfg: AtlasCruiseConfig | None = None,
        *,
        metadata_paths: list[str | Path] | None = None,
        split_file: str | Path | None = None,
        anchor_stride_frames: int = 3,
        max_samples_per_clip: int | None = None,
        validate: bool = True,
    ):
        self.recordings_root = Path(recordings_root)
        self.cfg = cfg or AtlasCruiseConfig()
        self.cfg.validate()
        self.metadata_paths = _resolve_metadata_paths(
            self.recordings_root,
            metadata_paths=metadata_paths,
            split_file=split_file,
        )
        self.anchor_stride_frames = max(1, int(anchor_stride_frames))
        self.max_samples_per_clip = None if max_samples_per_clip is None else max(1, int(max_samples_per_clip))
        self.validate = validate
        self._timeline_cache: dict[Path, _ClipTimeline] = {}
        self._target_cache: dict[Path, dict[str, np.ndarray]] = {}
        self.samples = self._build_indices()

    def __len__(self) -> int:
        return len(self.samples)

    def _load_timeline(self, metadata_path: Path) -> _ClipTimeline:
        metadata_path = metadata_path.resolve()
        timeline = self._timeline_cache.get(metadata_path)
        if timeline is None:
            timeline = _load_timeline(metadata_path)
            self._timeline_cache[metadata_path] = timeline
        return timeline

    def _target_package_path(self, metadata_path: Path) -> Path:
        return metadata_path.parent / "privileged" / self.cfg.target_package_name

    def _load_targets(self, metadata_path: Path) -> dict[str, np.ndarray]:
        package_path = self._target_package_path(metadata_path).resolve()
        cached = self._target_cache.get(package_path)
        if cached is not None:
            return cached
        if not package_path.exists():
            raise FileNotFoundError(
                f"Missing Atlas-Cruise target package: {package_path}. "
                "Build it with python -m gtapilot.atlas-cruise.tools.build_targets."
            )
        payload = np.load(package_path)
        targets = {key: payload[key] for key in payload.files}
        timeline = self._load_timeline(metadata_path)
        if "anchor_timestamps_ns" not in targets:
            raise ValueError(f"Target package missing anchor_timestamps_ns: {package_path}")
        if not np.array_equal(np.asarray(targets["anchor_timestamps_ns"], dtype=np.int64), timeline.frame_timestamps_ns):
            raise ValueError(f"Cruise target timestamps do not align with clip frames: {package_path}")
        self._target_cache[package_path] = targets
        return targets

    def _build_indices(self) -> list[_SampleIndex]:
        samples: list[_SampleIndex] = []
        offsets = np.asarray(self.cfg.resolved_visual_offsets_s(), dtype=np.float64)
        action_steps = self.cfg.action_history_steps
        action_step_ns = int(round(1_000_000_000.0 / self.cfg.action_sample_hz))
        for metadata_path in self.metadata_paths:
            timeline = self._load_timeline(metadata_path)
            targets = self._load_targets(metadata_path)
            clip_samples: list[_SampleIndex] = []
            for anchor_idx in range(0, timeline.frame_timestamps_ns.shape[0], self.anchor_stride_frames):
                anchor_ns = int(timeline.frame_timestamps_ns[anchor_idx])
                if int(np.asarray(targets["target_traj_valid"])[anchor_idx].sum()) < self.cfg.min_valid_traj_points:
                    continue
                visual_desired = np.asarray(
                    [anchor_ns + int(round(offset_s * 1_000_000_000.0)) for offset_s in offsets],
                    dtype=np.int64,
                )
                visual_indices = _select_latest_indices(timeline.frame_timestamps_ns, visual_desired)
                if visual_indices is None:
                    continue
                frame_ages_s = (visual_desired - timeline.frame_timestamps_ns[visual_indices]).astype(np.float64) / 1_000_000_000.0
                if np.any(frame_ages_s < -1e-9) or np.any(frame_ages_s > self.cfg.max_frame_staleness_s):
                    continue
                action_desired = np.asarray(
                    [anchor_ns - (action_steps - 1 - idx) * action_step_ns for idx in range(action_steps)],
                    dtype=np.int64,
                )
                action_indices = _select_latest_indices(timeline.action_timestamps_ns, action_desired)
                if action_indices is None:
                    continue
                action_ages_s = (action_desired - timeline.action_timestamps_ns[action_indices]).astype(np.float64) / 1_000_000_000.0
                if np.any(action_ages_s < -1e-9) or np.any(action_ages_s > self.cfg.max_action_staleness_s):
                    continue
                clip_samples.append(
                    _SampleIndex(
                        metadata_path=metadata_path.resolve(),
                        anchor_frame_index=anchor_idx,
                        visual_frame_indices=visual_indices,
                        visual_desired_timestamps_ns=visual_desired,
                        action_indices=action_indices,
                        action_desired_timestamps_ns=action_desired,
                    )
                )
            if self.max_samples_per_clip is not None and len(clip_samples) > self.max_samples_per_clip:
                select = np.linspace(0, len(clip_samples) - 1, self.max_samples_per_clip, dtype=np.int64)
                clip_samples = [clip_samples[int(idx)] for idx in select.tolist()]
            samples.extend(clip_samples)
        return samples

    def __getitem__(self, index: int) -> AtlasCruiseSample:
        sample_index = self.samples[index]
        timeline = self._load_timeline(sample_index.metadata_path)
        targets_np = self._load_targets(sample_index.metadata_path)
        video_indices = [
            max(0, int(timeline.video_frame_indices[source_idx]) - 1)
            for source_idx in sample_index.visual_frame_indices.tolist()
        ]
        decoded = decode_rgb_frame_union(timeline.video_path, sorted(set(video_indices)))
        frames = [decoded[int(video_idx)] for video_idx in video_indices]
        rgb_recent, ui_mask_recent, ui_states = preprocess_frame_sequence(frames, self.cfg)
        action_indices = sample_index.action_indices
        actions_hist = torch.from_numpy(timeline.action_vectors[action_indices]).float()
        dt_hist = _selected_dt_tensor(
            timeline.action_timestamps_ns,
            action_indices,
            sample_index.action_desired_timestamps_ns,
        )
        selected_frame_ts = timeline.frame_timestamps_ns[sample_index.visual_frame_indices]
        frame_age_s = (sample_index.visual_desired_timestamps_ns - selected_frame_ts).astype(np.float32) / 1_000_000_000.0
        frame_repeat = timeline.frame_is_repeat[sample_index.visual_frame_indices].astype(np.float32)
        frame_freshness = torch.from_numpy(np.stack([frame_age_s, frame_repeat], axis=-1)).float()
        anchor_idx = int(sample_index.anchor_frame_index)
        targets = {
            "target_traj": torch.from_numpy(np.asarray(targets_np["target_traj"][anchor_idx])).float(),
            "target_traj_valid": torch.from_numpy(np.asarray(targets_np["target_traj_valid"][anchor_idx])).bool(),
            "target_ego": torch.from_numpy(np.asarray(targets_np["target_ego"][anchor_idx])).float(),
            "target_slow_or_brake": torch.from_numpy(np.asarray(targets_np["target_slow_or_brake"][anchor_idx])).float(),
            "target_fallback": torch.from_numpy(np.asarray(targets_np["target_fallback"][anchor_idx])).float(),
        }
        if "target_control_aux" in targets_np:
            targets["target_control_aux"] = torch.from_numpy(np.asarray(targets_np["target_control_aux"][anchor_idx])).float()
        sample = AtlasCruiseSample(
            rgb_recent=rgb_recent,
            actions_hist=actions_hist,
            dt_hist=dt_hist,
            ui_mask_recent=ui_mask_recent,
            frame_freshness=frame_freshness,
            targets=targets,
            metadata={
                "clip_id": timeline.clip_id,
                "metadata_path": str(timeline.metadata_path),
                "video_path": str(timeline.video_path),
                "anchor_frame_index": anchor_idx,
                "anchor_timestamp_ns": int(timeline.frame_timestamps_ns[anchor_idx]),
                "ui_states": [asdict(state) for state in ui_states],
            },
        )
        if self.validate:
            validate_sample_shapes(sample, self.cfg)
        return sample


def collate_atlas_cruise_samples(samples: list[AtlasCruiseSample]) -> dict[str, Any]:
    if not samples:
        raise ValueError("Cannot collate an empty Atlas-Cruise batch.")
    target_keys = set.intersection(*(set(sample.targets) for sample in samples))
    return {
        "rgb_recent": torch.stack([sample.rgb_recent for sample in samples], dim=0),
        "actions_hist": torch.stack([sample.actions_hist for sample in samples], dim=0),
        "dt_hist": torch.stack([sample.dt_hist for sample in samples], dim=0),
        "ui_mask_recent": torch.stack([sample.ui_mask_recent for sample in samples], dim=0),
        "frame_freshness": torch.stack([sample.frame_freshness for sample in samples], dim=0),
        "targets": {
            key: torch.stack([sample.targets[key] for sample in samples], dim=0)
            for key in sorted(target_keys)
        },
        "metadata": [sample.metadata for sample in samples],
    }
