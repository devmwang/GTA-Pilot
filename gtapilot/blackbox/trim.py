from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gtapilot.blackbox.constants import (
    DEFAULT_OUTPUT_DIR,
    INTEGRITY_EDGE_GRACE_SECONDS,
    VIDEO_CODEC,
    VIDEO_CONTAINER,
    VIDEO_CRF,
    VIDEO_PIXEL_FORMAT,
    VIDEO_PRESET,
)
from gtapilot.blackbox.manifest_utils import (
    _active_devices_seen,
    _native_capture_timing_summary,
    _percentile_ns,
    _session_data_window_ns,
    _session_integrity_payload,
    _split_drop_events_for_integrity,
)

DEFAULT_ACTION_SOURCE = "manual_input"


@dataclass(frozen=True)
class ClipPaths:
    clip_name: str
    recordings_root: Path
    metadata_path: Path
    video_path: Path
    privileged_dir: Path | None


@dataclass(frozen=True)
class TrimSelection:
    keep_start_ns: int
    keep_end_ns: int
    start_zero_based: int
    end_exclusive: int
    kept_frame_count: int
    original_frame_count: int
    original_action_count: int


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _recordings_root() -> Path:
    return (_repo_root() / DEFAULT_OUTPUT_DIR).resolve()


def _parse_non_negative_seconds(raw_value: str, label: str) -> float:
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{label} must be a non-negative number of seconds.") from exc
    if value < 0.0:
        raise ValueError(f"{label} must be non-negative.")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> bytes:
    text = json.dumps(payload, indent=2)
    path.write_text(text, encoding="utf-8")
    return text.encode("utf-8")


def _resolve_clip_paths(clip_name: str) -> ClipPaths:
    recordings_root = _recordings_root()
    metadata_path = recordings_root / f"{clip_name}_metadata.json"
    video_path = recordings_root / f"{clip_name}_video.mkv"
    privileged_dir = recordings_root / f"{clip_name}_privileged"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if privileged_dir.exists():
        if not privileged_dir.is_dir():
            raise RuntimeError(f"Privileged sibling is not a directory: {privileged_dir}")
        if not (privileged_dir / "manifest.json").exists():
            raise RuntimeError(
                f"Privileged sibling exists but manifest.json is missing: {privileged_dir}"
            )
        resolved_privileged_dir: Path | None = privileged_dir
    else:
        resolved_privileged_dir = None

    return ClipPaths(
        clip_name=clip_name,
        recordings_root=recordings_root,
        metadata_path=metadata_path.resolve(),
        video_path=video_path.resolve(),
        privileged_dir=None if resolved_privileged_dir is None else resolved_privileged_dir.resolve(),
    )


def _schema_version(payload: dict[str, Any]) -> int:
    return int(payload.get("schema_version", 0))


def _clip_selection(
    manifest: dict[str, Any],
    *,
    trim_start_seconds: float,
    trim_end_seconds: float,
) -> tuple[list[dict[str, Any]], TrimSelection]:
    if _schema_version(manifest) != 2:
        raise ValueError("Unsupported blackbox manifest schema for trim.")

    if trim_start_seconds == 0.0 and trim_end_seconds == 0.0:
        raise ValueError("Zero-trim is not allowed. At least one trim value must be > 0.")

    session = dict(manifest.get("session", {}))
    frames = list(manifest.get("frames", []))
    actions = list(manifest.get("actions", []))
    if not frames:
        raise ValueError("Clip has no frames to trim.")

    session_start_ns = int(session.get("timeline_start_timestamp_ns", 0) or 0)
    session_end_ns = int(session.get("timeline_end_timestamp_ns", 0) or 0)
    keep_start_ns = session_start_ns + int(round(trim_start_seconds * 1_000_000_000.0))
    keep_end_ns = session_end_ns - int(round(trim_end_seconds * 1_000_000_000.0))
    if keep_end_ns < keep_start_ns:
        raise ValueError("Requested trim window removes the entire clip.")

    kept_frames = [
        dict(frame)
        for frame in frames
        if keep_start_ns <= int(frame.get("capture_timestamp_ns", 0) or 0) <= keep_end_ns
    ]
    if not kept_frames:
        raise ValueError("Requested trim window keeps zero frames.")

    original_indices = [int(frame.get("video_frame_index", 0)) for frame in kept_frames]
    first_index = original_indices[0]
    last_index = original_indices[-1]
    expected_count = last_index - first_index + 1
    if expected_count != len(kept_frames):
        raise RuntimeError("Kept frames do not form a contiguous video frame range.")
    if any(index != first_index + offset for offset, index in enumerate(original_indices)):
        raise RuntimeError("Kept frames are not contiguous by video_frame_index.")

    return kept_frames, TrimSelection(
        keep_start_ns=keep_start_ns,
        keep_end_ns=keep_end_ns,
        start_zero_based=first_index - 1,
        end_exclusive=last_index,
        kept_frame_count=len(kept_frames),
        original_frame_count=len(frames),
        original_action_count=len(actions),
    )


def _filter_actions(
    actions: list[dict[str, Any]],
    *,
    keep_start_ns: int,
    keep_end_ns: int,
) -> list[dict[str, Any]]:
    return [
        dict(action)
        for action in actions
        if keep_start_ns <= int(action.get("message_timestamp_ns", 0) or 0) <= keep_end_ns
    ]


def _filter_timeline_records(
    records: list[dict[str, Any]],
    *,
    keep_start_ns: int,
    keep_end_ns: int,
    timestamp_key: str,
) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for record in records:
        timestamp_ns = int(record.get(timestamp_key, 0) or 0)
        if timestamp_ns <= 0 or keep_start_ns <= timestamp_ns <= keep_end_ns:
            kept.append(dict(record))
    return kept


def _renumber_video_frame_indices(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    renumbered: list[dict[str, Any]] = []
    for index, frame in enumerate(frames, start=1):
        updated = dict(frame)
        updated["video_frame_index"] = int(index)
        renumbered.append(updated)
    return renumbered


def _writer_lag_summary_ns(frames: list[dict[str, Any]]) -> dict[str, int]:
    writer_lag_ns = [
        max(
            0,
            int(frame.get("writer_committed_timestamp_ns", 0) or 0)
            - int(frame.get("subscriber_received_timestamp_ns", 0) or 0),
        )
        for frame in frames
        if frame.get("writer_committed_timestamp_ns") is not None
        and frame.get("subscriber_received_timestamp_ns") is not None
    ]
    if not writer_lag_ns:
        return {"p50": 0, "p95": 0, "max": 0}
    return {
        "p50": _percentile_ns(writer_lag_ns, 50.0),
        "p95": _percentile_ns(writer_lag_ns, 95.0),
        "max": max(writer_lag_ns),
    }


def _sequence_gap_summary(sequence_ids: list[int]) -> tuple[int, int]:
    gap_count = 0
    missing_total = 0
    last_sequence_id: int | None = None
    for sequence_id in sequence_ids:
        if last_sequence_id is not None and sequence_id > last_sequence_id + 1:
            gap_count += 1
            missing_total += sequence_id - last_sequence_id - 1
        last_sequence_id = sequence_id
    return int(gap_count), int(missing_total)


def _overflow_metrics(
    drop_events: list[dict[str, Any]],
    *,
    channel: str,
) -> tuple[int, int, int]:
    overflow_count = 0
    dropped_messages = 0
    max_buffer_occupancy = 0
    for event in drop_events:
        if str(event.get("kind", "")) != "local_overflow":
            continue
        if str(event.get("channel", "")) != channel:
            continue
        overflow_count += 1
        dropped_messages += int(event.get("dropped_count", 0) or 0)
        buffer_occupancy = event.get("buffer_occupancy")
        if buffer_occupancy is not None:
            max_buffer_occupancy = max(max_buffer_occupancy, int(buffer_occupancy))
    return int(overflow_count), int(dropped_messages), int(max_buffer_occupancy)


def _source_name_from_stats(
    transport_stats: dict[str, Any],
    *,
    fallback: str,
) -> str:
    last_sequence_by_source = dict(transport_stats.get("last_sequence_by_source", {}))
    if last_sequence_by_source:
        return str(next(iter(last_sequence_by_source.keys())))
    return str(fallback)


def _recomputed_transport_stats(
    manifest: dict[str, Any],
    *,
    frames: list[dict[str, Any]],
    actions: list[dict[str, Any]],
    raw_drop_events: list[dict[str, Any]],
) -> dict[str, Any]:
    source_transport = dict(manifest.get("transport_stats", {}))
    vision_source = _source_name_from_stats(
        dict(source_transport.get("vision", {})),
        fallback=str((manifest.get("capture_session") or {}).get("capture_source", "unknown")),
    )
    action_source = _source_name_from_stats(
        dict(source_transport.get("actions", {})),
        fallback=DEFAULT_ACTION_SOURCE,
    )

    frame_sequence_ids = [int(frame.get("sequence_id", 0)) for frame in frames]
    action_sequence_ids = [int(action.get("sequence_id", 0)) for action in actions]
    vision_gap_count, vision_missing = _sequence_gap_summary(frame_sequence_ids)
    action_gap_count, action_missing = _sequence_gap_summary(action_sequence_ids)
    vision_overflow_count, vision_overflow_dropped, vision_max_buffer = _overflow_metrics(
        raw_drop_events,
        channel="vision.frames",
    )
    action_overflow_count, action_overflow_dropped, action_max_buffer = _overflow_metrics(
        raw_drop_events,
        channel="input.actions",
    )

    return {
        "vision": {
            "channel": "vision.frames",
            "messages_received": int(len(frames)),
            "sequence_gap_count": int(vision_gap_count),
            "missing_message_count": int(vision_missing),
            "local_overflow_count": int(vision_overflow_count),
            "local_overflow_dropped_messages": int(vision_overflow_dropped),
            "max_buffer_occupancy": int(vision_max_buffer),
            "last_sequence_by_source": (
                {}
                if not frame_sequence_ids
                else {vision_source: int(frame_sequence_ids[-1])}
            ),
        },
        "actions": {
            "channel": "input.actions",
            "messages_received": int(len(actions)),
            "sequence_gap_count": int(action_gap_count),
            "missing_message_count": int(action_missing),
            "local_overflow_count": int(action_overflow_count),
            "local_overflow_dropped_messages": int(action_overflow_dropped),
            "max_buffer_occupancy": int(action_max_buffer),
            "last_sequence_by_source": (
                {}
                if not action_sequence_ids
                else {action_source: int(action_sequence_ids[-1])}
            ),
        },
    }


def _recomputed_writer_stats(
    source_writer_stats: dict[str, Any],
    *,
    frames: list[dict[str, Any]],
    actions: list[dict[str, Any]],
) -> dict[str, Any]:
    writer_stats = dict(source_writer_stats)
    writer_stats["frame_queue_capacity"] = int(writer_stats.get("frame_queue_capacity", 32))
    writer_stats["action_queue_capacity"] = int(writer_stats.get("action_queue_capacity", 4096))
    writer_stats["frames_written"] = int(len(frames))
    writer_stats["actions_written"] = int(len(actions))
    writer_stats["writer_lag_ns"] = _writer_lag_summary_ns(frames)
    return writer_stats


def _filter_raw_drop_events(
    manifest: dict[str, Any],
    *,
    keep_start_ns: int,
    keep_end_ns: int,
) -> list[dict[str, Any]]:
    combined = list(manifest.get("drop_events", [])) + list(
        manifest.get("ignored_drop_events", [])
    )
    return _filter_timeline_records(
        [dict(event) for event in combined],
        keep_start_ns=keep_start_ns,
        keep_end_ns=keep_end_ns,
        timestamp_key="timestamp_ns",
    )


def _trimmed_manifest(
    manifest: dict[str, Any],
    *,
    kept_frames: list[dict[str, Any]],
    kept_actions: list[dict[str, Any]],
    kept_pipeline_samples: list[dict[str, Any]],
    raw_drop_events: list[dict[str, Any]],
    kept_session_events: list[dict[str, Any]],
    finalized_timestamp_ns: int,
) -> dict[str, Any]:
    source_session = dict(manifest.get("session", {}))
    source_capture_session = dict(manifest.get("capture_session", {}))
    source_video_session = dict(manifest.get("video_session", {}))
    source_input_session = dict(manifest.get("input_session", {}))
    source_writer_stats = dict(manifest.get("writer_stats", {}))
    source_integrity = dict(manifest.get("session_integrity", {}))

    session_start_timestamp_ns, session_end_timestamp_ns = _session_data_window_ns(
        frame_payloads=kept_frames,
        action_payloads=kept_actions,
        created_timestamp_ns=int(source_session.get("created_timestamp_ns", 0) or 0),
    )
    edge_grace_window_seconds = float(
        source_integrity.get("edge_grace_window_seconds", INTEGRITY_EDGE_GRACE_SECONDS)
    )
    edge_grace_window_ns = int(round(edge_grace_window_seconds * 1_000_000_000.0))
    drop_events, ignored_drop_events = _split_drop_events_for_integrity(
        drop_events=raw_drop_events,
        session_start_timestamp_ns=session_start_timestamp_ns,
        session_end_timestamp_ns=session_end_timestamp_ns,
        grace_window_ns=edge_grace_window_ns,
    )

    session = dict(source_session)
    session["finalized_timestamp_ns"] = int(finalized_timestamp_ns)
    session["timeline_start_timestamp_ns"] = int(session_start_timestamp_ns)
    session["timeline_end_timestamp_ns"] = int(session_end_timestamp_ns)
    session["frame_count"] = int(len(kept_frames))
    session["action_count"] = int(len(kept_actions))

    input_session = dict(source_input_session)
    input_session["active_devices_seen"] = _active_devices_seen(kept_actions)

    return {
        "schema_version": 2,
        "session": session,
        "capture_session": dict(source_capture_session),
        "video_session": dict(source_video_session),
        "input_session": input_session,
        "session_integrity": _session_integrity_payload(
            frame_payloads=kept_frames,
            drop_events=drop_events,
            ignored_drop_events=ignored_drop_events,
            edge_grace_window_seconds=edge_grace_window_seconds,
        ),
        "transport_stats": _recomputed_transport_stats(
            manifest,
            frames=kept_frames,
            actions=kept_actions,
            raw_drop_events=raw_drop_events,
        ),
        "writer_stats": _recomputed_writer_stats(
            source_writer_stats,
            frames=kept_frames,
            actions=kept_actions,
        ),
        "native_capture_stats": _native_capture_timing_summary(kept_pipeline_samples),
        "native_pipeline_samples": kept_pipeline_samples,
        "drop_events": drop_events,
        "ignored_drop_events": ignored_drop_events,
        "session_events": kept_session_events,
        "frames": kept_frames,
        "actions": kept_actions,
    }


def _ensure_video_manifest_match(paths: ClipPaths, manifest: dict[str, Any]) -> None:
    video_session = dict(manifest.get("video_session", {}))
    manifest_video_name = str(video_session.get("file_name", "") or "")
    if not manifest_video_name:
        raise RuntimeError("Manifest is missing video_session.file_name.")
    if manifest_video_name != paths.video_path.name:
        raise RuntimeError(
            "Manifest video_session.file_name does not match the canonical clip video path."
        )


def _ffmpeg_binary(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f"{name} was not found on PATH.")
    return path


def _video_frame_count(video_path: Path) -> int:
    ffprobe_path = _ffmpeg_binary("ffprobe")
    result = subprocess.run(
        [
            ffprobe_path,
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames,nb_frames",
            "-of",
            "json",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout or "{}")
    streams = list(payload.get("streams", []))
    if not streams:
        raise RuntimeError(f"ffprobe did not return a video stream for {video_path}.")
    stream = dict(streams[0])
    for key in ("nb_read_frames", "nb_frames"):
        value = stream.get(key)
        if value not in (None, "N/A", ""):
            return int(value)
    raise RuntimeError(f"Unable to determine video frame count for {video_path}.")


def _trim_video(
    *,
    input_path: Path,
    output_path: Path,
    start_zero_based: int,
    end_exclusive: int,
) -> None:
    ffmpeg_path = _ffmpeg_binary("ffmpeg")
    filter_expr = (
        f"trim=start_frame={int(start_zero_based)}:end_frame={int(end_exclusive)},"
        "setpts=PTS-STARTPTS"
    )
    subprocess.run(
        [
            ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(input_path),
            "-an",
            "-vf",
            filter_expr,
            "-c:v",
            VIDEO_CODEC,
            "-preset",
            VIDEO_PRESET,
            "-crf",
            str(VIDEO_CRF),
            "-pix_fmt",
            VIDEO_PIXEL_FORMAT,
            "-f",
            VIDEO_CONTAINER,
            str(output_path),
        ],
        check=True,
    )


def _metadata_sha1(metadata_bytes: bytes) -> str:
    return hashlib.sha1(metadata_bytes).hexdigest()


def _trim_privileged_package(
    *,
    privileged_dir: Path,
    output_dir: Path,
    source_metadata_path: Path,
    source_video_path: Path,
    trimmed_metadata_path: Path,
    trimmed_video_path: Path,
    trimmed_metadata_sha1: str,
    start_zero_based: int,
    end_exclusive: int,
    expected_frame_count: int,
) -> None:
    manifest_path = privileged_dir / "manifest.json"
    manifest = _load_json(manifest_path)
    if int(manifest.get("schema_version", 0)) != 2:
        raise ValueError(f"Unsupported privileged manifest schema: {manifest_path}")
    frame_count = int(manifest.get("frame_count", 0))
    if frame_count <= 0:
        raise ValueError(f"Invalid privileged frame_count in {manifest_path}")
    if frame_count < end_exclusive:
        raise ValueError("Privileged frame_count is smaller than the kept source frame window.")

    if manifest.get("source_metadata_file") is not None:
        if Path(str(manifest["source_metadata_file"])).resolve() != source_metadata_path.resolve():
            raise ValueError("Privileged manifest source_metadata_file does not match the clip metadata.")
    if manifest.get("source_video_file") is not None:
        if Path(str(manifest["source_video_file"])).resolve() != source_video_path.resolve():
            raise ValueError("Privileged manifest source_video_file does not match the clip video.")
    if manifest.get("source_metadata_sha1") is not None:
        if _metadata_sha1(source_metadata_path.read_bytes()) != str(manifest["source_metadata_sha1"]):
            raise ValueError("Privileged manifest source_metadata_sha1 does not match the clip metadata.")

    output_dir.mkdir(parents=True, exist_ok=True)
    frame_slice = slice(int(start_zero_based), int(end_exclusive))
    kept_files: dict[str, str] = {}

    manifest_files = dict(manifest.get("files", {}))
    for key, rel_path in manifest_files.items():
        source_path = privileged_dir / rel_path
        if not source_path.exists():
            raise FileNotFoundError(f"Privileged file missing: {source_path}")
        array = np.load(source_path, mmap_mode="r")
        target_path = output_dir / rel_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if key == "track_lag_indices":
            trimmed_array = np.asarray(array)
        else:
            if array.ndim == 0 or int(array.shape[0]) != frame_count:
                raise RuntimeError(
                    f"Privileged array {key} does not have frame-major leading dimension {frame_count}."
                )
            trimmed_array = np.asarray(array[frame_slice])
        np.save(target_path, trimmed_array)
        kept_files[key] = rel_path

    optional_arrays = {
        "frame_ids_file": "frame_ids.npy",
        "capture_timestamps_ns_file": "capture_timestamps_ns.npy",
        "video_frame_indices_file": "video_frame_indices.npy",
    }
    updated_optional_paths: dict[str, str | None] = {key: None for key in optional_arrays}
    for attr_name, default_name in optional_arrays.items():
        rel_path = manifest.get(attr_name)
        if not rel_path:
            continue
        source_path = privileged_dir / rel_path
        if not source_path.exists():
            raise FileNotFoundError(f"Privileged file missing: {source_path}")
        array = np.load(source_path, mmap_mode="r")
        if int(array.shape[0]) != frame_count:
            raise RuntimeError(
                f"Privileged optional array {attr_name} does not match frame_count {frame_count}."
            )
        trimmed_array = np.asarray(array[frame_slice])
        if attr_name == "video_frame_indices_file":
            trimmed_array = np.arange(1, expected_frame_count + 1, dtype=np.int64)
        rel_output_path = rel_path or default_name
        target_path = output_dir / rel_output_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(target_path, trimmed_array)
        updated_optional_paths[attr_name] = rel_output_path

    trimmed_manifest = {
        "clip_id": str(manifest["clip_id"]),
        "schema_version": 2,
        "source_video_file": str(trimmed_video_path.resolve()),
        "source_metadata_file": str(trimmed_metadata_path.resolve()),
        "source_metadata_sha1": trimmed_metadata_sha1,
        "frame_count": int(expected_frame_count),
        "grid_height_8x": int(manifest.get("grid_height_8x", 136)),
        "grid_width_8x": int(manifest.get("grid_width_8x", 240)),
        "track_lag_indices": [
            int(value) for value in manifest.get("track_lag_indices", [1, 2, 4, 8, 16, 31])
        ],
        "frame_ids_file": updated_optional_paths["frame_ids_file"],
        "capture_timestamps_ns_file": updated_optional_paths["capture_timestamps_ns_file"],
        "video_frame_indices_file": updated_optional_paths["video_frame_indices_file"],
        "files": kept_files,
        "metadata": dict(manifest.get("metadata", {})),
    }
    _write_json(output_dir / "manifest.json", trimmed_manifest)


def _safe_remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _replace_with_backup(
    *,
    originals: list[tuple[Path, Path]],
    replacements: list[tuple[Path, Path]],
) -> None:
    backed_up: list[tuple[Path, Path]] = []
    moved_in: list[tuple[Path, Path]] = []
    try:
        for source_path, backup_path in originals:
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source_path), str(backup_path))
            backed_up.append((source_path, backup_path))

        for source_path, target_path in replacements:
            _safe_remove_path(target_path)
            shutil.move(str(source_path), str(target_path))
            moved_in.append((source_path, target_path))
    except Exception:
        for _source_path, target_path in reversed(moved_in):
            if target_path.exists():
                _safe_remove_path(target_path)
        for original_path, backup_path in reversed(backed_up):
            if backup_path.exists():
                shutil.move(str(backup_path), str(original_path))
        raise


def _backup_dir(recordings_root: Path, clip_name: str) -> Path:
    originals_root = recordings_root / "originals"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup_dir = originals_root / f"{clip_name}_{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=False)
    return backup_dir


def _build_trimmed_assets(
    *,
    paths: ClipPaths,
    manifest: dict[str, Any],
    selection: TrimSelection,
    kept_frames: list[dict[str, Any]],
) -> tuple[Path, Path, Path | None]:
    temp_root = Path(
        tempfile.mkdtemp(prefix=f"{paths.clip_name}_trim_", dir=str(paths.recordings_root))
    )
    trimmed_metadata_path = temp_root / paths.metadata_path.name
    trimmed_video_path = temp_root / paths.video_path.name
    trimmed_privileged_dir = (
        None if paths.privileged_dir is None else temp_root / paths.privileged_dir.name
    )

    kept_actions = _filter_actions(
        list(manifest.get("actions", [])),
        keep_start_ns=selection.keep_start_ns,
        keep_end_ns=selection.keep_end_ns,
    )
    kept_pipeline_samples = _filter_timeline_records(
        list(manifest.get("native_pipeline_samples", [])),
        keep_start_ns=selection.keep_start_ns,
        keep_end_ns=selection.keep_end_ns,
        timestamp_key="capture_timestamp_ns",
    )
    raw_drop_events = _filter_raw_drop_events(
        manifest,
        keep_start_ns=selection.keep_start_ns,
        keep_end_ns=selection.keep_end_ns,
    )
    kept_session_events = _filter_timeline_records(
        list(manifest.get("session_events", [])),
        keep_start_ns=selection.keep_start_ns,
        keep_end_ns=selection.keep_end_ns,
        timestamp_key="timestamp_ns",
    )

    renumbered_frames = _renumber_video_frame_indices(kept_frames)
    trimmed_manifest = _trimmed_manifest(
        manifest,
        kept_frames=renumbered_frames,
        kept_actions=kept_actions,
        kept_pipeline_samples=kept_pipeline_samples,
        raw_drop_events=raw_drop_events,
        kept_session_events=kept_session_events,
        finalized_timestamp_ns=time.time_ns(),
    )
    metadata_bytes = _write_json(trimmed_metadata_path, trimmed_manifest)
    trimmed_metadata_sha1 = _metadata_sha1(metadata_bytes)

    _trim_video(
        input_path=paths.video_path,
        output_path=trimmed_video_path,
        start_zero_based=selection.start_zero_based,
        end_exclusive=selection.end_exclusive,
    )
    trimmed_video_frame_count = _video_frame_count(trimmed_video_path)
    if trimmed_video_frame_count != selection.kept_frame_count:
        raise RuntimeError(
            "Trimmed video frame count does not match trimmed metadata frame count: "
            f"{trimmed_video_frame_count} != {selection.kept_frame_count}"
        )

    if paths.privileged_dir is not None and trimmed_privileged_dir is not None:
        _trim_privileged_package(
            privileged_dir=paths.privileged_dir,
            output_dir=trimmed_privileged_dir,
            source_metadata_path=paths.metadata_path,
            source_video_path=paths.video_path,
            trimmed_metadata_path=trimmed_metadata_path,
            trimmed_video_path=trimmed_video_path,
            trimmed_metadata_sha1=trimmed_metadata_sha1,
            start_zero_based=selection.start_zero_based,
            end_exclusive=selection.end_exclusive,
            expected_frame_count=selection.kept_frame_count,
        )

    return temp_root, trimmed_metadata_path, trimmed_privileged_dir


def _validate_trimmed_assets(
    *,
    trimmed_metadata_path: Path,
    expected_frame_count: int,
) -> None:
    manifest = _load_json(trimmed_metadata_path)
    if _schema_version(manifest) != 2:
        raise RuntimeError("Trimmed metadata did not preserve schema version 2.")
    session = dict(manifest.get("session", {}))
    frames = list(manifest.get("frames", []))
    if int(session.get("frame_count", -1)) != int(expected_frame_count):
        raise RuntimeError("Trimmed session.frame_count is inconsistent.")
    if len(frames) != expected_frame_count:
        raise RuntimeError("Trimmed metadata frame array is inconsistent.")
    for expected_index, frame in enumerate(frames, start=1):
        if int(frame.get("video_frame_index", -1)) != expected_index:
            raise RuntimeError("Trimmed metadata video_frame_index values are not contiguous from 1.")


def _execute_trim(clip_name: str, trim_start_seconds: float, trim_end_seconds: float) -> None:
    paths = _resolve_clip_paths(clip_name)
    manifest = _load_json(paths.metadata_path)
    if _schema_version(manifest) != 2:
        raise ValueError(f"Unsupported blackbox manifest schema: {paths.metadata_path}")
    _ensure_video_manifest_match(paths, manifest)
    kept_frames, selection = _clip_selection(
        manifest,
        trim_start_seconds=trim_start_seconds,
        trim_end_seconds=trim_end_seconds,
    )

    temp_root: Path | None = None
    try:
        temp_root, trimmed_metadata_path, trimmed_privileged_dir = _build_trimmed_assets(
            paths=paths,
            manifest=manifest,
            selection=selection,
            kept_frames=kept_frames,
        )
        _validate_trimmed_assets(
            trimmed_metadata_path=trimmed_metadata_path,
            expected_frame_count=selection.kept_frame_count,
        )

        backup_dir = _backup_dir(paths.recordings_root, paths.clip_name)
        originals = [
            (paths.metadata_path, backup_dir / paths.metadata_path.name),
            (paths.video_path, backup_dir / paths.video_path.name),
        ]
        replacements = [
            (trimmed_metadata_path, paths.metadata_path),
            (temp_root / paths.video_path.name, paths.video_path),
        ]
        if paths.privileged_dir is not None and trimmed_privileged_dir is not None:
            originals.append((paths.privileged_dir, backup_dir / paths.privileged_dir.name))
            replacements.append((trimmed_privileged_dir, paths.privileged_dir))

        _replace_with_backup(originals=originals, replacements=replacements)
    finally:
        if temp_root is not None and temp_root.exists():
            shutil.rmtree(temp_root, ignore_errors=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Trim the beginning and/or end of a blackbox clip while preserving exact metadata/video frame alignment.",
    )
    parser.add_argument(
        "clip_name",
        help="Bare clip stem such as capture_20260401_184018_638080",
    )
    parser.add_argument("trim_start", help="Seconds to trim from the start")
    parser.add_argument("trim_end", help="Seconds to trim from the end")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    trim_start_seconds = _parse_non_negative_seconds(args.trim_start, "trim_start")
    trim_end_seconds = _parse_non_negative_seconds(args.trim_end, "trim_end")
    _execute_trim(args.clip_name, trim_start_seconds, trim_end_seconds)


if __name__ == "__main__":
    main()
