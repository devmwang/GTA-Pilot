from __future__ import annotations

import collections
import json
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, TypeVar

import cv2
import numpy as np
import zmq

from gtapilot.config import BLACKBOX_PREROLL_SECONDS, BLACKBOX_RECORD_ON_START
from gtapilot.ipc.channel import ChannelTransportTracker
from gtapilot.ipc.channels import (
    INPUT_ACTIONS_CHANNEL,
    VISION_FRAMES_CHANNEL,
)
from gtapilot.ipc.settings_client import SettingsClient
from gtapilot.ipc.settings_registry import (
    SETTINGS_RPC_PORT,
    SETTINGS_UPDATES_PORT,
)
from gtapilot.ipc.settings_types import SettingValue
from gtapilot.ipc.types import (
    ChannelEnvelope,
    ChannelMessage,
    ChannelSpec,
    ChannelTransportEvent,
    ChannelTransportStats,
)

DEFAULT_OUTPUT_DIR = Path("blackbox-recordings")
SCHEMA_VERSION = 1
PREROLL_JPEG_QUALITY = 85
SETTINGS_REFRESH_INTERVAL_SECONDS = 0.5
VIDEO_CODEC = "libx264"
VIDEO_CONTAINER = "matroska"
VIDEO_FILE_SUFFIX = ".mkv"
VIDEO_PIXEL_FORMAT = "yuv420p"
VIDEO_PRESET = "veryfast"
VIDEO_CRF = 18
FPS_COMPARE_EPSILON = 1e-3
FRAME_WRITE_QUEUE_SIZE = 32
ACTION_WRITE_QUEUE_SIZE = 4096
WRITER_JOURNAL_FLUSH_ITEMS = 60
RECORDER_VISION_RCVHWM = 32
RECORDER_ACTION_RCVHWM = 128
POLL_TIMEOUT_MS = 100
POLL_TIMEOUT_SECONDS = POLL_TIMEOUT_MS / 1000.0
VISION_DRAIN_BATCH_LIMIT = 4
ACTION_DRAIN_BATCH_LIMIT = 128
PREROLL_FRAME_BUFFER_HEADROOM = 8
PREROLL_ACTION_BUFFER_HEADROOM = 32
INGEST_MODE_INACTIVE = "inactive"
INGEST_MODE_ACTIVE = "active"
INTEGRITY_EDGE_GRACE_SECONDS = 5.0
INTEGRITY_EDGE_GRACE_NS = int(INTEGRITY_EDGE_GRACE_SECONDS * 1_000_000_000)

T = TypeVar("T")


@dataclass(slots=True)
class BlackboxSessionPaths:
    video_path: Path
    metadata_path: Path
    frames_journal_path: Path
    actions_journal_path: Path
    session_timestamp: str


@dataclass(slots=True, frozen=True)
class VideoSessionConfig:
    width: int
    height: int
    nominal_fps: float


@dataclass(slots=True, frozen=True)
class AlignedActionData:
    payload: dict[str, Any] | None
    vector: list[float]
    envelope: dict[str, Any] | None


@dataclass(slots=True)
class BufferedFrame:
    frame_envelope: ChannelEnvelope
    jpeg_bytes: bytes
    aligned_action: AlignedActionData
    subscriber_received_timestamp_ns: int
    subscriber_queue_latency_ns: int


@dataclass(slots=True)
class FrameWriteTask:
    frame_rgb: np.ndarray
    frame_envelope: ChannelEnvelope
    aligned_action: AlignedActionData
    subscriber_received_timestamp_ns: int
    subscriber_queue_latency_ns: int


@dataclass(slots=True)
class RawSocketMessage:
    envelope: ChannelEnvelope
    payload_bytes: bytes
    subscriber_received_timestamp_ns: int


def _build_session_paths(output_dir: Path) -> BlackboxSessionPaths:
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_prefix = f"capture_{session_timestamp}"
    return BlackboxSessionPaths(
        video_path=output_dir / f"{output_prefix}_video{VIDEO_FILE_SUFFIX}",
        metadata_path=output_dir / f"{output_prefix}_metadata.json",
        frames_journal_path=output_dir / f"{output_prefix}_frames.tmp.jsonl",
        actions_journal_path=output_dir / f"{output_prefix}_actions.tmp.jsonl",
        session_timestamp=session_timestamp,
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        records.append(json.loads(stripped))
    return records


def _bounded_append(
    queue: collections.deque[T],
    item: T,
    *,
    capacity: int,
) -> None:
    queue.append(item)
    while len(queue) > capacity:
        queue.popleft()


def _aligned_message_before_with_cursor(
    timestamp_ns: int,
    latest_message: ChannelMessage[T] | None,
    drained_messages: list[ChannelMessage[T]],
    *,
    start_index: int = 0,
) -> tuple[ChannelMessage[T] | None, int]:
    aligned_message = None
    next_index = start_index
    if (
        latest_message is not None
        and latest_message.envelope.message_timestamp_ns <= timestamp_ns
    ):
        aligned_message = latest_message

    while next_index < len(drained_messages):
        message = drained_messages[next_index]
        if message.envelope.message_timestamp_ns > timestamp_ns:
            break
        aligned_message = message
        next_index += 1

    return aligned_message, next_index


def _normalize_nominal_fps(value: float) -> float:
    return round(float(value), 6)


def _format_nominal_fps(value: float) -> str:
    return f"{_normalize_nominal_fps(value):.6f}".rstrip("0").rstrip(".")


def _frame_config_from_parts(
    frame_metadata: dict[str, Any],
) -> VideoSessionConfig:
    raw_nominal_fps = frame_metadata.get("nominal_fps")
    if raw_nominal_fps is None:
        raise RuntimeError(
            "Blackbox recording requires frame metadata field 'nominal_fps'."
        )

    nominal_fps = float(raw_nominal_fps)
    if nominal_fps <= 0.0:
        raise RuntimeError(
            f"Blackbox recording requires positive 'nominal_fps', got {nominal_fps!r}."
        )

    return VideoSessionConfig(
        width=int(frame_metadata["w"]),
        height=int(frame_metadata["h"]),
        nominal_fps=_normalize_nominal_fps(nominal_fps),
    )


def _configs_match(left: VideoSessionConfig, right: VideoSessionConfig) -> bool:
    return (
        left.width == right.width
        and left.height == right.height
        and abs(left.nominal_fps - right.nominal_fps) <= FPS_COMPARE_EPSILON
    )


def _empty_aligned_action() -> AlignedActionData:
    return AlignedActionData(
        payload=None,
        vector=[0.0] * 6,
        envelope=None,
    )


def _aligned_action_from_message(
    action_message: ChannelMessage | None,
) -> AlignedActionData:
    if action_message is None:
        return _empty_aligned_action()
    return AlignedActionData(
        payload=action_message.payload.to_dict(),
        vector=action_message.payload.vector,
        envelope=action_message.envelope.to_dict(),
    )


def _subscriber_timestamps(
    envelope: ChannelEnvelope,
    subscriber_received_timestamp_ns: int | None,
) -> tuple[int, int]:
    received_timestamp_ns = int(
        subscriber_received_timestamp_ns or envelope.publish_timestamp_ns
    )
    return received_timestamp_ns, max(
        0,
        received_timestamp_ns - int(envelope.publish_timestamp_ns),
    )


def _action_stream_entry(action_message: ChannelMessage) -> dict[str, Any]:
    subscriber_received_timestamp_ns, subscriber_queue_latency_ns = (
        _subscriber_timestamps(
            action_message.envelope,
            action_message.subscriber_received_timestamp_ns,
        )
    )
    return {
        "envelope": action_message.envelope.to_dict(),
        "payload": action_message.payload.to_dict(),
        "action_vector": action_message.payload.vector,
        "subscriber_received_timestamp_ns": subscriber_received_timestamp_ns,
        "subscriber_queue_latency_ns": subscriber_queue_latency_ns,
    }


def _frame_entry_from_task(
    *,
    task: FrameWriteTask,
    session_paths: BlackboxSessionPaths,
    session_config: VideoSessionConfig,
    video_frame_index: int,
    writer_committed_timestamp_ns: int,
) -> dict[str, Any]:
    frame_metadata = task.frame_envelope.metadata
    height, width = task.frame_rgb.shape[:2]
    return {
        "video_file_name": session_paths.video_path.name,
        "video_frame_index": int(video_frame_index),
        "video_nominal_fps": float(session_config.nominal_fps),
        "video_codec": VIDEO_CODEC,
        "video_container": VIDEO_CONTAINER,
        "encoded_width": int(session_config.width),
        "encoded_height": int(session_config.height),
        "frame_id": int(frame_metadata.get("frame_id", -1)),
        "capture_frame_id": int(
            frame_metadata.get(
                "capture_frame_id",
                frame_metadata.get("frame_id", -1),
            )
        ),
        "capture_timestamp_ns": _frame_timestamp_ns_from_envelope(task.frame_envelope),
        "publish_timestamp_ns": int(task.frame_envelope.publish_timestamp_ns),
        "subscriber_received_timestamp_ns": int(task.subscriber_received_timestamp_ns),
        "writer_committed_timestamp_ns": int(writer_committed_timestamp_ns),
        "subscriber_queue_latency_ns": int(task.subscriber_queue_latency_ns),
        "resolution_height": int(frame_metadata.get("h", height)),
        "resolution_width": int(frame_metadata.get("w", width)),
        "frame_source": task.frame_envelope.source,
        "frame_envelope": task.frame_envelope.to_dict(),
        "frame_metadata": dict(frame_metadata),
        "action": None if task.aligned_action.payload is None else dict(task.aligned_action.payload),
        "action_envelope": (
            None if task.aligned_action.envelope is None else dict(task.aligned_action.envelope)
        ),
        "action_vector": list(task.aligned_action.vector),
    }


def _encode_preroll_jpeg(frame_rgb: np.ndarray) -> bytes | None:
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    success, buffer = cv2.imencode(
        ".jpg",
        frame_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), PREROLL_JPEG_QUALITY],
    )
    if not success:
        return None
    return buffer.tobytes()


def _decode_preroll_jpeg_to_rgb(jpeg_bytes: bytes) -> np.ndarray | None:
    frame_buffer = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    frame_bgr = cv2.imdecode(frame_buffer, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        return None
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def _percentile_ns(values: list[int], percentile: float) -> int:
    if not values:
        return 0
    ordered = sorted(int(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * max(0.0, min(100.0, float(percentile))) / 100.0
    lower_index = int(position)
    upper_index = min(len(ordered) - 1, lower_index + 1)
    blend = position - lower_index
    lower = ordered[lower_index]
    upper = ordered[upper_index]
    return int(round(lower + (upper - lower) * blend))


def _timing_summary_ns(values: list[int]) -> dict[str, int]:
    if not values:
        return {"count": 0, "p50": 0, "p95": 0, "max": 0}
    normalized = [max(0, int(value)) for value in values]
    return {
        "count": int(len(normalized)),
        "p50": _percentile_ns(normalized, 50.0),
        "p95": _percentile_ns(normalized, 95.0),
        "max": max(normalized),
    }


def _preroll_capacity(preroll_seconds: float, headroom: int) -> int:
    return max(
        1,
        int(round(max(0.0, float(preroll_seconds)) * 60.0))
        + int(headroom),
    )


def _frame_timestamp_ns_from_envelope(envelope: ChannelEnvelope) -> int:
    return int(
        envelope.metadata.get(
            "capture_timestamp_ns",
            envelope.message_timestamp_ns,
        )
    )


def _buffered_frame_timestamp_ns(frame: BufferedFrame) -> int:
    return _frame_timestamp_ns_from_envelope(frame.frame_envelope)


def _build_sub_socket(
    context: zmq.Context,
    spec: ChannelSpec,
    *,
    host: str,
    rcvhwm: int,
) -> zmq.Socket:
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.RCVHWM, int(rcvhwm))
    socket.connect(f"tcp://{host}:{spec.port}")
    socket.subscribe(spec.topic)
    return socket


def _delta_transport_stats(
    current: ChannelTransportStats,
    baseline: ChannelTransportStats | None,
) -> dict[str, Any]:
    if baseline is None:
        return current.to_dict()
    return {
        "channel": current.channel,
        "messages_received": int(current.messages_received - baseline.messages_received),
        "sequence_gap_count": int(current.sequence_gap_count - baseline.sequence_gap_count),
        "missing_message_count": int(
            current.missing_message_count - baseline.missing_message_count
        ),
        "local_overflow_count": int(
            current.local_overflow_count - baseline.local_overflow_count
        ),
        "local_overflow_dropped_messages": int(
            current.local_overflow_dropped_messages
            - baseline.local_overflow_dropped_messages
        ),
        "max_buffer_occupancy": int(current.max_buffer_occupancy),
        "last_sequence_by_source": {
            str(source): int(sequence_id)
            for source, sequence_id in current.last_sequence_by_source.items()
        },
    }


def _frame_gap_stats(frame_payloads: list[dict[str, Any]]) -> dict[str, int]:
    published_gap_count = 0
    published_gap_total = 0
    capture_gap_count = 0
    capture_gap_total = 0
    last_frame_id: int | None = None
    last_capture_frame_id: int | None = None

    for frame_payload in frame_payloads:
        frame_id = int(frame_payload.get("frame_id", -1))
        capture_frame_id = int(frame_payload.get("capture_frame_id", frame_id))
        if last_frame_id is not None and frame_id > last_frame_id + 1:
            published_gap_count += 1
            published_gap_total += frame_id - last_frame_id - 1
        last_frame_id = frame_id

        if last_capture_frame_id is None:
            last_capture_frame_id = capture_frame_id
            continue
        if capture_frame_id == last_capture_frame_id:
            continue
        if capture_frame_id > last_capture_frame_id + 1:
            capture_gap_count += 1
            capture_gap_total += capture_frame_id - last_capture_frame_id - 1
        last_capture_frame_id = capture_frame_id

    return {
        "published_gap_count": int(published_gap_count),
        "published_gap_total": int(published_gap_total),
        "capture_gap_count": int(capture_gap_count),
        "capture_gap_total": int(capture_gap_total),
    }


def _session_data_window_ns(
    *,
    frame_payloads: list[dict[str, Any]],
    action_payloads: list[dict[str, Any]],
    created_timestamp_ns: int,
) -> tuple[int, int]:
    timestamps_ns: list[int] = []
    for frame_payload in frame_payloads:
        timestamp_ns = int(
            frame_payload.get(
                "capture_timestamp_ns",
                frame_payload.get("publish_timestamp_ns", 0),
            )
            or 0
        )
        if timestamp_ns > 0:
            timestamps_ns.append(timestamp_ns)

    for action_payload in action_payloads:
        envelope = dict(action_payload.get("envelope") or {})
        timestamp_ns = int(
            envelope.get(
                "message_timestamp_ns",
                action_payload.get("subscriber_received_timestamp_ns", 0),
            )
            or 0
        )
        if timestamp_ns > 0:
            timestamps_ns.append(timestamp_ns)

    if not timestamps_ns:
        fallback_timestamp_ns = max(0, int(created_timestamp_ns))
        return fallback_timestamp_ns, fallback_timestamp_ns

    return min(timestamps_ns), max(timestamps_ns)


def _is_grace_filterable_drop_event(event: dict[str, Any]) -> bool:
    kind = str(event.get("kind", ""))
    channel = str(event.get("channel", ""))
    if kind in {"writer_frame_queue_overflow", "writer_action_queue_overflow"}:
        return True
    if kind == "native_overload":
        return True
    if kind in {"sequence_gap", "local_overflow"} and channel in {
        VISION_FRAMES_CHANNEL.name,
        INPUT_ACTIONS_CHANNEL.name,
    }:
        return True
    return False


def _split_drop_events_for_integrity(
    *,
    drop_events: list[dict[str, Any]],
    session_start_timestamp_ns: int,
    session_end_timestamp_ns: int,
    grace_window_ns: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not drop_events:
        return [], []

    effective_drop_events: list[dict[str, Any]] = []
    ignored_drop_events: list[dict[str, Any]] = []
    grace_window_ns = max(0, int(grace_window_ns))
    for drop_event in drop_events:
        if grace_window_ns <= 0 or not _is_grace_filterable_drop_event(drop_event):
            effective_drop_events.append(drop_event)
            continue

        timestamp_ns = int(drop_event.get("timestamp_ns", 0) or 0)
        if timestamp_ns <= 0:
            effective_drop_events.append(drop_event)
            continue

        if (
            timestamp_ns <= session_start_timestamp_ns + grace_window_ns
            or timestamp_ns >= session_end_timestamp_ns - grace_window_ns
        ):
            ignored_drop_events.append(dict(drop_event))
            continue

        effective_drop_events.append(drop_event)

    return effective_drop_events, ignored_drop_events


def _native_capture_timing_summary(
    frame_payloads: list[dict[str, Any]],
) -> dict[str, Any]:
    latest_pipeline_stats: dict[str, Any] = {}
    sample_capture_frame_id: int | None = None
    frame_arrival_wait_ns: list[int] = []
    gpu_readback_ns: list[int] = []
    cpu_convert_ns: list[int] = []
    publish_deadline_lag_ns: list[int] = []

    for frame_payload in frame_payloads:
        pipeline_stats = (frame_payload.get("frame_metadata") or {}).get("pipeline_stats")
        if not isinstance(pipeline_stats, dict):
            continue
        latest_pipeline_stats = dict(pipeline_stats)
        current_sample_capture_frame_id = int(
            pipeline_stats.get(
                "sample_capture_frame_id",
                frame_payload.get("capture_frame_id", frame_payload.get("frame_id", -1)),
            )
        )
        if (
            sample_capture_frame_id is not None
            and current_sample_capture_frame_id == sample_capture_frame_id
        ):
            continue
        sample_capture_frame_id = current_sample_capture_frame_id

        for bucket, key in (
            (frame_arrival_wait_ns, "frame_arrival_wait_ns"),
            (gpu_readback_ns, "gpu_readback_ns"),
            (cpu_convert_ns, "cpu_convert_ns"),
            (publish_deadline_lag_ns, "publish_deadline_lag_ns"),
        ):
            value = pipeline_stats.get(key)
            if value is not None:
                bucket.append(int(value))

    return {
        "sample_count": int(len(frame_arrival_wait_ns)),
        "frame_arrival_wait_ns": _timing_summary_ns(frame_arrival_wait_ns),
        "gpu_readback_ns": _timing_summary_ns(gpu_readback_ns),
        "cpu_convert_ns": _timing_summary_ns(cpu_convert_ns),
        "publish_deadline_lag_ns": _timing_summary_ns(publish_deadline_lag_ns),
        "latest_pipeline_stats": latest_pipeline_stats,
    }


class FFmpegVideoWriter:
    def __init__(self, *, output_path: Path, config: VideoSessionConfig):
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path is None:
            raise RuntimeError(
                "ffmpeg was not found on PATH. Blackbox video recording requires ffmpeg."
            )

        self.output_path = output_path
        self.config = config
        self.command = [
            ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{config.width}x{config.height}",
            "-framerate",
            _format_nominal_fps(config.nominal_fps),
            "-i",
            "pipe:0",
            "-an",
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
        ]
        self._process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        self._closed = False

    def _read_stderr_text(self) -> str:
        if self._process.stderr is None:
            return ""
        try:
            return self._process.stderr.read().decode("utf-8", errors="replace").strip()
        except Exception:
            return ""

    def _raise_process_failure(self, context: str) -> None:
        stderr_text = self._read_stderr_text()
        detail = f" {stderr_text}" if stderr_text else ""
        raise RuntimeError(f"{context}.{detail}".strip())

    def write_frame(self, frame_rgb: np.ndarray) -> None:
        if self._closed:
            raise RuntimeError("ffmpeg video writer is already closed.")

        if (
            frame_rgb.ndim != 3
            or frame_rgb.shape[0] != self.config.height
            or frame_rgb.shape[1] != self.config.width
            or frame_rgb.shape[2] != 3
        ):
            raise RuntimeError(
                "Frame shape does not match active blackbox video session format."
            )

        if self._process.stdin is None:
            self._raise_process_failure("ffmpeg stdin is not available")

        frame_rgb = np.ascontiguousarray(frame_rgb)
        try:
            self._process.stdin.write(memoryview(frame_rgb))
        except BrokenPipeError:
            self._raise_process_failure("ffmpeg exited while writing blackbox video")

    def close(self) -> None:
        if self._closed:
            return

        self._closed = True
        if self._process.stdin is not None:
            try:
                self._process.stdin.close()
            except Exception:
                pass

        try:
            return_code = self._process.wait(timeout=15.0)
        except subprocess.TimeoutExpired:
            self._process.kill()
            return_code = self._process.wait(timeout=5.0)

        if return_code != 0:
            self._raise_process_failure(
                f"ffmpeg exited with status {return_code} for {self.output_path.name}"
            )


class BlackboxSessionWriter:
    def __init__(self, *, paths: BlackboxSessionPaths, config: VideoSessionConfig):
        self.paths = paths
        self.config = config
        self._video_writer = FFmpegVideoWriter(output_path=paths.video_path, config=config)
        self._frames_journal = paths.frames_journal_path.open("w", encoding="utf-8")
        self._actions_journal = paths.actions_journal_path.open("w", encoding="utf-8")
        self._frame_queue: collections.deque[FrameWriteTask] = collections.deque()
        self._action_queue: collections.deque[dict[str, Any]] = collections.deque()
        self._queue_lock = threading.Lock()
        self._available = threading.Condition(self._queue_lock)
        self._stop_requested = False
        self._writer_error: Exception | None = None
        self._frames_written = 0
        self._actions_written = 0
        self._frame_queue_max_depth = 0
        self._action_queue_max_depth = 0
        self._frame_queue_overflow_count = 0
        self._frame_queue_dropped_frames = 0
        self._action_queue_overflow_count = 0
        self._action_queue_dropped_entries = 0
        self._writer_lag_ns: list[int] = []
        self._drop_events: list[dict[str, Any]] = []
        self._frame_flush_items = 0
        self._action_flush_items = 0
        self._prefer_frame_next = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _append_drop_event(self, payload: dict[str, Any]) -> None:
        self._drop_events.append(payload)

    def _enqueue_bounded(
        self,
        queue: collections.deque[T],
        item: T,
        *,
        capacity: int,
        max_depth_attr: str,
        overflow_count_attr: str,
        dropped_count_attr: str,
        drop_event: Callable[[T], dict[str, Any]],
    ) -> None:
        if len(queue) >= capacity and queue:
            dropped_item = queue.popleft()
            setattr(
                self,
                overflow_count_attr,
                int(getattr(self, overflow_count_attr)) + 1,
            )
            setattr(
                self,
                dropped_count_attr,
                int(getattr(self, dropped_count_attr)) + 1,
            )
            self._append_drop_event(drop_event(dropped_item))
        queue.append(item)
        setattr(
            self,
            max_depth_attr,
            max(int(getattr(self, max_depth_attr)), len(queue)),
        )
        self._available.notify_all()

    def _action_overflow_event(self, dropped_entry: dict[str, Any]) -> dict[str, Any]:
        envelope = dict(dropped_entry.get("envelope", {}))
        sequence_id = envelope.get("sequence_id")
        return {
            "kind": "writer_action_queue_overflow",
            "channel": "blackbox.writer",
            "source": str(envelope.get("source", "unknown")),
            "timestamp_ns": time.time_ns(),
            "sequence_id_start": None if sequence_id is None else int(sequence_id),
            "sequence_id_end": None if sequence_id is None else int(sequence_id),
            "dropped_count": 1,
            "details": {"queue_capacity": ACTION_WRITE_QUEUE_SIZE},
        }

    def _frame_overflow_event(self, dropped_task: FrameWriteTask) -> dict[str, Any]:
        frame_metadata = dropped_task.frame_envelope.metadata
        frame_id = int(frame_metadata.get("frame_id", -1))
        return {
            "kind": "writer_frame_queue_overflow",
            "channel": "blackbox.writer",
            "source": dropped_task.frame_envelope.source,
            "timestamp_ns": time.time_ns(),
            "sequence_id_start": int(dropped_task.frame_envelope.sequence_id),
            "sequence_id_end": int(dropped_task.frame_envelope.sequence_id),
            "dropped_count": 1,
            "details": {
                "queue_capacity": FRAME_WRITE_QUEUE_SIZE,
                "frame_id": frame_id,
                "capture_frame_id": int(
                    frame_metadata.get("capture_frame_id", frame_id)
                ),
            },
        }

    def enqueue_action(self, action_entry: dict[str, Any]) -> None:
        with self._queue_lock:
            if self._writer_error is not None:
                raise self._writer_error
            self._enqueue_bounded(
                self._action_queue,
                dict(action_entry),
                capacity=ACTION_WRITE_QUEUE_SIZE,
                max_depth_attr="_action_queue_max_depth",
                overflow_count_attr="_action_queue_overflow_count",
                dropped_count_attr="_action_queue_dropped_entries",
                drop_event=self._action_overflow_event,
            )

    def enqueue_frame(self, task: FrameWriteTask) -> None:
        with self._queue_lock:
            if self._writer_error is not None:
                raise self._writer_error
            self._enqueue_bounded(
                self._frame_queue,
                task,
                capacity=FRAME_WRITE_QUEUE_SIZE,
                max_depth_attr="_frame_queue_max_depth",
                overflow_count_attr="_frame_queue_overflow_count",
                dropped_count_attr="_frame_queue_dropped_frames",
                drop_event=self._frame_overflow_event,
            )

    def _write_action_entry(self, action_entry: dict[str, Any]) -> None:
        action_payload = dict(action_entry)
        action_payload["writer_committed_timestamp_ns"] = time.time_ns()
        self._actions_journal.write(json.dumps(action_payload) + "\n")
        self._actions_written += 1
        self._action_flush_items += 1
        if self._action_flush_items >= WRITER_JOURNAL_FLUSH_ITEMS:
            self._actions_journal.flush()
            self._action_flush_items = 0

    def _write_frame_task(self, task: FrameWriteTask) -> None:
        self._video_writer.write_frame(task.frame_rgb)
        writer_committed_timestamp_ns = time.time_ns()
        self._frames_written += 1
        frame_entry = _frame_entry_from_task(
            task=task,
            session_paths=self.paths,
            session_config=self.config,
            video_frame_index=self._frames_written,
            writer_committed_timestamp_ns=writer_committed_timestamp_ns,
        )
        self._frames_journal.write(json.dumps(frame_entry) + "\n")
        self._frame_flush_items += 1
        if self._frame_flush_items >= WRITER_JOURNAL_FLUSH_ITEMS:
            self._frames_journal.flush()
            self._frame_flush_items = 0
        self._writer_lag_ns.append(
            max(
                0,
                writer_committed_timestamp_ns - int(task.subscriber_received_timestamp_ns),
            )
        )

    def _run(self) -> None:
        try:
            while True:
                with self._queue_lock:
                    while (
                        not self._stop_requested
                        and not self._action_queue
                        and not self._frame_queue
                    ):
                        self._available.wait(timeout=0.5)
                    if (
                        self._stop_requested
                        and not self._action_queue
                        and not self._frame_queue
                    ):
                        break
                    frame_task: FrameWriteTask | None = None
                    action_entry: dict[str, Any] | None = None
                    if self._frame_queue and self._action_queue:
                        if self._prefer_frame_next:
                            frame_task = self._frame_queue.popleft()
                        else:
                            action_entry = self._action_queue.popleft()
                        self._prefer_frame_next = not self._prefer_frame_next
                    elif self._frame_queue:
                        frame_task = self._frame_queue.popleft()
                        self._prefer_frame_next = False
                    elif self._action_queue:
                        action_entry = self._action_queue.popleft()
                        self._prefer_frame_next = True
                if action_entry is not None:
                    self._write_action_entry(action_entry)
                    continue
                if frame_task is not None:
                    self._write_frame_task(frame_task)
        except Exception as exc:
            self._writer_error = exc

    def close(self) -> dict[str, Any]:
        with self._queue_lock:
            self._stop_requested = True
            self._available.notify_all()
        self._thread.join(timeout=30.0)
        if self._thread.is_alive():
            self._writer_error = self._writer_error or RuntimeError(
                "Blackbox session writer thread did not stop cleanly."
            )

        close_error: Exception | None = None
        try:
            self._frames_journal.flush()
            self._actions_journal.flush()
        except Exception as exc:
            close_error = exc
        finally:
            try:
                self._frames_journal.close()
            except Exception:
                pass
            try:
                self._actions_journal.close()
            except Exception:
                pass
        try:
            self._video_writer.close()
        except Exception as exc:
            close_error = close_error or exc

        if self._writer_error is not None:
            raise self._writer_error
        if close_error is not None:
            raise close_error
        return self.snapshot_writer_stats()

    def snapshot_writer_stats(self) -> dict[str, Any]:
        return {
            "frame_queue_capacity": FRAME_WRITE_QUEUE_SIZE,
            "frame_queue_max_depth": int(self._frame_queue_max_depth),
            "frame_queue_overflow_count": int(self._frame_queue_overflow_count),
            "frame_queue_dropped_frames": int(self._frame_queue_dropped_frames),
            "action_queue_capacity": ACTION_WRITE_QUEUE_SIZE,
            "action_queue_max_depth": int(self._action_queue_max_depth),
            "action_queue_overflow_count": int(self._action_queue_overflow_count),
            "action_queue_dropped_entries": int(self._action_queue_dropped_entries),
            "frames_written": int(self._frames_written),
            "actions_written": int(self._actions_written),
            "writer_lag_ns": {
                "p50": _percentile_ns(self._writer_lag_ns, 50.0),
                "p95": _percentile_ns(self._writer_lag_ns, 95.0),
                "max": max(self._writer_lag_ns) if self._writer_lag_ns else 0,
            },
        }

    @property
    def drop_events(self) -> list[dict[str, Any]]:
        return list(self._drop_events)


class BlackboxRecorder:
    def __init__(
        self,
        *,
        output_dir: str | Path | None = None,
        vision_channel: ChannelSpec = VISION_FRAMES_CHANNEL,
        action_channel: ChannelSpec = INPUT_ACTIONS_CHANNEL,
        settings_client: SettingsClient | None = None,
        settings_host: str = "127.0.0.1",
        settings_updates_port: str = SETTINGS_UPDATES_PORT,
        settings_rpc_port: str = SETTINGS_RPC_PORT,
        record_on_start: bool = BLACKBOX_RECORD_ON_START,
        preroll_seconds: float = BLACKBOX_PREROLL_SECONDS,
    ):
        self.output_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.vision_channel = vision_channel
        self.action_channel = action_channel
        self.context = zmq.Context()
        self.poller = zmq.Poller()
        self.vision_socket = _build_sub_socket(
            self.context,
            vision_channel,
            host="127.0.0.1",
            rcvhwm=RECORDER_VISION_RCVHWM,
        )
        self.action_socket = _build_sub_socket(
            self.context,
            action_channel,
            host="127.0.0.1",
            rcvhwm=RECORDER_ACTION_RCVHWM,
        )
        self.poller.register(self.vision_socket, zmq.POLLIN)
        self.poller.register(self.action_socket, zmq.POLLIN)
        self.vision_transport_tracker = ChannelTransportTracker(
            vision_channel.name,
            event_callback=self._handle_transport_event,
        )
        self.action_transport_tracker = ChannelTransportTracker(
            action_channel.name,
            event_callback=self._handle_transport_event,
        )

        self.settings_client = settings_client or SettingsClient(
            source_name="blackbox",
            host=settings_host,
            updates_port=settings_updates_port,
            rpc_port=settings_rpc_port,
        )
        self._owns_settings_client = settings_client is None
        if self._owns_settings_client:
            self.settings_client.start()

        self.recording_enabled = bool(record_on_start)
        self.preroll_seconds = max(0.0, float(preroll_seconds))
        self.preroll_ns = int(self.preroll_seconds * 1_000_000_000)
        self.recording_setting_revision = 0
        self.recording_setting_updated_timestamp_ns = 0
        self._last_settings_refresh_monotonic = 0.0

        self.latest_action_message: ChannelMessage | None = None
        self.preroll_frames: collections.deque[BufferedFrame] = collections.deque()
        self.preroll_actions: collections.deque[ChannelMessage] = collections.deque()
        self.preroll_transport_events: collections.deque[ChannelTransportEvent] = (
            collections.deque()
        )

        self.paths: BlackboxSessionPaths | None = None
        self.session_config: VideoSessionConfig | None = None
        self.session_writer: BlackboxSessionWriter | None = None
        self._written_action_keys: set[tuple[str, int]] = set()
        self._session_transport_baselines: dict[str, ChannelTransportStats] = {}
        self._session_drop_events: list[dict[str, Any]] = []
        self._session_native_overload_active = False
        self._session_created_timestamp_ns = 0
        self._vision_frames_decoded_total = 0
        self._vision_frames_decoded_while_inactive = 0
        self._session_vision_frames_decoded_baseline = 0
        self._session_vision_frames_decoded_while_inactive_baseline = 0
        self._current_ingest_mode = INGEST_MODE_ACTIVE

        initial_recording_setting = self._refresh_runtime_settings()
        if initial_recording_setting is not None:
            self.recording_enabled = bool(initial_recording_setting.value)
            self.recording_setting_revision = int(initial_recording_setting.revision)
            self.recording_setting_updated_timestamp_ns = int(
                initial_recording_setting.updated_timestamp_ns
            )
        self._current_ingest_mode = self._ingest_mode_for(self.recording_enabled)

    def _session_active(self) -> bool:
        return (
            self.session_writer is not None
            and self.paths is not None
            and self.session_config is not None
        )

    def _ingest_mode_for(self, recording_enabled: bool) -> str:
        if bool(recording_enabled) or self.preroll_ns > 0:
            return INGEST_MODE_ACTIVE
        return INGEST_MODE_INACTIVE

    def _clear_preroll_buffers(self) -> None:
        self.preroll_frames.clear()
        self.preroll_actions.clear()
        self.preroll_transport_events.clear()

    def _refresh_runtime_settings(self) -> SettingValue | None:
        now = time.monotonic()
        if now - self._last_settings_refresh_monotonic >= SETTINGS_REFRESH_INTERVAL_SECONDS:
            try:
                self.settings_client.refresh_snapshot()
            except Exception:
                pass
            self._last_settings_refresh_monotonic = now
        preroll_seconds = self.settings_client.get(
            "blackbox.preroll_seconds",
            self.preroll_seconds,
        )
        self.preroll_seconds = max(0.0, float(preroll_seconds))
        self.preroll_ns = int(self.preroll_seconds * 1_000_000_000)
        return self.settings_client.get_setting_value("blackbox.recording_enabled")

    def _recording_enabled_at(
        self,
        timestamp_ns: int,
        *,
        current_recording_enabled: bool,
        current_revision: int,
        current_updated_timestamp_ns: int,
        previous_recording_enabled: bool,
        previous_revision: int,
    ) -> bool:
        if current_revision <= previous_revision:
            return current_recording_enabled
        if timestamp_ns < current_updated_timestamp_ns:
            return previous_recording_enabled
        return current_recording_enabled

    def _handle_transport_event(self, event: ChannelTransportEvent) -> None:
        if not self._session_active():
            _bounded_append(
                self.preroll_transport_events,
                event,
                capacity=_preroll_capacity(
                    self.preroll_seconds,
                    PREROLL_ACTION_BUFFER_HEADROOM,
                ),
            )
            return
        self._session_drop_events.append(event.to_dict())

    def _handle_native_pipeline_telemetry(self, frame_metadata: dict[str, Any]) -> None:
        if not self._session_active():
            return
        pipeline_stats = frame_metadata.get("pipeline_stats")
        if not isinstance(pipeline_stats, dict):
            return
        overload_active = bool(pipeline_stats.get("overload_active", False))
        if overload_active and not self._session_native_overload_active:
            self._session_drop_events.append(
                {
                    "kind": "native_overload",
                    "channel": self.vision_channel.name,
                    "source": "display_capture_dx11",
                    "timestamp_ns": int(
                        frame_metadata.get("capture_timestamp_ns", time.time_ns())
                    ),
                    "details": dict(pipeline_stats),
                }
            )
        self._session_native_overload_active = overload_active

    def _recv_raw_message(
        self,
        socket: zmq.Socket,
        *,
        spec: ChannelSpec,
        transport_tracker: ChannelTransportTracker,
    ) -> RawSocketMessage | None:
        try:
            topic, envelope_bytes, payload_bytes = socket.recv_multipart(flags=zmq.NOBLOCK)
        except zmq.Again:
            return None
        subscriber_received_timestamp_ns = time.time_ns()
        if topic != spec.topic:
            return None
        envelope = ChannelEnvelope.from_dict(json.loads(envelope_bytes.decode("utf-8")))
        if envelope.channel != spec.name or envelope.encoding != spec.codec.encoding_name:
            return None
        transport_tracker.note_received(
            envelope,
            timestamp_ns=subscriber_received_timestamp_ns,
        )
        transport_tracker.dispatch_pending_callbacks()
        return RawSocketMessage(
            envelope=envelope,
            payload_bytes=payload_bytes,
            subscriber_received_timestamp_ns=subscriber_received_timestamp_ns,
        )

    def _decode_raw_message(
        self,
        raw_message: RawSocketMessage,
        *,
        spec: ChannelSpec,
    ) -> ChannelMessage[Any]:
        decoded_message = ChannelMessage(
            envelope=raw_message.envelope,
            payload=spec.codec.decode(raw_message.payload_bytes, raw_message.envelope),
            subscriber_received_timestamp_ns=raw_message.subscriber_received_timestamp_ns,
        )
        if spec.name == self.vision_channel.name:
            self._vision_frames_decoded_total += 1
            if self._current_ingest_mode == INGEST_MODE_INACTIVE:
                self._vision_frames_decoded_while_inactive += 1
        return decoded_message

    def _drain_raw_messages(
        self,
        socket: zmq.Socket,
        *,
        spec: ChannelSpec,
        transport_tracker: ChannelTransportTracker,
        max_messages: int | None = None,
    ) -> list[RawSocketMessage]:
        messages: list[RawSocketMessage] = []
        while True:
            if max_messages is not None and len(messages) >= max_messages:
                break
            message = self._recv_raw_message(
                socket,
                spec=spec,
                transport_tracker=transport_tracker,
            )
            if message is None:
                break
            messages.append(message)
        return messages

    def _discard_socket_backlog(
        self,
        socket: zmq.Socket,
        *,
        spec: ChannelSpec,
        transport_tracker: ChannelTransportTracker,
    ) -> None:
        while True:
            if (
                self._recv_raw_message(
                    socket,
                    spec=spec,
                    transport_tracker=transport_tracker,
                )
                is None
            ):
                break

    def _prune_preroll(self, reference_timestamp_ns: int) -> None:
        if self.preroll_ns <= 0:
            self._clear_preroll_buffers()
            return

        cutoff_timestamp_ns = reference_timestamp_ns - self.preroll_ns
        while (
            self.preroll_frames
            and _buffered_frame_timestamp_ns(self.preroll_frames[0])
            < cutoff_timestamp_ns
        ):
            self.preroll_frames.popleft()
        while (
            self.preroll_actions
            and self.preroll_actions[0].envelope.message_timestamp_ns < cutoff_timestamp_ns
        ):
            self.preroll_actions.popleft()
        while (
            self.preroll_transport_events
            and self.preroll_transport_events[0].timestamp_ns < cutoff_timestamp_ns
        ):
            self.preroll_transport_events.popleft()

    def _buffer_preroll_frame(
        self,
        frame_message: ChannelMessage[np.ndarray],
        aligned_action: AlignedActionData,
    ) -> None:
        if self.preroll_ns <= 0:
            return

        jpeg_bytes = _encode_preroll_jpeg(frame_message.payload)
        if jpeg_bytes is None:
            return

        subscriber_received_timestamp_ns, subscriber_queue_latency_ns = (
            _subscriber_timestamps(
                frame_message.envelope,
                frame_message.subscriber_received_timestamp_ns,
            )
        )
        _bounded_append(
            self.preroll_frames,
            BufferedFrame(
                frame_envelope=frame_message.envelope,
                jpeg_bytes=jpeg_bytes,
                aligned_action=aligned_action,
                subscriber_received_timestamp_ns=subscriber_received_timestamp_ns,
                subscriber_queue_latency_ns=subscriber_queue_latency_ns,
            ),
            capacity=_preroll_capacity(
                self.preroll_seconds,
                PREROLL_FRAME_BUFFER_HEADROOM,
            ),
        )

    def _frame_task_from_message(
        self,
        frame_message: ChannelMessage[np.ndarray],
        aligned_action: AlignedActionData,
    ) -> FrameWriteTask:
        subscriber_received_timestamp_ns, subscriber_queue_latency_ns = (
            _subscriber_timestamps(
                frame_message.envelope,
                frame_message.subscriber_received_timestamp_ns,
            )
        )
        frame_rgb = np.asarray(frame_message.payload, dtype=np.uint8)
        if not frame_rgb.flags["C_CONTIGUOUS"]:
            frame_rgb = np.ascontiguousarray(frame_rgb)
        return FrameWriteTask(
            frame_rgb=frame_rgb,
            frame_envelope=frame_message.envelope,
            aligned_action=aligned_action,
            subscriber_received_timestamp_ns=subscriber_received_timestamp_ns,
            subscriber_queue_latency_ns=subscriber_queue_latency_ns,
        )

    def _frame_task_from_buffered(self, buffered_frame: BufferedFrame) -> FrameWriteTask | None:
        frame_rgb = _decode_preroll_jpeg_to_rgb(buffered_frame.jpeg_bytes)
        if frame_rgb is None:
            return None
        return FrameWriteTask(
            frame_rgb=frame_rgb,
            frame_envelope=buffered_frame.frame_envelope,
            aligned_action=buffered_frame.aligned_action,
            subscriber_received_timestamp_ns=buffered_frame.subscriber_received_timestamp_ns,
            subscriber_queue_latency_ns=buffered_frame.subscriber_queue_latency_ns,
        )

    def _enqueue_action_message(self, action_message: ChannelMessage) -> None:
        if not self._session_active() or self.session_writer is None:
            return
        action_key = (
            action_message.envelope.source,
            int(action_message.envelope.sequence_id),
        )
        if action_key in self._written_action_keys:
            return
        self._written_action_keys.add(action_key)
        self.session_writer.enqueue_action(_action_stream_entry(action_message))

    def _append_matching_preroll_frames(self) -> None:
        if self.session_writer is None or self.session_config is None:
            return
        for buffered_frame in self.preroll_frames:
            buffered_config = _frame_config_from_parts(
                buffered_frame.frame_envelope.metadata
            )
            if not _configs_match(buffered_config, self.session_config):
                continue
            task = self._frame_task_from_buffered(buffered_frame)
            if task is None:
                continue
            self._handle_native_pipeline_telemetry(task.frame_envelope.metadata)
            self.session_writer.enqueue_frame(task)
        self.preroll_frames.clear()

    def start_session(self, session_config: VideoSessionConfig) -> None:
        if self._session_active():
            return

        self.paths = _build_session_paths(self.output_dir)
        self.session_config = session_config
        self.session_writer = BlackboxSessionWriter(
            paths=self.paths,
            config=session_config,
        )
        self._written_action_keys.clear()
        self._session_drop_events = []
        self._session_native_overload_active = False
        self._session_created_timestamp_ns = time.time_ns()
        self._session_vision_frames_decoded_baseline = int(
            self._vision_frames_decoded_total
        )
        self._session_vision_frames_decoded_while_inactive_baseline = int(
            self._vision_frames_decoded_while_inactive
        )
        self._session_transport_baselines = {
            "vision": self.vision_transport_tracker.snapshot(),
            "actions": self.action_transport_tracker.snapshot(),
        }
        self._session_drop_events.extend(
            event.to_dict() for event in self.preroll_transport_events
        )
        self.preroll_transport_events.clear()

        for action_message in self.preroll_actions:
            self._enqueue_action_message(action_message)
        self.preroll_actions.clear()
        self._append_matching_preroll_frames()

    def _final_manifest_payload(
        self,
        *,
        frame_payloads: list[dict[str, Any]],
        action_payloads: list[dict[str, Any]],
        writer_stats: dict[str, Any],
        writer_drop_events: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if self.paths is None or self.session_config is None:
            raise RuntimeError("Blackbox session paths/config are not available.")

        transport_stats = {
            "vision": _delta_transport_stats(
                self.vision_transport_tracker.snapshot(),
                self._session_transport_baselines.get("vision"),
            ),
            "actions": _delta_transport_stats(
                self.action_transport_tracker.snapshot(),
                self._session_transport_baselines.get("actions"),
            ),
        }
        raw_drop_events = list(self._session_drop_events) + list(writer_drop_events)
        gap_stats = _frame_gap_stats(frame_payloads)
        repeat_frame_count = sum(
            1
            for frame_payload in frame_payloads
            if bool((frame_payload.get("frame_metadata") or {}).get("is_repeat", False))
        )
        fresh_capture_frame_count = 0
        last_capture_frame_id: int | None = None
        native_capture_performance = _native_capture_timing_summary(frame_payloads)
        native_pipeline_summary = dict(
            native_capture_performance.get("latest_pipeline_stats", {})
        )
        for frame_payload in frame_payloads:
            capture_frame_id = int(
                frame_payload.get("capture_frame_id", frame_payload.get("frame_id", -1))
            )
            if last_capture_frame_id is None or capture_frame_id != last_capture_frame_id:
                fresh_capture_frame_count += 1
                last_capture_frame_id = capture_frame_id

        session_start_timestamp_ns, session_end_timestamp_ns = _session_data_window_ns(
            frame_payloads=frame_payloads,
            action_payloads=action_payloads,
            created_timestamp_ns=self._session_created_timestamp_ns,
        )
        drop_events, ignored_drop_events = _split_drop_events_for_integrity(
            drop_events=raw_drop_events,
            session_start_timestamp_ns=session_start_timestamp_ns,
            session_end_timestamp_ns=session_end_timestamp_ns,
            grace_window_ns=INTEGRITY_EDGE_GRACE_NS,
        )

        integrity_degraded = bool(drop_events)

        performance_stats = {
            "native_capture": native_capture_performance,
            "blackbox_ingest": {
                "session_mode": INGEST_MODE_ACTIVE,
                "vision_frames_decoded_total": max(
                    0,
                    int(self._vision_frames_decoded_total)
                    - int(self._session_vision_frames_decoded_baseline),
                ),
                "vision_frames_decoded_while_inactive": int(
                    max(
                        0,
                        int(self._vision_frames_decoded_while_inactive)
                        - int(
                            self._session_vision_frames_decoded_while_inactive_baseline
                        ),
                    )
                ),
            },
            "writer": {
                "writer_lag_ns": dict(writer_stats.get("writer_lag_ns", {})),
                "frame_queue_max_depth": int(
                    writer_stats.get("frame_queue_max_depth", 0)
                ),
                "frame_queue_dropped_frames": int(
                    writer_stats.get("frame_queue_dropped_frames", 0)
                ),
                "action_queue_max_depth": int(
                    writer_stats.get("action_queue_max_depth", 0)
                ),
                "action_queue_dropped_entries": int(
                    writer_stats.get("action_queue_dropped_entries", 0)
                ),
            },
        }
        return {
            "schema_version": SCHEMA_VERSION,
            "session_timestamp": self.paths.session_timestamp,
            "created_timestamp_ns": int(self._session_created_timestamp_ns),
            "finalized_timestamp_ns": time.time_ns(),
            "frame_channel": self.vision_channel.name,
            "frame_topic": self.vision_channel.topic.decode("utf-8"),
            "action_channel": self.action_channel.name,
            "action_topic": self.action_channel.topic.decode("utf-8"),
            "video_file_name": self.paths.video_path.name,
            "video_codec": VIDEO_CODEC,
            "video_container": VIDEO_CONTAINER,
            "video_nominal_fps": float(self.session_config.nominal_fps),
            "encoded_width": int(self.session_config.width),
            "encoded_height": int(self.session_config.height),
            "frame_count": int(len(frame_payloads)),
            "action_count": int(len(action_payloads)),
            "session_integrity": {
                "status": "degraded" if integrity_degraded else "ok",
                "drop_event_count": int(len(drop_events)),
                "ignored_drop_event_count": int(len(ignored_drop_events)),
                "edge_grace_window_seconds": float(INTEGRITY_EDGE_GRACE_SECONDS),
            },
            "session_stats": {
                "frame_count": int(len(frame_payloads)),
                "action_count": int(len(action_payloads)),
                "repeat_frame_count": int(repeat_frame_count),
                "fresh_capture_frame_count": int(fresh_capture_frame_count),
                **gap_stats,
                "native_pipeline": native_pipeline_summary,
                "timeline_start_timestamp_ns": int(session_start_timestamp_ns),
                "timeline_end_timestamp_ns": int(session_end_timestamp_ns),
            },
            "performance_stats": performance_stats,
            "transport_stats": transport_stats,
            "writer_stats": writer_stats,
            "drop_events": drop_events,
            "ignored_drop_events": ignored_drop_events,
            "frames": frame_payloads,
            "actions": action_payloads,
        }

    def stop_session(self) -> None:
        if not self._session_active() or self.session_writer is None or self.paths is None:
            return

        writer = self.session_writer
        try:
            writer_stats = writer.close()
            frame_payloads = _read_jsonl(self.paths.frames_journal_path)
            action_payloads = _read_jsonl(self.paths.actions_journal_path)
            manifest = self._final_manifest_payload(
                frame_payloads=frame_payloads,
                action_payloads=action_payloads,
                writer_stats=writer_stats,
                writer_drop_events=writer.drop_events,
            )
            self.paths.metadata_path.write_text(
                json.dumps(manifest, indent=2),
                encoding="utf-8",
            )
        finally:
            for journal_path in (
                self.paths.frames_journal_path,
                self.paths.actions_journal_path,
            ):
                try:
                    journal_path.unlink(missing_ok=True)
                except Exception:
                    pass
            self.paths = None
            self.session_config = None
            self.session_writer = None
            self._written_action_keys.clear()
            self._session_transport_baselines = {}
            self._session_drop_events = []
            self._session_native_overload_active = False
            self._session_created_timestamp_ns = 0
            self._session_vision_frames_decoded_baseline = int(
                self._vision_frames_decoded_total
            )
            self._session_vision_frames_decoded_while_inactive_baseline = int(
                self._vision_frames_decoded_while_inactive
            )

    def _roll_session_if_needed(self, frame_message: ChannelMessage[np.ndarray]) -> None:
        frame_config = _frame_config_from_parts(frame_message.envelope.metadata)
        if not self._session_active():
            self.start_session(frame_config)
            return

        if self.session_config is None or _configs_match(self.session_config, frame_config):
            return

        self.stop_session()
        self.start_session(frame_config)

    def _handle_action_message(
        self,
        action_message: ChannelMessage,
        *,
        recording_enabled_for_action: bool,
    ) -> None:
        if recording_enabled_for_action and self._session_active():
            self._enqueue_action_message(action_message)
        else:
            _bounded_append(
                self.preroll_actions,
                action_message,
                capacity=_preroll_capacity(
                    self.preroll_seconds,
                    PREROLL_ACTION_BUFFER_HEADROOM,
                ),
            )
        self.latest_action_message = action_message

    def _handle_frame_message(
        self,
        frame_message: ChannelMessage[np.ndarray],
        drained_action_messages: list[ChannelMessage],
        action_cursor: int,
        *,
        frame_timestamp_ns: int,
        recording_enabled_for_frame: bool,
    ) -> int:
        frame_action_message, next_action_cursor = _aligned_message_before_with_cursor(
            frame_timestamp_ns,
            self.latest_action_message,
            drained_action_messages,
            start_index=action_cursor,
        )
        aligned_action = _aligned_action_from_message(frame_action_message)
        if recording_enabled_for_frame:
            self._roll_session_if_needed(frame_message)
            if self.session_writer is not None:
                frame_task = self._frame_task_from_message(frame_message, aligned_action)
                self._handle_native_pipeline_telemetry(frame_task.frame_envelope.metadata)
                self.session_writer.enqueue_frame(frame_task)
        else:
            if self._session_active():
                self.stop_session()
            self._buffer_preroll_frame(frame_message, aligned_action)

        self._prune_preroll(frame_timestamp_ns)
        return next_action_cursor

    def _set_ingest_mode(self, ingest_mode: str) -> None:
        if ingest_mode == self._current_ingest_mode:
            return
        if ingest_mode == INGEST_MODE_INACTIVE and self._session_active():
            self.stop_session()
        self.latest_action_message = None
        self._clear_preroll_buffers()
        if ingest_mode == INGEST_MODE_ACTIVE:
            for socket, spec, tracker in (
                (
                    self.action_socket,
                    self.action_channel,
                    self.action_transport_tracker,
                ),
                (
                    self.vision_socket,
                    self.vision_channel,
                    self.vision_transport_tracker,
                ),
            ):
                self._discard_socket_backlog(
                    socket,
                    spec=spec,
                    transport_tracker=tracker,
                )
        self._current_ingest_mode = ingest_mode

    def close(self) -> None:
        stop_error: Exception | None = None
        try:
            self.stop_session()
        except Exception as exc:
            stop_error = exc

        try:
            self.poller.unregister(self.vision_socket)
        except Exception:
            pass
        try:
            self.poller.unregister(self.action_socket)
        except Exception:
            pass
        for socket in (self.action_socket, self.vision_socket):
            try:
                socket.close()
            except Exception:
                pass
        try:
            self.context.term()
        except Exception:
            pass
        if self._owns_settings_client:
            self.settings_client.close()
        if stop_error is not None:
            raise stop_error

    def run_forever(self) -> None:
        try:
            while True:
                previous_recording_enabled = self.recording_enabled
                previous_recording_revision = self.recording_setting_revision
                recording_setting = self._refresh_runtime_settings()
                current_recording_enabled = previous_recording_enabled
                current_recording_revision = previous_recording_revision
                current_recording_updated_timestamp_ns = (
                    self.recording_setting_updated_timestamp_ns
                )
                if recording_setting is not None:
                    current_recording_enabled = bool(recording_setting.value)
                    current_recording_revision = int(recording_setting.revision)
                    current_recording_updated_timestamp_ns = int(
                        recording_setting.updated_timestamp_ns
                    )

                next_ingest_mode = self._ingest_mode_for(current_recording_enabled)
                self._set_ingest_mode(next_ingest_mode)
                def recording_enabled_at(timestamp_ns: int) -> bool:
                    return self._recording_enabled_at(
                        timestamp_ns,
                        current_recording_enabled=current_recording_enabled,
                        current_revision=current_recording_revision,
                        current_updated_timestamp_ns=current_recording_updated_timestamp_ns,
                        previous_recording_enabled=previous_recording_enabled,
                        previous_revision=previous_recording_revision,
                    )

                if self._current_ingest_mode == INGEST_MODE_INACTIVE:
                    self.recording_enabled = current_recording_enabled
                    self.recording_setting_revision = current_recording_revision
                    self.recording_setting_updated_timestamp_ns = (
                        current_recording_updated_timestamp_ns
                    )
                    time.sleep(POLL_TIMEOUT_SECONDS)
                    continue

                events = dict(self.poller.poll(POLL_TIMEOUT_MS))
                drained_action_messages: list[ChannelMessage] = []
                if events.get(self.action_socket) == zmq.POLLIN:
                    raw_action_messages = self._drain_raw_messages(
                        self.action_socket,
                        spec=self.action_channel,
                        transport_tracker=self.action_transport_tracker,
                        max_messages=ACTION_DRAIN_BATCH_LIMIT,
                    )
                    drained_action_messages = [
                        self._decode_raw_message(
                            raw_action_message,
                            spec=self.action_channel,
                        )
                        for raw_action_message in raw_action_messages
                    ]
                    for action_message in drained_action_messages:
                        self._handle_action_message(
                            action_message,
                            recording_enabled_for_action=recording_enabled_at(
                                action_message.envelope.message_timestamp_ns
                            ),
                        )

                if events.get(self.vision_socket) == zmq.POLLIN:
                    action_cursor = 0
                    drained_frame_messages = self._drain_raw_messages(
                        self.vision_socket,
                        spec=self.vision_channel,
                        transport_tracker=self.vision_transport_tracker,
                        max_messages=VISION_DRAIN_BATCH_LIMIT,
                    )
                    for raw_frame_message in drained_frame_messages:
                        frame_timestamp_ns = _frame_timestamp_ns_from_envelope(
                            raw_frame_message.envelope
                        )
                        recording_enabled_for_frame = recording_enabled_at(
                            frame_timestamp_ns
                        )
                        should_decode_frame = (
                            recording_enabled_for_frame or self.preroll_ns > 0
                        )
                        if not should_decode_frame:
                            if self._session_active():
                                self.stop_session()
                            continue
                        frame_message = self._decode_raw_message(
                            raw_frame_message,
                            spec=self.vision_channel,
                        )
                        action_cursor = self._handle_frame_message(
                            frame_message,
                            drained_action_messages,
                            action_cursor,
                            frame_timestamp_ns=frame_timestamp_ns,
                            recording_enabled_for_frame=recording_enabled_for_frame,
                        )

                self.recording_enabled = current_recording_enabled
                self.recording_setting_revision = current_recording_revision
                self.recording_setting_updated_timestamp_ns = (
                    current_recording_updated_timestamp_ns
                )
        finally:
            self.close()


def main(
    output_dir: str | Path | None = None,
    vision_channel: ChannelSpec = VISION_FRAMES_CHANNEL,
    action_channel: ChannelSpec = INPUT_ACTIONS_CHANNEL,
    settings_host: str = "127.0.0.1",
    settings_updates_port: str = SETTINGS_UPDATES_PORT,
    settings_rpc_port: str = SETTINGS_RPC_PORT,
    record_on_start: bool = BLACKBOX_RECORD_ON_START,
    preroll_seconds: float = BLACKBOX_PREROLL_SECONDS,
) -> None:
    recorder = BlackboxRecorder(
        output_dir=output_dir,
        vision_channel=vision_channel,
        action_channel=action_channel,
        settings_host=settings_host,
        settings_updates_port=settings_updates_port,
        settings_rpc_port=settings_rpc_port,
        record_on_start=record_on_start,
        preroll_seconds=preroll_seconds,
    )
    recorder.run_forever()
