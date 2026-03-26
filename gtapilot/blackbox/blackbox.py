from __future__ import annotations

import collections
import io
import json
import os
import tarfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

import cv2
import numpy as np

from gtapilot.config import BLACKBOX_PREROLL_SECONDS, BLACKBOX_RECORD_ON_START
from gtapilot.ipc.channel import ChannelSubscriber
from gtapilot.ipc.channels import (
    INPUT_ACTIONS_CHANNEL,
    VISION_FRAMES_CHANNEL,
    frame_capture_timestamp_ns,
)
from gtapilot.ipc.settings_client import SettingsClient
from gtapilot.ipc.settings_registry import (
    SETTINGS_RPC_PORT,
    SETTINGS_UPDATES_PORT,
)
from gtapilot.ipc.settings_types import SettingValue
from gtapilot.ipc.types import ChannelEnvelope, ChannelMessage, ChannelSpec

DEFAULT_OUTPUT_DIR = Path("blackbox-recordings")
SCHEMA_VERSION = 4
FLUSH_EVERY_FRAMES = 25
FLUSH_EVERY_SECONDS = 2.0
PREROLL_JPEG_QUALITY = 85
SETTINGS_REFRESH_INTERVAL_SECONDS = 0.5

T = TypeVar("T")


@dataclass(slots=True)
class BlackboxSessionPaths:
    output_dir: Path
    tar_path: Path
    metadata_path: Path
    metadata_tmp_path: Path
    session_timestamp: str


@dataclass(slots=True)
class BufferedFrame:
    frame_envelope: ChannelEnvelope
    frame_metadata: dict[str, Any]
    capture_timestamp_ns: int
    resolution_height: int
    resolution_width: int
    jpeg_bytes: bytes
    action_message: ChannelMessage | None


def _build_session_paths(output_dir: Path) -> BlackboxSessionPaths:
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_prefix = f"capture_{session_timestamp}"
    return BlackboxSessionPaths(
        output_dir=output_dir,
        tar_path=output_dir / f"{output_prefix}_frames.tar",
        metadata_path=output_dir / f"{output_prefix}_metadata.json",
        metadata_tmp_path=output_dir / f"{output_prefix}_metadata.tmp.json",
        session_timestamp=session_timestamp,
    )


def _flush_manifest(
    manifest: dict[str, Any],
    metadata_tmp_path: Path,
    metadata_path: Path,
) -> None:
    metadata_tmp_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(metadata_tmp_path, metadata_path)


def _aligned_message_before(
    timestamp_ns: int,
    latest_message: ChannelMessage[T] | None,
    drained_messages: list[ChannelMessage[T]],
) -> ChannelMessage[T] | None:
    aligned_message = None
    if (
        latest_message is not None
        and latest_message.envelope.message_timestamp_ns <= timestamp_ns
    ):
        aligned_message = latest_message

    for message in drained_messages:
        if message.envelope.message_timestamp_ns <= timestamp_ns:
            aligned_message = message

    return aligned_message


def _frame_entry_from_parts(
    *,
    frame_envelope: ChannelEnvelope,
    frame_metadata: dict[str, Any],
    resolution_height: int,
    resolution_width: int,
    frame_archive_name: str,
    action_message: ChannelMessage | None,
) -> dict[str, Any]:
    action_payload = None
    action_vector = [0.0] * 6
    action_envelope = None
    if action_message is not None:
        action_payload = action_message.payload.to_dict()
        action_vector = action_message.payload.vector
        action_envelope = action_message.envelope.to_dict()

    return {
        "frame_archive_name": frame_archive_name,
        "frame_id": int(frame_metadata.get("frame_id", -1)),
        "capture_timestamp_ns": int(
            frame_metadata.get(
                "capture_timestamp_ns",
                frame_envelope.message_timestamp_ns,
            )
        ),
        "publish_timestamp_ns": int(frame_envelope.publish_timestamp_ns),
        "received_timestamp_ns": time.time_ns(),
        "resolution_height": int(frame_metadata.get("h", resolution_height)),
        "resolution_width": int(frame_metadata.get("w", resolution_width)),
        "frame_source": frame_envelope.source,
        "frame_envelope": frame_envelope.to_dict(),
        "frame_metadata": dict(frame_metadata),
        "action": action_payload,
        "action_envelope": action_envelope,
        "action_vector": action_vector,
    }


def _frame_entry(
    frame_message: ChannelMessage,
    frame_archive_name: str,
    action_message: ChannelMessage | None,
) -> dict[str, Any]:
    return _frame_entry_from_parts(
        frame_envelope=frame_message.envelope,
        frame_metadata=dict(frame_message.envelope.metadata),
        resolution_height=int(frame_message.payload.shape[0]),
        resolution_width=int(frame_message.payload.shape[1]),
        frame_archive_name=frame_archive_name,
        action_message=action_message,
    )


def _action_stream_entry(action_message: ChannelMessage) -> dict[str, Any]:
    return {
        "envelope": action_message.envelope.to_dict(),
        "payload": action_message.payload.to_dict(),
        "action_vector": action_message.payload.vector,
    }


def _encode_bmp_from_rgb(frame_rgb: np.ndarray) -> bytes | None:
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    success, buffer = cv2.imencode(".bmp", frame_bgr)
    if not success:
        return None
    return buffer.tobytes()


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


def _decode_preroll_jpeg_to_bmp(jpeg_bytes: bytes) -> tuple[bytes | None, int, int]:
    frame_buffer = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    frame_bgr = cv2.imdecode(frame_buffer, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        return (None, 0, 0)

    success, buffer = cv2.imencode(".bmp", frame_bgr)
    if not success:
        return (None, 0, 0)

    height, width = frame_bgr.shape[:2]
    return (buffer.tobytes(), int(height), int(width))


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

        self.vision_subscriber = ChannelSubscriber(vision_channel)
        self.action_subscriber = ChannelSubscriber(action_channel)
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

        self.paths: BlackboxSessionPaths | None = None
        self.manifest: dict[str, Any] | None = None
        self.frame_count = 0
        self.action_count = 0
        self.last_flush = time.time()
        self._tar_out_file: tarfile.TarFile | None = None
        self._written_action_keys: set[tuple[str, int]] = set()
        initial_recording_setting = self._refresh_runtime_settings()
        if initial_recording_setting is not None:
            self.recording_enabled = bool(initial_recording_setting.value)
            self.recording_setting_revision = int(initial_recording_setting.revision)
            self.recording_setting_updated_timestamp_ns = int(
                initial_recording_setting.updated_timestamp_ns
            )

    def _session_active(self) -> bool:
        return self._tar_out_file is not None and self.manifest is not None

    def _refresh_runtime_settings(self) -> SettingValue | None:
        now = time.monotonic()
        if (
            now - self._last_settings_refresh_monotonic
            >= SETTINGS_REFRESH_INTERVAL_SECONDS
        ):
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

    def _prune_preroll(self, reference_timestamp_ns: int) -> None:
        if self.preroll_ns <= 0:
            self.preroll_frames.clear()
            self.preroll_actions.clear()
            return

        cutoff_timestamp_ns = reference_timestamp_ns - self.preroll_ns
        while (
            self.preroll_frames
            and self.preroll_frames[0].capture_timestamp_ns < cutoff_timestamp_ns
        ):
            self.preroll_frames.popleft()
        while (
            self.preroll_actions
            and self.preroll_actions[0].envelope.message_timestamp_ns < cutoff_timestamp_ns
        ):
            self.preroll_actions.popleft()

    def _append_action(self, action_message: ChannelMessage) -> None:
        if not self._session_active() or self.manifest is None:
            return

        action_key = (
            action_message.envelope.source,
            int(action_message.envelope.sequence_id),
        )
        if action_key in self._written_action_keys:
            return

        self._written_action_keys.add(action_key)
        self.manifest["actions"].append(_action_stream_entry(action_message))
        self.action_count += 1
        self.manifest["action_count"] = self.action_count

    def _write_frame_bytes(
        self,
        *,
        frame_bmp_bytes: bytes,
        frame_envelope: ChannelEnvelope,
        frame_metadata: dict[str, Any],
        resolution_height: int,
        resolution_width: int,
        action_message: ChannelMessage | None,
    ) -> bool:
        if not self._session_active() or self.manifest is None or self._tar_out_file is None:
            return False

        self.frame_count += 1
        frame_filename_in_tar = f"frame_{self.frame_count:06d}.bmp"
        tar_info = tarfile.TarInfo(name=frame_filename_in_tar)
        tar_info.size = len(frame_bmp_bytes)
        tar_info.mtime = int(time.time())
        self._tar_out_file.addfile(tar_info, io.BytesIO(frame_bmp_bytes))

        self.manifest["frames"].append(
            _frame_entry_from_parts(
                frame_envelope=frame_envelope,
                frame_metadata=frame_metadata,
                resolution_height=resolution_height,
                resolution_width=resolution_width,
                frame_archive_name=frame_filename_in_tar,
                action_message=action_message,
            )
        )
        self.manifest["frame_count"] = self.frame_count

        now = time.time()
        if (
            self.frame_count % FLUSH_EVERY_FRAMES == 0
            or now - self.last_flush >= FLUSH_EVERY_SECONDS
        ):
            self.flush_manifest()
            self.last_flush = now
        return True

    def _append_live_frame(
        self,
        frame_message: ChannelMessage,
        action_message: ChannelMessage | None,
    ) -> bool:
        frame_bmp_bytes = _encode_bmp_from_rgb(frame_message.payload)
        if frame_bmp_bytes is None:
            return False
        return self._write_frame_bytes(
            frame_bmp_bytes=frame_bmp_bytes,
            frame_envelope=frame_message.envelope,
            frame_metadata=dict(frame_message.envelope.metadata),
            resolution_height=int(frame_message.payload.shape[0]),
            resolution_width=int(frame_message.payload.shape[1]),
            action_message=action_message,
        )

    def _append_buffered_frame(self, buffered_frame: BufferedFrame) -> bool:
        frame_bmp_bytes, resolution_height, resolution_width = _decode_preroll_jpeg_to_bmp(
            buffered_frame.jpeg_bytes
        )
        if frame_bmp_bytes is None:
            return False

        return self._write_frame_bytes(
            frame_bmp_bytes=frame_bmp_bytes,
            frame_envelope=buffered_frame.frame_envelope,
            frame_metadata=buffered_frame.frame_metadata,
            resolution_height=resolution_height,
            resolution_width=resolution_width,
            action_message=buffered_frame.action_message,
        )

    def _buffer_preroll_frame(
        self,
        frame_message: ChannelMessage,
        action_message: ChannelMessage | None,
    ) -> None:
        if self.preroll_ns <= 0:
            return

        jpeg_bytes = _encode_preroll_jpeg(frame_message.payload)
        if jpeg_bytes is None:
            return

        self.preroll_frames.append(
            BufferedFrame(
                frame_envelope=frame_message.envelope,
                frame_metadata=dict(frame_message.envelope.metadata),
                capture_timestamp_ns=frame_capture_timestamp_ns(frame_message),
                resolution_height=int(frame_message.payload.shape[0]),
                resolution_width=int(frame_message.payload.shape[1]),
                jpeg_bytes=jpeg_bytes,
                action_message=action_message,
            )
        )

    def start_session(self) -> None:
        if self._session_active():
            return

        self.paths = _build_session_paths(self.output_dir)
        self.manifest = {
            "schema_version": SCHEMA_VERSION,
            "session_timestamp": self.paths.session_timestamp,
            "created_timestamp_ns": time.time_ns(),
            "frame_channel": self.vision_subscriber.spec.name,
            "frame_topic": self.vision_subscriber.spec.topic.decode("utf-8"),
            "action_channel": self.action_subscriber.spec.name,
            "action_topic": self.action_subscriber.spec.topic.decode("utf-8"),
            "frames": [],
            "actions": [],
        }
        self.frame_count = 0
        self.action_count = 0
        self.last_flush = time.time()
        self._written_action_keys.clear()
        self._tar_out_file = tarfile.open(self.paths.tar_path, "w")

        for action_message in self.preroll_actions:
            self._append_action(action_message)
        for buffered_frame in self.preroll_frames:
            self._append_buffered_frame(buffered_frame)

        self.preroll_actions.clear()
        self.preroll_frames.clear()

    def stop_session(self) -> None:
        if not self._session_active():
            return

        try:
            self.flush_manifest()
        except Exception:
            pass
        try:
            if self._tar_out_file is not None:
                self._tar_out_file.close()
        except Exception:
            pass

        self.paths = None
        self.manifest = None
        self._tar_out_file = None
        self.frame_count = 0
        self.action_count = 0
        self._written_action_keys.clear()

    def flush_manifest(self) -> None:
        if self.manifest is None or self.paths is None:
            return
        _flush_manifest(
            self.manifest,
            self.paths.metadata_tmp_path,
            self.paths.metadata_path,
        )

    def record_next_frame(self, timeout_sec: float | None = None) -> bool:
        frame_message = self.vision_subscriber.receive(
            blocking=True,
            timeout_sec=timeout_sec,
        )
        if frame_message is None:
            return False

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

        drained_action_messages = self.action_subscriber.drain()
        frame_timestamp_ns = frame_capture_timestamp_ns(frame_message)
        frame_action_message = _aligned_message_before(
            frame_timestamp_ns,
            self.latest_action_message,
            drained_action_messages,
        )
        recording_enabled_for_frame = self._recording_enabled_at(
            frame_timestamp_ns,
            current_recording_enabled=current_recording_enabled,
            current_revision=current_recording_revision,
            current_updated_timestamp_ns=current_recording_updated_timestamp_ns,
            previous_recording_enabled=previous_recording_enabled,
            previous_revision=previous_recording_revision,
        )

        if recording_enabled_for_frame and not self._session_active():
            self.start_session()

        for action_message in drained_action_messages:
            recording_enabled_for_action = self._recording_enabled_at(
                action_message.envelope.message_timestamp_ns,
                current_recording_enabled=current_recording_enabled,
                current_revision=current_recording_revision,
                current_updated_timestamp_ns=current_recording_updated_timestamp_ns,
                previous_recording_enabled=previous_recording_enabled,
                previous_revision=previous_recording_revision,
            )
            if recording_enabled_for_action and self._session_active():
                self._append_action(action_message)
            else:
                self.preroll_actions.append(action_message)
            self.latest_action_message = action_message

        if recording_enabled_for_frame:
            self._append_live_frame(frame_message, frame_action_message)
        else:
            if self._session_active():
                self.stop_session()
            self._buffer_preroll_frame(frame_message, frame_action_message)

        self._prune_preroll(frame_timestamp_ns)

        self.recording_enabled = current_recording_enabled
        self.recording_setting_revision = current_recording_revision
        self.recording_setting_updated_timestamp_ns = (
            current_recording_updated_timestamp_ns
        )
        return True

    def close(self) -> None:
        self.stop_session()
        if self._owns_settings_client:
            self.settings_client.close()
        self.action_subscriber.close()
        self.vision_subscriber.close()

    def run_forever(self) -> None:
        try:
            while True:
                self.record_next_frame()
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
