from __future__ import annotations

import io
import json
import os
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

from gtapilot.ipc.channel import ChannelSubscriber
from gtapilot.ipc.channels import INPUT_ACTIONS_CHANNEL, VISION_FRAMES_CHANNEL
from gtapilot.ipc.types import ChannelMessage, ChannelSpec

DEFAULT_OUTPUT_DIR = Path("blackbox-recordings")
SCHEMA_VERSION = 3
FLUSH_EVERY_FRAMES = 25
FLUSH_EVERY_SECONDS = 2.0


@dataclass(slots=True)
class BlackboxSessionPaths:
    output_dir: Path
    tar_path: Path
    metadata_path: Path
    metadata_tmp_path: Path
    session_timestamp: str


def _build_session_paths(output_dir: Path) -> BlackboxSessionPaths:
    session_timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_prefix = f"capture_{session_timestamp}"
    return BlackboxSessionPaths(
        output_dir=output_dir,
        tar_path=output_dir / f"{output_prefix}_frames.tar",
        metadata_path=output_dir / f"{output_prefix}_metadata.json",
        metadata_tmp_path=output_dir / f"{output_prefix}_metadata.tmp.json",
        session_timestamp=session_timestamp,
    )


def _flush_manifest(manifest: dict[str, Any], metadata_tmp_path: Path, metadata_path: Path) -> None:
    metadata_tmp_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(metadata_tmp_path, metadata_path)


def _frame_entry(
    frame_message: ChannelMessage,
    frame_archive_name: str,
    action_message: ChannelMessage | None,
) -> dict[str, Any]:
    frame_metadata = dict(frame_message.envelope.metadata)
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
                "capture_timestamp_ns", frame_message.envelope.message_timestamp_ns
            )
        ),
        "publish_timestamp_ns": int(frame_message.envelope.publish_timestamp_ns),
        "received_timestamp_ns": time.time_ns(),
        "resolution_height": int(frame_metadata.get("h", frame_message.payload.shape[0])),
        "resolution_width": int(frame_metadata.get("w", frame_message.payload.shape[1])),
        "frame_source": frame_message.envelope.source,
        "frame_envelope": frame_message.envelope.to_dict(),
        "frame_metadata": frame_metadata,
        "action": action_payload,
        "action_envelope": action_envelope,
        "action_vector": action_vector,
    }


def _action_stream_entry(action_message: ChannelMessage) -> dict[str, Any]:
    return {
        "envelope": action_message.envelope.to_dict(),
        "payload": action_message.payload.to_dict(),
        "action_vector": action_message.payload.vector,
    }


class BlackboxRecorder:
    def __init__(
        self,
        *,
        output_dir: str | Path | None = None,
        vision_channel: ChannelSpec = VISION_FRAMES_CHANNEL,
        action_channel: ChannelSpec = INPUT_ACTIONS_CHANNEL,
    ):
        self.paths = _build_session_paths(Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR)
        self.paths.output_dir.mkdir(parents=True, exist_ok=True)

        self.vision_subscriber = ChannelSubscriber(vision_channel)
        self.action_subscriber = ChannelSubscriber(action_channel)
        self.manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "session_timestamp": self.paths.session_timestamp,
            "created_timestamp_ns": time.time_ns(),
            "frame_channel": vision_channel.name,
            "frame_topic": vision_channel.topic.decode("utf-8"),
            "action_channel": action_channel.name,
            "action_topic": action_channel.topic.decode("utf-8"),
            "frames": [],
            "actions": [],
        }
        self.frame_count = 0
        self.action_count = 0
        self.last_flush = time.time()
        self.latest_action_message: ChannelMessage | None = None
        self._tar_out_file = tarfile.open(self.paths.tar_path, "w")

    def record_next_frame(self, timeout_sec: float | None = None) -> bool:
        frame_message = self.vision_subscriber.receive(
            blocking=True,
            timeout_sec=timeout_sec,
        )
        if frame_message is None:
            return False

        frame_capture_timestamp_ns = int(
            frame_message.envelope.metadata.get(
                "capture_timestamp_ns", frame_message.envelope.message_timestamp_ns
            )
        )
        frame_action_message = self.latest_action_message
        for action_message in self.action_subscriber.drain():
            self.manifest["actions"].append(_action_stream_entry(action_message))
            self.action_count += 1
            self.manifest["action_count"] = self.action_count
            if action_message.envelope.message_timestamp_ns <= frame_capture_timestamp_ns:
                frame_action_message = action_message
            self.latest_action_message = action_message

        self.frame_count += 1
        frame_bgr = cv2.cvtColor(frame_message.payload, cv2.COLOR_RGB2BGR)
        success, buffer = cv2.imencode(".bmp", frame_bgr)
        if not success:
            return False

        bmp_bytes = buffer.tobytes()
        frame_filename_in_tar = f"frame_{self.frame_count:06d}.bmp"
        tar_info = tarfile.TarInfo(name=frame_filename_in_tar)
        tar_info.size = len(bmp_bytes)
        tar_info.mtime = int(time.time())
        self._tar_out_file.addfile(tar_info, io.BytesIO(bmp_bytes))

        self.manifest["frames"].append(
            _frame_entry(frame_message, frame_filename_in_tar, frame_action_message)
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

    def flush_manifest(self) -> None:
        _flush_manifest(
            self.manifest,
            self.paths.metadata_tmp_path,
            self.paths.metadata_path,
        )

    def close(self) -> None:
        try:
            self.flush_manifest()
        except Exception:
            pass
        try:
            self._tar_out_file.close()
        except Exception:
            pass
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
) -> None:
    recorder = BlackboxRecorder(
        output_dir=output_dir,
        vision_channel=vision_channel,
        action_channel=action_channel,
    )
    recorder.run_forever()
