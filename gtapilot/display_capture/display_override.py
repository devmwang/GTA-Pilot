"""Display Override process.

Reads frames from a provided video file and publishes them through the generic
`vision.frames` channel at a fixed 60 Hz output cadence.
"""

from __future__ import annotations

import math
import os
import time

import cv2
import numpy as np

from gtapilot.ipc.channel import ChannelPublisher
from gtapilot.ipc.channels import VISION_FRAMES_CHANNEL, VISION_PREVIEW_CHANNEL
from gtapilot.timing import HighResolutionTimer, advance_fixed_deadline, sleep_until

OUTPUT_FPS = 60.0
OUTPUT_FRAME_INTERVAL = 1.0 / OUTPUT_FPS
DEFAULT_SOURCE_FPS = 30.0
FPS_SANITY_LIMIT = 1000.0
OUTPUT_SIZE = (1920, 1080)
PREVIEW_SIZE = (1280, 720)
FRAME_NOMINAL_FPS = OUTPUT_FPS
DEFAULT_PREVIEW_MAX_FPS = 30.0


def _preview_max_fps() -> float:
    raw_value = os.environ.get("GTAPILOT_PREVIEW_MAX_FPS", "").strip()
    if not raw_value:
        return DEFAULT_PREVIEW_MAX_FPS
    try:
        preview_fps = float(raw_value)
    except ValueError:
        return DEFAULT_PREVIEW_MAX_FPS
    if preview_fps <= 0.0:
        return 0.0
    return min(preview_fps, FRAME_NOMINAL_FPS)


def _normalize_source_fps(value: float) -> float:
    if not value or value <= 0.0 or value > FPS_SANITY_LIMIT:
        return DEFAULT_SOURCE_FPS
    return float(value)


class LoopingVideoSource:
    def __init__(self, video_path: str):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Unable to open video file: {video_path}")

        self.source_fps = _normalize_source_fps(self.cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.total_frames: int | None = total_frames if total_frames > 0 else None
        self._source_cursor = 0.0
        self._source_step = self.source_fps / OUTPUT_FPS
        self._decoded_frame_index = -1
        self._last_output_source_index: int | None = None
        self._last_frame_rgb: np.ndarray | None = None
        self._last_capture_frame_id = 0
        self._last_capture_timestamp_ns = 0

    def _reset_capture(self) -> None:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self._decoded_frame_index = -1
        self._last_frame_rgb = None

    def _wrap_source_cursor_if_needed(self) -> None:
        if self.total_frames is None or self._source_cursor < float(self.total_frames):
            return
        self._source_cursor = math.fmod(self._source_cursor, float(self.total_frames))
        self._reset_capture()

    def _restart_from_beginning(self) -> None:
        if self.total_frames is None:
            estimated_total_frames = self._decoded_frame_index + 1
            if estimated_total_frames <= 0:
                raise RuntimeError("Video override source ended before a frame was decoded.")
            self.total_frames = estimated_total_frames
        self._source_cursor = math.fmod(self._source_cursor, float(self.total_frames))
        self._reset_capture()

    def _read_next_source_frame(self) -> bool:
        ret, frame_bgr = self.cap.read()
        if not ret:
            return False

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if frame_rgb.shape[1] != OUTPUT_SIZE[0] or frame_rgb.shape[0] != OUTPUT_SIZE[1]:
            frame_rgb = cv2.resize(
                frame_rgb,
                OUTPUT_SIZE,
                interpolation=cv2.INTER_LINEAR,
            )

        self._decoded_frame_index += 1
        self._last_frame_rgb = frame_rgb
        return True

    def _frame_for_current_cursor(self) -> tuple[int, np.ndarray]:
        while True:
            self._wrap_source_cursor_if_needed()
            target_source_index = int(math.floor(self._source_cursor + 1e-9))

            while self._decoded_frame_index < target_source_index:
                if self._read_next_source_frame():
                    continue
                self._restart_from_beginning()
                break
            else:
                if self._last_frame_rgb is None:
                    raise RuntimeError("Video override could not decode a frame.")
                return target_source_index, self._last_frame_rgb

    def next_output_frame(self) -> tuple[np.ndarray, bool, int, int]:
        source_index, frame_rgb = self._frame_for_current_cursor()
        is_repeat = self._last_output_source_index == source_index
        if not is_repeat:
            self._last_capture_frame_id += 1
            self._last_capture_timestamp_ns = time.time_ns()
        self._last_output_source_index = source_index
        self._source_cursor += self._source_step
        return (
            frame_rgb,
            is_repeat,
            self._last_capture_frame_id,
            self._last_capture_timestamp_ns,
        )

    def close(self) -> None:
        self.cap.release()


def main(video_path: str):
    publisher = ChannelPublisher(
        VISION_FRAMES_CHANNEL,
        source_name="display_override",
    )
    preview_publisher = ChannelPublisher(
        VISION_PREVIEW_CHANNEL,
        source_name="display_override_preview",
    )
    video_source = LoopingVideoSource(video_path)
    frame_id = 1
    print(
        "DisplayOverride: streaming "
        f"'{video_path}' at fixed {OUTPUT_FPS:.2f} Hz from source ~{video_source.source_fps:.2f} FPS"
    )

    next_frame_time = time.perf_counter()
    preview_max_fps = _preview_max_fps()
    preview_interval_ns = (
        int(round(1_000_000_000.0 / preview_max_fps))
        if preview_max_fps > 0.0
        else None
    )
    last_preview_capture_timestamp_ns = 0
    try:
        with HighResolutionTimer(1):
            while True:
                sleep_until(next_frame_time)
                (
                    frame_rgb,
                    is_repeat,
                    capture_frame_id,
                    capture_timestamp_ns,
                ) = video_source.next_output_frame()
                publisher.publish(
                    frame_rgb,
                    timestamp_ns=capture_timestamp_ns,
                    metadata={
                        "frame_id": frame_id,
                        "capture_frame_id": capture_frame_id,
                        "capture_timestamp_ns": capture_timestamp_ns,
                        "nominal_fps": FRAME_NOMINAL_FPS,
                        "is_repeat": is_repeat,
                        "capture_mode": "video_override",
                    },
                )
                if preview_interval_ns is not None and (
                    last_preview_capture_timestamp_ns == 0
                    or capture_timestamp_ns
                    >= last_preview_capture_timestamp_ns + preview_interval_ns
                ):
                    preview_frame_rgb = cv2.resize(
                        frame_rgb,
                        PREVIEW_SIZE,
                        interpolation=cv2.INTER_LINEAR,
                    )
                    preview_publisher.publish(
                        preview_frame_rgb,
                        timestamp_ns=capture_timestamp_ns,
                        metadata={
                            "frame_id": frame_id,
                            "capture_frame_id": capture_frame_id,
                            "capture_timestamp_ns": capture_timestamp_ns,
                            "nominal_fps": preview_max_fps,
                            "source_nominal_fps": FRAME_NOMINAL_FPS,
                            "is_repeat": is_repeat,
                            "capture_mode": "video_override",
                            "preview_source": "vision.frames",
                        },
                    )
                    last_preview_capture_timestamp_ns = capture_timestamp_ns
                frame_id += 1
                next_frame_time = advance_fixed_deadline(
                    next_frame_time,
                    OUTPUT_FRAME_INTERVAL,
                )
    finally:
        try:
            video_source.close()
        except Exception:
            pass
        publisher.close()
        preview_publisher.close()
        print("DisplayOverride: shutdown complete.")
