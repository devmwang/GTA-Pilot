"""Display Override process.

Reads frames from a provided video file and publishes them through the generic
`vision.frames` channel at a fixed 60 Hz output cadence.
"""

from __future__ import annotations

import math
import time

import cv2
import numpy as np

from gtapilot.ipc.channel import ChannelPublisher
from gtapilot.ipc.channels import VISION_FRAMES_CHANNEL
from gtapilot.timing import HighResolutionTimer, advance_fixed_deadline, sleep_until

OUTPUT_FPS = 60.0
OUTPUT_FRAME_INTERVAL = 1.0 / OUTPUT_FPS
DEFAULT_SOURCE_FPS = 30.0
FPS_SANITY_LIMIT = 1000.0
OUTPUT_SIZE = (1920, 1080)


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

    def next_output_frame(self) -> tuple[np.ndarray, bool]:
        source_index, frame_rgb = self._frame_for_current_cursor()
        is_repeat = self._last_output_source_index == source_index
        self._last_output_source_index = source_index
        self._source_cursor += self._source_step
        return frame_rgb, is_repeat

    def close(self) -> None:
        self.cap.release()


def main(video_path: str):
    publisher = ChannelPublisher(
        VISION_FRAMES_CHANNEL,
        source_name="display_override",
    )
    video_source = LoopingVideoSource(video_path)
    frame_id = 1
    print(
        "DisplayOverride: streaming "
        f"'{video_path}' at fixed {OUTPUT_FPS:.2f} Hz from source ~{video_source.source_fps:.2f} FPS"
    )

    next_frame_time = time.perf_counter()
    try:
        with HighResolutionTimer(1):
            while True:
                sleep_until(next_frame_time)
                frame_rgb, is_repeat = video_source.next_output_frame()
                capture_timestamp_ns = time.time_ns()
                publisher.publish(
                    frame_rgb,
                    timestamp_ns=capture_timestamp_ns,
                    metadata={
                        "frame_id": frame_id,
                        "capture_timestamp_ns": capture_timestamp_ns,
                        "nominal_fps": OUTPUT_FPS,
                        "is_repeat": is_repeat,
                    },
                )
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
        print("DisplayOverride: shutdown complete.")
