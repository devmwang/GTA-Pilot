import time

import cv2

from gtapilot.ipc.channel import ChannelPublisher
from gtapilot.ipc.channels import VISION_FRAMES_CHANNEL

"""Display Override process
Reads frames from a provided video file and publishes them through the
generic `vision.frames` channel
at approximately the video's native FPS.
"""


def main(video_path: str):
    publisher = ChannelPublisher(
        VISION_FRAMES_CHANNEL,
        source_name="display_override",
    )
    frame_id = 1

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video file: {video_path}")

    # Attempt to read FPS from container metadata
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or fps > 1000:  # sanity check
        fps = 30.0  # fallback
    frame_interval = 1.0 / fps
    print(f"DisplayOverride: streaming '{video_path}' at ~{fps:.2f} FPS")

    try:
        next_frame_time = time.perf_counter()
        while True:

            # Maintain timing based on captured video FPS with drift correction
            now = time.perf_counter()
            if now < next_frame_time:
                # Sleep only the remaining time slice to keep schedule
                time.sleep(max(0.0, next_frame_time - now))

            ret, frame_bgr = cap.read()
            if not ret:
                # Loop video from start (or break if you prefer one-shot)
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # Convert BGR (OpenCV) to RGB to match live capture output_color="RGB"
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            if frame_rgb.shape[1] != 1920 or frame_rgb.shape[0] != 1080:
                frame_rgb = cv2.resize(
                    frame_rgb,
                    (1920, 1080),
                    interpolation=cv2.INTER_LINEAR,
                )
            capture_timestamp_ns = time.time_ns()
            publisher.publish(
                frame_rgb,
                timestamp_ns=capture_timestamp_ns,
                metadata={
                    "frame_id": frame_id,
                    "capture_timestamp_ns": capture_timestamp_ns,
                    "nominal_fps": float(fps),
                    "is_repeat": False,
                },
            )
            frame_id += 1

            # Schedule next deadline; if we fell behind by >1 frame, reset to avoid drift
            next_frame_time += frame_interval
            now = time.perf_counter()
            if now - next_frame_time > frame_interval:
                next_frame_time = now + frame_interval
    finally:
        try:
            cap.release()
        except Exception:
            pass
        publisher.close()
        print("DisplayOverride: shutdown complete.")
