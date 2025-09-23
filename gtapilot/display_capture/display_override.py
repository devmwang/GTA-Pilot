import time

import cv2

from gtapilot.ipc.vision_ipc import VisionIpcPublisher

"""Display Override process
Reads frames from a provided video file and publishes them through Vision IPC
at approximately the video's native FPS.
"""


def main(video_path: str):
    publisher = VisionIpcPublisher()

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
            publisher.publish_frame(frame_rgb)

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
