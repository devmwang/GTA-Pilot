import time

import bettercam
import cv2

from gtapilot.ipc.channel import ChannelPublisher
from gtapilot.ipc.channels import VISION_FRAMES_CHANNEL

TARGET_FPS = 20
TARGET_FRAME_TIME = 1 / TARGET_FPS
TARGET_SIZE = (1920, 1080)


def main(display=0):
    publisher = ChannelPublisher(
        VISION_FRAMES_CHANNEL,
        source_name="bettercam_capture",
    )
    camera = bettercam.create(output_idx=display, output_color="RGB")
    camera.start(target_fps=TARGET_FPS)
    frame_id = 1

    try:
        while True:
            frame = camera.get_latest_frame()
            if frame is not None:
                if frame.shape[1] != TARGET_SIZE[0] or frame.shape[0] != TARGET_SIZE[1]:
                    frame = cv2.resize(frame, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
                capture_timestamp_ns = time.time_ns()
                publisher.publish(
                    frame,
                    timestamp_ns=capture_timestamp_ns,
                    metadata={
                        "frame_id": frame_id,
                        "capture_timestamp_ns": capture_timestamp_ns,
                        "is_repeat": False,
                    },
                )
                frame_id += 1
            # Sleep just enough to manage CPU if camera internal loop is fast
            time.sleep(0.001)
    finally:
        try:
            camera.stop()
        except Exception:
            pass
        publisher.close()
