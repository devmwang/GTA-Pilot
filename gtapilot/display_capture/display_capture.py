import time

import bettercam

from gtapilot.ipc.vision_ipc import VisionIpcPublisher

TARGET_FPS = 20
TARGET_FRAME_TIME = 1 / TARGET_FPS


def main(display=0):
    visionIPCPublisher = VisionIpcPublisher()
    camera = bettercam.create(output_idx=display, output_color="RGB")
    camera.start(target_fps=TARGET_FPS)

    try:
        while True:
            frame = camera.get_latest_frame()
            if frame is not None:
                visionIPCPublisher.publish_frame(frame)
            # Sleep just enough to manage CPU if camera internal loop is fast
            time.sleep(0.001)
    finally:
        try:
            camera.stop()
        except Exception:
            pass
        visionIPCPublisher.close()
