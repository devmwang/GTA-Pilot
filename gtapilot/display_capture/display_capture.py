import bettercam

from gtapilot.ipc.vision_ipc import VisionIPCPublisher


TARGET_FPS = 20
TARGET_FRAME_TIME = 1 / TARGET_FPS


def main(display=0):
    visionIPCPublisher = VisionIPCPublisher()
    camera = bettercam.create(output_idx=display, output_color="RGB")
    camera.start(target_fps=TARGET_FPS)

    while True:
        frame = camera.get_latest_frame()
        visionIPCPublisher.publish_frame(frame)
