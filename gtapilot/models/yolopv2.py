import time

import cv2

from gtapilot.ipc.vision_ipc import VisionIPCSubscriber


def main():
    visionIPCSubscriber = VisionIPCSubscriber()

    try:
        while True:
            frame = visionIPCSubscriber.receive_frame(blocking=True)

            if frame is not None:
                # Process frame with YOLOPv2
                continue

            else:
                print("No frame received")

    finally:
        visionIPCSubscriber.close()
