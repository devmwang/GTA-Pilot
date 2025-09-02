import time

import cv2

from gtapilot.ipc.vision_ipc import VisionIPCSubscriber


def main():
    visionIPCSubscriber = VisionIPCSubscriber()

    fps = 0
    frame_count = 0
    fps_start_time = time.time()

    try:
        while True:
            frame = visionIPCSubscriber.receive_frame(blocking=True)

            if frame is not None:
                # Process the frame (e.g., display it, save it, etc.)
                frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_LINEAR)
                frame = cv2.cvtColor(
                    frame, cv2.COLOR_RGB2BGR
                )  # Convert RGB to BGR for OpenCV

                # Calculate and update FPS
                frame_count += 1
                if time.time() - fps_start_time >= 1.0:  # Update FPS every second
                    fps = frame_count / (time.time() - fps_start_time)
                    frame_count = 0
                    fps_start_time = time.time()

                # Display FPS on frame
                cv2.putText(
                    frame,
                    f"FPS: {fps:.2f}",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                cv2.imshow("GTA Pilot Visualization", frame)

            else:
                print("No frame received")

            key = cv2.waitKey(1)
            if key & 0xFF == ord("q"):
                break
            if key == 27:  # ESC pressed inside window
                break
            
    finally:
        visionIPCSubscriber.close()
        cv2.destroyAllWindows()
