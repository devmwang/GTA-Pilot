import time

import cv2

from gtapilot.ipc.vision_ipc import VisionIPCSubscriber
from gtapilot.ipc.messaging import SubMaster


def main():
    visionIPCSubscriber = VisionIPCSubscriber()
    sub = SubMaster()
    # Subscribe to topics produced by YOLOPv2 model process
    sub.add_topic("lane_lines")
    sub.add_topic("driveable_area")
    sub.add_topic("vehicle_bounding_boxes")

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

                # Overlay drivable area and lane masks if available
                try:
                    msg_da = sub.topics.get("driveable_area")
                    if (
                        msg_da
                        and isinstance(msg_da.data, dict)
                        and "mask" in msg_da.data
                    ):
                        da_mask = msg_da.data["mask"]
                        if da_mask is not None:
                            if da_mask.shape[:2] != (frame.shape[0], frame.shape[1]):
                                da_mask = cv2.resize(
                                    da_mask,
                                    (frame.shape[1], frame.shape[0]),
                                    interpolation=cv2.INTER_NEAREST,
                                )
                            da_col = frame.copy()
                            da_col[:, :, 1] = cv2.max(da_col[:, :, 1], da_mask)
                            frame = cv2.addWeighted(frame, 0.7, da_col, 0.3, 0)

                    msg_ll = sub.topics.get("lane_lines")
                    if (
                        msg_ll
                        and isinstance(msg_ll.data, dict)
                        and "mask" in msg_ll.data
                    ):
                        ll_mask = msg_ll.data["mask"]
                        if ll_mask is not None:
                            if ll_mask.shape[:2] != (frame.shape[0], frame.shape[1]):
                                ll_mask = cv2.resize(
                                    ll_mask,
                                    (frame.shape[1], frame.shape[0]),
                                    interpolation=cv2.INTER_NEAREST,
                                )
                            ll_col = frame.copy()
                            ll_col[:, :, 2] = cv2.max(ll_col[:, :, 2], ll_mask)
                            frame = cv2.addWeighted(frame, 0.7, ll_col, 0.3, 0)

                    msg_boxes = sub.topics.get("vehicle_bounding_boxes")
                    if (
                        msg_boxes
                        and isinstance(msg_boxes.data, dict)
                        and "boxes" in msg_boxes.data
                    ):
                        for b in msg_boxes.data["boxes"] or []:
                            try:
                                x1 = int(b.get("x1", 0))
                                y1 = int(b.get("y1", 0))
                                x2 = int(b.get("x2", 0))
                                y2 = int(b.get("y2", 0))
                                conf = float(b.get("conf", 0.0))
                                cls_id = int(b.get("cls_id", 0))
                            except Exception:
                                continue
                            if x2 > x1 and y2 > y1:
                                cv2.rectangle(
                                    frame, (x1, y1), (x2, y2), (255, 255, 0), 2
                                )
                                cv2.putText(
                                    frame,
                                    f"{cls_id}:{conf:.2f}",
                                    (x1, max(0, y1 - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (255, 255, 0),
                                    1,
                                    cv2.LINE_AA,
                                )
                except Exception:
                    # Keep visualization robust to messaging deserialization or shape issues
                    pass

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
        try:
            sub.close()
        except Exception:
            pass
        visionIPCSubscriber.close()
        cv2.destroyAllWindows()
