from __future__ import annotations

import time

import cv2

from gtapilot.ipc.channel import ChannelSubscriber
from gtapilot.ipc.channels import INPUT_ACTIONS_CHANNEL, VISION_FRAMES_CHANNEL


def _draw_text(frame, text: str, y: int, color=(0, 255, 0)):
    cv2.putText(
        frame,
        text,
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA,
    )


def main():
    vision_subscriber = ChannelSubscriber(VISION_FRAMES_CHANNEL, latest_only=True)
    action_subscriber = ChannelSubscriber(INPUT_ACTIONS_CHANNEL, latest_only=True)

    fps = 0.0
    frame_count = 0
    fps_start_time = time.time()

    try:
        while True:
            packet = vision_subscriber.receive(blocking=True)
            if packet is None:
                continue

            frame = packet.payload
            if frame.shape[0] != 1080 or frame.shape[1] != 1920:
                frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_LINEAR)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            frame_count += 1
            now = time.time()
            if now - fps_start_time >= 1.0:
                fps = frame_count / (now - fps_start_time)
                frame_count = 0
                fps_start_time = now

            action_message = action_subscriber.get_latest(
                before_timestamp_ns=int(
                    packet.envelope.metadata.get(
                        "capture_timestamp_ns",
                        packet.envelope.message_timestamp_ns,
                    )
                )
            )
            action = None if action_message is None else action_message.payload

            _draw_text(frame, f"FPS: {fps:.2f}", 40)
            _draw_text(
                frame,
                f"Frame {packet.envelope.metadata.get('frame_id', '?')} "
                f"source={packet.envelope.source}",
                80,
            )

            if action is not None:
                _draw_text(
                    frame,
                    "Action "
                    f"steer={action.steer:+.1f} throttle={action.throttle:.1f} "
                    f"brake={action.brake:.1f} handbrake={action.handbrake:.1f} "
                    f"reverse={action.reverse:.1f}",
                    120,
                    color=(255, 255, 0),
                )
                pilot_mode = "POLICY" if action.pilot_active >= 0.5 else "MANUAL"
                _draw_text(
                    frame,
                    f"Pilot: {pilot_mode} action_source={action_message.envelope.source}",
                    160,
                    color=(255, 255, 0),
                )

            cv2.imshow("GTA Pilot Visualization", frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord("q") or key == 27:
                break
    finally:
        try:
            action_subscriber.close()
        except Exception:
            pass
        vision_subscriber.close()
        cv2.destroyAllWindows()
