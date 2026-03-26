from __future__ import annotations

import ctypes
import time
from dataclasses import dataclass

import cv2

from gtapilot.ipc.channel import ChannelSubscriber
from gtapilot.ipc.channels import (
    INPUT_ACTIONS_CHANNEL,
    VISION_FRAMES_CHANNEL,
    frame_capture_timestamp_ns,
)
from gtapilot.ipc.settings_client import SettingsClient
from gtapilot.ipc.settings_registry import (
    SETTINGS_RPC_PORT,
    SETTINGS_UPDATES_PORT,
)

SETTINGS_REFRESH_INTERVAL_SECONDS = 0.5
WINDOW_NAME = "GTA Pilot Visualization"


@dataclass(slots=True, frozen=True)
class MonitorBounds:
    left: int
    top: int
    right: int
    bottom: int
    is_primary: bool


def _enumerate_monitors() -> list[MonitorBounds]:
    if not hasattr(ctypes, "windll"):  # pragma: no cover - Windows-only path
        return []

    user32 = ctypes.windll.user32

    class RECT(ctypes.Structure):
        _fields_ = [
            ("left", ctypes.c_long),
            ("top", ctypes.c_long),
            ("right", ctypes.c_long),
            ("bottom", ctypes.c_long),
        ]

    class MONITORINFO(ctypes.Structure):
        _fields_ = [
            ("cbSize", ctypes.c_ulong),
            ("rcMonitor", RECT),
            ("rcWork", RECT),
            ("dwFlags", ctypes.c_ulong),
        ]

    monitors: list[MonitorBounds] = []
    monitor_enum_proc = ctypes.WINFUNCTYPE(
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(RECT),
        ctypes.c_long,
    )

    def _callback(hmonitor, _hdc, _rect, _lparam):
        monitor_info = MONITORINFO()
        monitor_info.cbSize = ctypes.sizeof(MONITORINFO)
        if not user32.GetMonitorInfoW(hmonitor, ctypes.byref(monitor_info)):
            return 1
        monitors.append(
            MonitorBounds(
                left=int(monitor_info.rcWork.left),
                top=int(monitor_info.rcWork.top),
                right=int(monitor_info.rcWork.right),
                bottom=int(monitor_info.rcWork.bottom),
                is_primary=bool(monitor_info.dwFlags & 1),
            )
        )
        return 1

    user32.EnumDisplayMonitors(None, None, monitor_enum_proc(_callback), 0)
    return monitors


def _pick_visualization_monitor(
    monitors: list[MonitorBounds],
    capture_display_id: int | None,
) -> MonitorBounds | None:
    if len(monitors) <= 1:
        return None
    if capture_display_id is not None and 0 <= capture_display_id < len(monitors):
        for index, monitor in enumerate(monitors):
            if index != capture_display_id:
                return monitor
    for monitor in monitors:
        if not monitor.is_primary:
            return monitor
    return monitors[1]


def _position_visualization_window(capture_display_id: int | None) -> None:
    target_monitor = _pick_visualization_monitor(
        _enumerate_monitors(),
        capture_display_id,
    )
    if target_monitor is None:
        return
    cv2.moveWindow(
        WINDOW_NAME,
        int(target_monitor.left + 40),
        int(target_monitor.top + 40),
    )


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


def _capture_overlay(
    blackbox_enabled: bool,
    recording_enabled: bool,
    record_hotkey: str,
) -> tuple[str, tuple[int, int, int]]:
    if not blackbox_enabled:
        return ("Blackbox: DISABLED", (128, 128, 128))
    if recording_enabled:
        return (f"Blackbox: REC hotkey={record_hotkey}", (0, 0, 255))
    return (f"Blackbox: IDLE hotkey={record_hotkey}", (0, 128, 255))


def _input_device_overlay(action) -> str:
    return f"Input Device: {action.active_device}"


def _controller_summary_overlay(action) -> str | None:
    controller_inputs = dict(action.device_inputs.get("xinput_controller", {}))
    if not bool(controller_inputs.get("connected", False)):
        return None

    sticks = dict(controller_inputs.get("sticks", {}))
    triggers = dict(controller_inputs.get("triggers", {}))
    buttons = dict(controller_inputs.get("buttons", {}))
    return (
        "Controller "
        f"LX={float(sticks.get('left_x', 0.0)):+.2f} "
        f"RT={float(triggers.get('right', 0.0)):.2f} "
        f"LT={float(triggers.get('left', 0.0)):.2f} "
        f"RB={1 if bool(buttons.get('rb', False)) else 0}"
    )


def main(
    settings_host: str = "127.0.0.1",
    settings_updates_port: str = SETTINGS_UPDATES_PORT,
    settings_rpc_port: str = SETTINGS_RPC_PORT,
    capture_display_id: int | None = None,
):
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    vision_subscriber = ChannelSubscriber(VISION_FRAMES_CHANNEL, latest_only=True)
    action_subscriber = ChannelSubscriber(INPUT_ACTIONS_CHANNEL, latest_only=True)
    settings_client = SettingsClient(
        source_name="visualization",
        host=settings_host,
        updates_port=settings_updates_port,
        rpc_port=settings_rpc_port,
    )
    settings_client.start()

    fps = 0.0
    frame_count = 0
    fps_start_time = time.time()
    last_settings_refresh = 0.0
    window_positioned = False

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
            if now - last_settings_refresh >= SETTINGS_REFRESH_INTERVAL_SECONDS:
                try:
                    settings_client.refresh_snapshot()
                except Exception:
                    pass
                last_settings_refresh = now

            action_message = action_subscriber.get_latest(
                before_timestamp_ns=frame_capture_timestamp_ns(packet)
            )
            action = None if action_message is None else action_message.payload
            blackbox_enabled = bool(settings_client.get("blackbox.enabled", False))
            recording_enabled = bool(
                settings_client.get("blackbox.recording_enabled", False)
            )
            record_hotkey = str(
                settings_client.get("blackbox.record_hotkey", "F8")
            )

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
                _draw_text(
                    frame,
                    _input_device_overlay(action),
                    200,
                    color=(255, 255, 0),
                )
                controller_summary = _controller_summary_overlay(action)
                if controller_summary is not None:
                    _draw_text(
                        frame,
                        controller_summary,
                        240,
                        color=(255, 255, 0),
                    )

            capture_text, capture_color = _capture_overlay(
                blackbox_enabled,
                recording_enabled,
                record_hotkey,
            )
            _draw_text(
                frame,
                capture_text,
                280 if action is not None else 200,
                color=capture_color,
            )

            if not window_positioned:
                _position_visualization_window(capture_display_id)
                window_positioned = True

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord("q") or key == 27:
                break
    finally:
        settings_client.close()
        try:
            action_subscriber.close()
        except Exception:
            pass
        vision_subscriber.close()
        cv2.destroyAllWindows()
