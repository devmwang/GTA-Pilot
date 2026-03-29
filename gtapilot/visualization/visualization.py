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
ACTION_ALIGNMENT_HISTORY_SIZE = 32
ACTION_ALIGNMENT_FUTURE_TOLERANCE_NS = 25_000_000
WINDOW_NAME = "GTA Pilot Visualization"


@dataclass(slots=True, frozen=True)
class MonitorBounds:
    handle: int
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
                handle=int(ctypes.cast(hmonitor, ctypes.c_void_p).value or 0),
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


def _primary_monitor(monitors: list[MonitorBounds]) -> MonitorBounds | None:
    for monitor in monitors:
        if monitor.is_primary:
            return monitor
    return monitors[0] if monitors else None


def _parse_hwnd(raw_hwnd: object) -> int | None:
    if raw_hwnd is None:
        return None
    if isinstance(raw_hwnd, int):
        return raw_hwnd if raw_hwnd > 0 else None
    hwnd_text = str(raw_hwnd).strip()
    if not hwnd_text:
        return None
    try:
        return int(hwnd_text, 16) if hwnd_text.lower().startswith("0x") else int(hwnd_text)
    except ValueError:
        return None


def _monitor_handle_for_window(window_handle: int | None) -> int | None:
    if window_handle is None or not hasattr(ctypes, "windll"):  # pragma: no cover
        return None

    user32 = ctypes.windll.user32
    user32.MonitorFromWindow.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    user32.MonitorFromWindow.restype = ctypes.c_void_p
    monitor_handle = user32.MonitorFromWindow(
        ctypes.c_void_p(window_handle),
        ctypes.c_uint(2),  # MONITOR_DEFAULTTONEAREST
    )
    return int(monitor_handle or 0) or None


def _pick_visualization_monitor(
    monitors: list[MonitorBounds],
    *,
    target_window_hwnd: object | None = None,
) -> MonitorBounds | None:
    primary_monitor = _primary_monitor(monitors)
    if len(monitors) <= 1:
        return primary_monitor

    game_monitor_handle = _monitor_handle_for_window(_parse_hwnd(target_window_hwnd))
    if game_monitor_handle is not None:
        alternate_monitors = [
            monitor for monitor in monitors if monitor.handle != game_monitor_handle
        ]
        if alternate_monitors:
            for monitor in alternate_monitors:
                if monitor.is_primary:
                    return monitor
            return alternate_monitors[0]

    return primary_monitor


def _position_visualization_window(*, target_window_hwnd: object | None = None) -> None:
    target_monitor = _pick_visualization_monitor(
        _enumerate_monitors(),
        target_window_hwnd=target_window_hwnd,
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


def _aligned_action_message(
    action_subscriber: ChannelSubscriber,
    frame_timestamp_ns: int,
):
    action_message = action_subscriber.get_latest(
        before_timestamp_ns=frame_timestamp_ns
    )
    if action_message is not None:
        return action_message

    latest_action_message = action_subscriber.get_latest()
    if latest_action_message is None:
        return None

    timestamp_delta_ns = (
        int(latest_action_message.envelope.message_timestamp_ns)
        - int(frame_timestamp_ns)
    )
    if 0 <= timestamp_delta_ns <= ACTION_ALIGNMENT_FUTURE_TOLERANCE_NS:
        return latest_action_message
    return None


def main(
    settings_host: str = "127.0.0.1",
    settings_updates_port: str = SETTINGS_UPDATES_PORT,
    settings_rpc_port: str = SETTINGS_RPC_PORT,
):
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    vision_subscriber = ChannelSubscriber(VISION_FRAMES_CHANNEL, latest_only=True)
    action_subscriber = ChannelSubscriber(
        INPUT_ACTIONS_CHANNEL,
        buffer_size=ACTION_ALIGNMENT_HISTORY_SIZE,
    )
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
    positioned_for_target_window_hwnd: str | None = None

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

            action_message = _aligned_action_message(
                action_subscriber,
                frame_capture_timestamp_ns(packet),
            )
            action = None if action_message is None else action_message.payload
            blackbox_enabled = bool(settings_client.get("blackbox.enabled", False))
            recording_enabled = bool(
                settings_client.get("blackbox.recording_enabled", False)
            )
            record_hotkey = str(
                settings_client.get("blackbox.record_hotkey", "F8")
            )
            frame_nominal_fps = float(
                packet.envelope.metadata.get("nominal_fps", 0.0) or 0.0
            )
            is_repeat = bool(packet.envelope.metadata.get("is_repeat", False))
            capture_mode = str(
                packet.envelope.metadata.get("capture_mode", "unknown")
            )
            target_window_title = str(
                packet.envelope.metadata.get("target_window_title", "")
            )
            target_window_hwnd = str(
                packet.envelope.metadata.get("target_window_hwnd", "")
            )

            _draw_text(frame, f"FPS: {fps:.2f} nominal={frame_nominal_fps:.2f}", 40)
            _draw_text(
                frame,
                f"Frame {packet.envelope.metadata.get('frame_id', '?')} "
                f"capture={packet.envelope.metadata.get('capture_frame_id', '?')} "
                f"mode={capture_mode} repeat={1 if is_repeat else 0}",
                80,
            )
            if target_window_title:
                _draw_text(
                    frame,
                    f"Target: {target_window_title}",
                    120,
                )
                action_base_y = 160
            else:
                action_base_y = 120

            if action is not None:
                action_source = (
                    "unknown"
                    if action_message is None
                    else action_message.envelope.source
                )
                _draw_text(
                    frame,
                    "Action "
                    f"steer={action.steer:+.1f} throttle={action.throttle:.1f} "
                    f"brake={action.brake:.1f} handbrake={action.handbrake:.1f} "
                    f"reverse={action.reverse:.1f}",
                    action_base_y,
                    color=(255, 255, 0),
                )
                pilot_mode = "POLICY" if action.pilot_active >= 0.5 else "MANUAL"
                _draw_text(
                    frame,
                    f"Pilot: {pilot_mode} action_source={action_source}",
                    action_base_y + 40,
                    color=(255, 255, 0),
                )
                _draw_text(
                    frame,
                    _input_device_overlay(action),
                    action_base_y + 80,
                    color=(255, 255, 0),
                )
                controller_summary = _controller_summary_overlay(action)
                if controller_summary is not None:
                    _draw_text(
                        frame,
                        controller_summary,
                        action_base_y + 120,
                        color=(255, 255, 0),
                    )
                    blackbox_y = action_base_y + 160
                else:
                    blackbox_y = action_base_y + 120
            else:
                blackbox_y = action_base_y

            capture_text, capture_color = _capture_overlay(
                blackbox_enabled,
                recording_enabled,
                record_hotkey,
            )
            _draw_text(
                frame,
                capture_text,
                blackbox_y,
                color=capture_color,
            )

            if target_window_hwnd != positioned_for_target_window_hwnd:
                _position_visualization_window(target_window_hwnd=target_window_hwnd)
                positioned_for_target_window_hwnd = target_window_hwnd

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
