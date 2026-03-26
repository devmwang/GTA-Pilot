from __future__ import annotations

import ctypes
import time
from typing import Any

from gtapilot.config import BLACKBOX_RECORD_HOTKEY, BLACKBOX_RECORD_ON_START
from gtapilot.ipc.channel import ChannelPublisher
from gtapilot.ipc.channels import INPUT_ACTIONS_CHANNEL, ActionPacket
from gtapilot.ipc.settings_client import SettingsClient
from gtapilot.ipc.settings_registry import (
    SETTINGS_RPC_PORT,
    SETTINGS_UPDATES_PORT,
)

POLL_HZ = 50
POLL_INTERVAL = 1.0 / POLL_HZ

KEYBOARD_DEVICE = "keyboard"
CONTROLLER_DEVICE = "xinput_controller"

KEYBOARD_DRIVING_KEYS = (
    "w",
    "a",
    "s",
    "d",
    "up",
    "down",
    "left",
    "right",
    "space",
)

VK_LEFT = 0x25
VK_UP = 0x26
VK_RIGHT = 0x27
VK_DOWN = 0x28
VK_SPACE = 0x20
VK_F8 = 0x77

XINPUT_STICK_ACTIVITY_THRESHOLD = 0.08
XINPUT_TRIGGER_ACTIVITY_THRESHOLD = 0.05

XINPUT_DLL_NAMES = ("xinput1_4.dll", "xinput1_3.dll", "xinput9_1_0.dll")
XINPUT_ERROR_SUCCESS = 0

XINPUT_GAMEPAD_DPAD_UP = 0x0001
XINPUT_GAMEPAD_DPAD_DOWN = 0x0002
XINPUT_GAMEPAD_DPAD_LEFT = 0x0004
XINPUT_GAMEPAD_DPAD_RIGHT = 0x0008
XINPUT_GAMEPAD_START = 0x0010
XINPUT_GAMEPAD_BACK = 0x0020
XINPUT_GAMEPAD_LEFT_THUMB = 0x0040
XINPUT_GAMEPAD_RIGHT_THUMB = 0x0080
XINPUT_GAMEPAD_LEFT_SHOULDER = 0x0100
XINPUT_GAMEPAD_RIGHT_SHOULDER = 0x0200
XINPUT_GAMEPAD_A = 0x1000
XINPUT_GAMEPAD_B = 0x2000
XINPUT_GAMEPAD_X = 0x4000
XINPUT_GAMEPAD_Y = 0x8000

_XINPUT_DLL: ctypes.WinDLL | None = None
_XINPUT_DLL_ATTEMPTED = False


class XINPUT_GAMEPAD(ctypes.Structure):
    _fields_ = [
        ("wButtons", ctypes.c_ushort),
        ("bLeftTrigger", ctypes.c_ubyte),
        ("bRightTrigger", ctypes.c_ubyte),
        ("sThumbLX", ctypes.c_short),
        ("sThumbLY", ctypes.c_short),
        ("sThumbRX", ctypes.c_short),
        ("sThumbRY", ctypes.c_short),
    ]


class XINPUT_STATE(ctypes.Structure):
    _fields_ = [
        ("dwPacketNumber", ctypes.c_ulong),
        ("Gamepad", XINPUT_GAMEPAD),
    ]


def _zero_action_values() -> dict[str, float]:
    return {
        "steer": 0.0,
        "throttle": 0.0,
        "brake": 0.0,
        "handbrake": 0.0,
        "reverse": 0.0,
        "pilot_active": 0.0,
    }


def _empty_keyboard_inputs() -> dict[str, Any]:
    return {
        "connected": True,
        "driving_active": False,
        "keys": {key: False for key in KEYBOARD_DRIVING_KEYS},
    }


def _empty_controller_inputs() -> dict[str, Any]:
    return {
        "connected": False,
        "controller_index": None,
        "packet_number": None,
        "driving_active": False,
        "buttons": {
            "a": False,
            "b": False,
            "x": False,
            "y": False,
            "lb": False,
            "rb": False,
            "back": False,
            "start": False,
            "left_thumb": False,
            "right_thumb": False,
            "dpad_up": False,
            "dpad_down": False,
            "dpad_left": False,
            "dpad_right": False,
        },
        "triggers": {"left": 0.0, "right": 0.0},
        "sticks": {
            "left_x": 0.0,
            "left_y": 0.0,
            "right_x": 0.0,
            "right_y": 0.0,
        },
    }


def _key_pressed(vk_code: int) -> bool:
    if ctypes.windll is None:  # pragma: no cover - Windows-only path
        return False
    return bool(ctypes.windll.user32.GetAsyncKeyState(vk_code) & 0x8000)


def _read_keyboard_driving_keys() -> dict[str, bool]:
    return {
        "w": _key_pressed(ord("W")),
        "a": _key_pressed(ord("A")),
        "s": _key_pressed(ord("S")),
        "d": _key_pressed(ord("D")),
        "up": _key_pressed(VK_UP),
        "down": _key_pressed(VK_DOWN),
        "left": _key_pressed(VK_LEFT),
        "right": _key_pressed(VK_RIGHT),
        "space": _key_pressed(VK_SPACE),
    }


def _record_hotkey_pressed(hotkey: str) -> bool:
    if hotkey.strip().upper() != "F8":
        raise ValueError(f"Unsupported record hotkey '{hotkey}'. Expected F8.")
    return _key_pressed(VK_F8)


def _keyboard_inputs_from_keys(keys: dict[str, bool]) -> dict[str, Any]:
    return {
        "connected": True,
        "driving_active": any(bool(keys.get(key, False)) for key in KEYBOARD_DRIVING_KEYS),
        "keys": {key: bool(keys.get(key, False)) for key in KEYBOARD_DRIVING_KEYS},
    }


def _keyboard_action_from_inputs(keyboard_inputs: dict[str, Any]) -> dict[str, float]:
    keys = dict(keyboard_inputs.get("keys", {}))
    steer_left = bool(keys.get("a", False) or keys.get("left", False))
    steer_right = bool(keys.get("d", False) or keys.get("right", False))
    if steer_left and not steer_right:
        steer = -1.0
    elif steer_right and not steer_left:
        steer = 1.0
    else:
        steer = 0.0

    throttle = 1.0 if bool(keys.get("w", False) or keys.get("up", False)) else 0.0
    brake = 1.0 if bool(keys.get("s", False) or keys.get("down", False)) else 0.0
    handbrake = 1.0 if bool(keys.get("space", False)) else 0.0
    return {
        "steer": steer,
        "throttle": throttle,
        "brake": brake,
        "handbrake": handbrake,
        "reverse": 0.0,
        "pilot_active": 0.0,
    }


def _normalize_trigger_byte(value: int) -> float:
    return max(0.0, min(1.0, float(int(value)) / 255.0))


def _normalize_stick_short(value: int) -> float:
    normalized = float(value) / 32767.0 if value >= 0 else float(value) / 32768.0
    return max(-1.0, min(1.0, normalized))


def _apply_axis_deadzone(value: float, threshold: float) -> float:
    if abs(float(value)) < threshold:
        return 0.0
    return float(value)


def _apply_trigger_threshold(value: float, threshold: float) -> float:
    if float(value) < threshold:
        return 0.0
    return float(value)


def _controller_buttons_from_mask(button_mask: int) -> dict[str, bool]:
    return {
        "a": bool(button_mask & XINPUT_GAMEPAD_A),
        "b": bool(button_mask & XINPUT_GAMEPAD_B),
        "x": bool(button_mask & XINPUT_GAMEPAD_X),
        "y": bool(button_mask & XINPUT_GAMEPAD_Y),
        "lb": bool(button_mask & XINPUT_GAMEPAD_LEFT_SHOULDER),
        "rb": bool(button_mask & XINPUT_GAMEPAD_RIGHT_SHOULDER),
        "back": bool(button_mask & XINPUT_GAMEPAD_BACK),
        "start": bool(button_mask & XINPUT_GAMEPAD_START),
        "left_thumb": bool(button_mask & XINPUT_GAMEPAD_LEFT_THUMB),
        "right_thumb": bool(button_mask & XINPUT_GAMEPAD_RIGHT_THUMB),
        "dpad_up": bool(button_mask & XINPUT_GAMEPAD_DPAD_UP),
        "dpad_down": bool(button_mask & XINPUT_GAMEPAD_DPAD_DOWN),
        "dpad_left": bool(button_mask & XINPUT_GAMEPAD_DPAD_LEFT),
        "dpad_right": bool(button_mask & XINPUT_GAMEPAD_DPAD_RIGHT),
    }


def _controller_inputs_from_values(
    *,
    connected: bool,
    controller_index: int | None,
    packet_number: int | None,
    button_mask: int = 0,
    left_trigger: int = 0,
    right_trigger: int = 0,
    left_x: int = 0,
    left_y: int = 0,
    right_x: int = 0,
    right_y: int = 0,
) -> dict[str, Any]:
    if not connected:
        return _empty_controller_inputs()

    triggers = {
        "left": _normalize_trigger_byte(left_trigger),
        "right": _normalize_trigger_byte(right_trigger),
    }
    sticks = {
        "left_x": _normalize_stick_short(left_x),
        "left_y": _normalize_stick_short(left_y),
        "right_x": _normalize_stick_short(right_x),
        "right_y": _normalize_stick_short(right_y),
    }
    buttons = _controller_buttons_from_mask(button_mask)
    driving_active = (
        abs(sticks["left_x"]) >= XINPUT_STICK_ACTIVITY_THRESHOLD
        or triggers["right"] >= XINPUT_TRIGGER_ACTIVITY_THRESHOLD
        or triggers["left"] >= XINPUT_TRIGGER_ACTIVITY_THRESHOLD
        or buttons["rb"]
    )
    return {
        "connected": True,
        "controller_index": int(controller_index) if controller_index is not None else None,
        "packet_number": int(packet_number) if packet_number is not None else None,
        "driving_active": driving_active,
        "buttons": buttons,
        "triggers": triggers,
        "sticks": sticks,
    }


def _controller_inputs_from_state(
    controller_index: int,
    state: XINPUT_STATE,
) -> dict[str, Any]:
    gamepad = state.Gamepad
    return _controller_inputs_from_values(
        connected=True,
        controller_index=controller_index,
        packet_number=int(state.dwPacketNumber),
        button_mask=int(gamepad.wButtons),
        left_trigger=int(gamepad.bLeftTrigger),
        right_trigger=int(gamepad.bRightTrigger),
        left_x=int(gamepad.sThumbLX),
        left_y=int(gamepad.sThumbLY),
        right_x=int(gamepad.sThumbRX),
        right_y=int(gamepad.sThumbRY),
    )


def _controller_action_from_inputs(controller_inputs: dict[str, Any]) -> dict[str, float]:
    if not bool(controller_inputs.get("connected", False)):
        return _zero_action_values()

    triggers = dict(controller_inputs.get("triggers", {}))
    sticks = dict(controller_inputs.get("sticks", {}))
    buttons = dict(controller_inputs.get("buttons", {}))
    return {
        "steer": _apply_axis_deadzone(
            float(sticks.get("left_x", 0.0)),
            XINPUT_STICK_ACTIVITY_THRESHOLD,
        ),
        "throttle": _apply_trigger_threshold(
            float(triggers.get("right", 0.0)),
            XINPUT_TRIGGER_ACTIVITY_THRESHOLD,
        ),
        "brake": _apply_trigger_threshold(
            float(triggers.get("left", 0.0)),
            XINPUT_TRIGGER_ACTIVITY_THRESHOLD,
        ),
        "handbrake": 1.0 if bool(buttons.get("rb", False)) else 0.0,
        "reverse": 0.0,
        "pilot_active": 0.0,
    }


def _load_xinput_dll() -> ctypes.WinDLL | None:
    global _XINPUT_DLL_ATTEMPTED, _XINPUT_DLL
    if _XINPUT_DLL_ATTEMPTED:
        return _XINPUT_DLL

    _XINPUT_DLL_ATTEMPTED = True
    if not hasattr(ctypes, "WinDLL"):  # pragma: no cover - Windows-only path
        return None

    for dll_name in XINPUT_DLL_NAMES:
        try:
            xinput_dll = ctypes.WinDLL(dll_name)
            xinput_dll.XInputGetState.argtypes = [
                ctypes.c_uint,
                ctypes.POINTER(XINPUT_STATE),
            ]
            xinput_dll.XInputGetState.restype = ctypes.c_uint
            _XINPUT_DLL = xinput_dll
            return _XINPUT_DLL
        except OSError:
            continue
    return None


def _read_xinput_controller_inputs() -> dict[str, Any]:
    xinput_dll = _load_xinput_dll()
    if xinput_dll is None:
        return _empty_controller_inputs()

    for controller_index in range(4):
        state = XINPUT_STATE()
        result = xinput_dll.XInputGetState(
            ctypes.c_uint(controller_index),
            ctypes.byref(state),
        )
        if int(result) == XINPUT_ERROR_SUCCESS:
            return _controller_inputs_from_state(controller_index, state)
    return _empty_controller_inputs()


def _resolve_active_device(
    *,
    keyboard_inputs: dict[str, Any],
    controller_inputs: dict[str, Any],
    keyboard_last_active_timestamp_ns: int,
    controller_last_active_timestamp_ns: int,
    previous_active_device: str,
) -> str:
    keyboard_active = bool(keyboard_inputs.get("driving_active", False))
    controller_active = bool(
        controller_inputs.get("connected", False)
        and controller_inputs.get("driving_active", False)
    )

    if keyboard_active and not controller_active:
        return KEYBOARD_DEVICE
    if controller_active and not keyboard_active:
        return CONTROLLER_DEVICE
    if not keyboard_active and not controller_active:
        return "none"

    if keyboard_last_active_timestamp_ns > controller_last_active_timestamp_ns:
        return KEYBOARD_DEVICE
    if controller_last_active_timestamp_ns > keyboard_last_active_timestamp_ns:
        return CONTROLLER_DEVICE
    if previous_active_device in {KEYBOARD_DEVICE, CONTROLLER_DEVICE}:
        return previous_active_device
    return KEYBOARD_DEVICE


def _effective_action_for_device(
    active_device: str,
    *,
    keyboard_action: dict[str, float],
    controller_action: dict[str, float],
) -> dict[str, float]:
    if active_device == KEYBOARD_DEVICE:
        return dict(keyboard_action)
    if active_device == CONTROLLER_DEVICE:
        return dict(controller_action)
    return _zero_action_values()


def _build_action_packet(
    *,
    active_device: str,
    effective_action: dict[str, float],
    keyboard_action: dict[str, float],
    controller_action: dict[str, float],
    keyboard_inputs: dict[str, Any],
    controller_inputs: dict[str, Any],
) -> ActionPacket:
    return ActionPacket(
        steer=float(effective_action["steer"]),
        throttle=float(effective_action["throttle"]),
        brake=float(effective_action["brake"]),
        handbrake=float(effective_action["handbrake"]),
        reverse=float(effective_action["reverse"]),
        pilot_active=float(effective_action["pilot_active"]),
        active_device=active_device,
        device_actions={
            KEYBOARD_DEVICE: dict(keyboard_action),
            CONTROLLER_DEVICE: dict(controller_action),
        },
        device_inputs={
            KEYBOARD_DEVICE: dict(keyboard_inputs),
            CONTROLLER_DEVICE: dict(controller_inputs),
        },
    )


def _update_recording_state(
    *,
    record_hotkey_pressed: bool,
    record_hotkey_was_pressed: bool,
    recording_enabled: bool,
) -> tuple[bool, bool, bool]:
    if record_hotkey_pressed and not record_hotkey_was_pressed:
        return (not recording_enabled, record_hotkey_pressed, True)
    return (recording_enabled, record_hotkey_pressed, False)


def _authoritative_recording_enabled(
    settings_client: SettingsClient,
) -> bool:
    setting_value = settings_client.fetch_setting_value("blackbox.recording_enabled")
    if setting_value is None:
        return bool(
            settings_client.get(
                "blackbox.recording_enabled",
                BLACKBOX_RECORD_ON_START,
            )
        )
    return bool(setting_value.value)


def main(
    settings_host: str = "127.0.0.1",
    settings_updates_port: str = SETTINGS_UPDATES_PORT,
    settings_rpc_port: str = SETTINGS_RPC_PORT,
):
    action_publisher = ChannelPublisher(
        INPUT_ACTIONS_CHANNEL,
        source_name="manual_input",
    )
    settings_client = SettingsClient(
        source_name="manual_input",
        host=settings_host,
        updates_port=settings_updates_port,
        rpc_port=settings_rpc_port,
    )
    settings_client.start()
    record_hotkey_was_pressed = False
    active_device = "none"
    keyboard_last_active_timestamp_ns = 0
    controller_last_active_timestamp_ns = 0
    try:
        while True:
            started_at = time.perf_counter()
            sample_timestamp_ns = time.time_ns()

            keyboard_inputs = _keyboard_inputs_from_keys(_read_keyboard_driving_keys())
            controller_inputs = _read_xinput_controller_inputs()
            keyboard_action = _keyboard_action_from_inputs(keyboard_inputs)
            controller_action = _controller_action_from_inputs(controller_inputs)

            if keyboard_inputs["driving_active"]:
                keyboard_last_active_timestamp_ns = sample_timestamp_ns
            if controller_inputs["driving_active"]:
                controller_last_active_timestamp_ns = sample_timestamp_ns

            active_device = _resolve_active_device(
                keyboard_inputs=keyboard_inputs,
                controller_inputs=controller_inputs,
                keyboard_last_active_timestamp_ns=keyboard_last_active_timestamp_ns,
                controller_last_active_timestamp_ns=controller_last_active_timestamp_ns,
                previous_active_device=active_device,
            )
            effective_action = _effective_action_for_device(
                active_device,
                keyboard_action=keyboard_action,
                controller_action=controller_action,
            )

            current_recording_enabled = bool(
                settings_client.get(
                    "blackbox.recording_enabled",
                    BLACKBOX_RECORD_ON_START,
                )
            )
            record_hotkey_pressed = _record_hotkey_pressed(BLACKBOX_RECORD_HOTKEY)
            next_recording_enabled, record_hotkey_was_pressed, did_toggle = (
                _update_recording_state(
                    record_hotkey_pressed=record_hotkey_pressed,
                    record_hotkey_was_pressed=record_hotkey_was_pressed,
                    recording_enabled=current_recording_enabled,
                )
            )
            if did_toggle:
                next_recording_enabled = not _authoritative_recording_enabled(
                    settings_client
                )
                settings_client.set(
                    "blackbox.recording_enabled",
                    next_recording_enabled,
                )

            action_publisher.publish(
                _build_action_packet(
                    active_device=active_device,
                    effective_action=effective_action,
                    keyboard_action=keyboard_action,
                    controller_action=controller_action,
                    keyboard_inputs=keyboard_inputs,
                    controller_inputs=controller_inputs,
                ),
                timestamp_ns=sample_timestamp_ns,
            )

            elapsed = time.perf_counter() - started_at
            if elapsed < POLL_INTERVAL:
                time.sleep(POLL_INTERVAL - elapsed)
    finally:
        settings_client.close()
        action_publisher.close()
