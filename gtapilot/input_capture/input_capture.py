from __future__ import annotations

import ctypes
import time

from gtapilot.ipc.channel import ChannelPublisher
from gtapilot.ipc.channels import INPUT_ACTIONS_CHANNEL, ActionPacket

POLL_HZ = 50
POLL_INTERVAL = 1.0 / POLL_HZ

VK_LEFT = 0x25
VK_UP = 0x26
VK_RIGHT = 0x27
VK_DOWN = 0x28
VK_SPACE = 0x20


def _key_pressed(vk_code: int) -> bool:
    if ctypes.windll is None:  # pragma: no cover - Windows-only path
        return False
    return bool(ctypes.windll.user32.GetAsyncKeyState(vk_code) & 0x8000)


def _read_manual_inputs() -> dict[str, bool]:
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


def _compute_action(raw_inputs: dict[str, bool]) -> dict[str, float]:
    steer_left = raw_inputs["a"] or raw_inputs["left"]
    steer_right = raw_inputs["d"] or raw_inputs["right"]
    if steer_left and not steer_right:
        steer = -1.0
    elif steer_right and not steer_left:
        steer = 1.0
    else:
        steer = 0.0

    throttle = 1.0 if (raw_inputs["w"] or raw_inputs["up"]) else 0.0
    brake = 1.0 if (raw_inputs["s"] or raw_inputs["down"]) else 0.0
    handbrake = 1.0 if raw_inputs["space"] else 0.0

    # Reverse is not directly observable from keyboard-only inputs without vehicle
    # state; keep it explicit and false until the ScriptHook-based vehicle-state
    # pipeline lands.
    reverse = 0.0
    pilot_active = 0.0
    return {
        "steer": steer,
        "throttle": throttle,
        "brake": brake,
        "handbrake": handbrake,
        "reverse": reverse,
        "pilot_active": pilot_active,
    }


def main():
    publisher = ChannelPublisher(
        INPUT_ACTIONS_CHANNEL,
        source_name="manual_keyboard",
    )
    try:
        while True:
            started_at = time.perf_counter()
            raw_inputs = _read_manual_inputs()
            action = _compute_action(raw_inputs)
            sample_timestamp_ns = time.time_ns()
            publisher.publish(
                ActionPacket(raw_inputs=raw_inputs, **action),
                timestamp_ns=sample_timestamp_ns,
            )

            elapsed = time.perf_counter() - started_at
            if elapsed < POLL_INTERVAL:
                time.sleep(POLL_INTERVAL - elapsed)
    finally:
        publisher.close()
