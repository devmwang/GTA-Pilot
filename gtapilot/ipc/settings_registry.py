from __future__ import annotations

from typing import Any

from gtapilot.config import (
    BLACKBOX_ENABLED,
    BLACKBOX_PREROLL_SECONDS,
    BLACKBOX_RECORD_HOTKEY,
    BLACKBOX_RECORD_ON_START,
)
from gtapilot.ipc.settings_types import SettingSpec

SETTINGS_UPDATES_PORT = "55553"
SETTINGS_RPC_PORT = "55554"
SETTINGS_UPDATES_TOPIC = b"settings_updates"
INITIAL_SETTINGS_SOURCE = "settings_runtime"


def _validate_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    raise TypeError("Expected a bool value.")


def _validate_non_negative_float(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("Expected an int or float value.")
    normalized = float(value)
    if normalized < 0.0:
        raise ValueError("Expected a non-negative float value.")
    return normalized


def _validate_hotkey(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError("Expected a string hotkey value.")
    normalized = value.strip().upper()
    if not normalized:
        raise ValueError("Hotkey value cannot be empty.")
    if normalized != "F8":
        raise ValueError(f"Unsupported hotkey '{normalized}'. Expected F8.")
    return normalized


def build_settings_registry(
    default_overrides: dict[str, Any] | None = None,
) -> dict[str, SettingSpec]:
    overrides = dict(default_overrides or {})
    return {
        "blackbox.enabled": SettingSpec(
            key="blackbox.enabled",
            default=_validate_bool(overrides.get("blackbox.enabled", BLACKBOX_ENABLED)),
            mutable=False,
            validator=_validate_bool,
        ),
        "blackbox.recording_enabled": SettingSpec(
            key="blackbox.recording_enabled",
            default=_validate_bool(
                overrides.get("blackbox.recording_enabled", BLACKBOX_RECORD_ON_START)
            ),
            mutable=True,
            allowed_sources=frozenset({"manual_input", "tests"}),
            validator=_validate_bool,
        ),
        "blackbox.preroll_seconds": SettingSpec(
            key="blackbox.preroll_seconds",
            default=_validate_non_negative_float(
                overrides.get("blackbox.preroll_seconds", BLACKBOX_PREROLL_SECONDS)
            ),
            mutable=True,
            allowed_sources=frozenset({"trusted_local_tool", "tests"}),
            validator=_validate_non_negative_float,
        ),
        "blackbox.record_hotkey": SettingSpec(
            key="blackbox.record_hotkey",
            default=_validate_hotkey(
                overrides.get("blackbox.record_hotkey", BLACKBOX_RECORD_HOTKEY)
            ),
            mutable=False,
            validator=_validate_hotkey,
        ),
    }
