from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


Validator = Callable[[Any], Any]


@dataclass(slots=True, frozen=True)
class SettingSpec:
    key: str
    default: Any
    mutable: bool
    allowed_sources: frozenset[str] = field(default_factory=frozenset)
    validator: Validator | None = None

    def normalize(self, value: Any) -> Any:
        if self.validator is None:
            return value
        return self.validator(value)


@dataclass(slots=True)
class SettingValue:
    key: str
    value: Any
    revision: int
    updated_timestamp_ns: int
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "revision": int(self.revision),
            "updated_timestamp_ns": int(self.updated_timestamp_ns),
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SettingValue":
        return cls(
            key=str(payload["key"]),
            value=payload.get("value"),
            revision=int(payload["revision"]),
            updated_timestamp_ns=int(payload["updated_timestamp_ns"]),
            source=str(payload["source"]),
        )


@dataclass(slots=True)
class SettingsRequest:
    op: str
    key: str | None = None
    value: Any = None
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "key": self.key,
            "value": self.value,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SettingsRequest":
        return cls(
            op=str(payload["op"]),
            key=None if payload.get("key") is None else str(payload["key"]),
            value=payload.get("value"),
            source=str(payload.get("source", "")),
        )


@dataclass(slots=True)
class SettingsResponse:
    ok: bool
    error: str | None = None
    value: SettingValue | None = None
    snapshot: dict[str, SettingValue] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": bool(self.ok),
            "error": self.error,
            "value": None if self.value is None else self.value.to_dict(),
            "snapshot": (
                None
                if self.snapshot is None
                else {
                    key: setting_value.to_dict()
                    for key, setting_value in self.snapshot.items()
                }
            ),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SettingsResponse":
        value_payload = payload.get("value")
        snapshot_payload = payload.get("snapshot")
        return cls(
            ok=bool(payload["ok"]),
            error=None if payload.get("error") is None else str(payload["error"]),
            value=(
                None
                if value_payload is None
                else SettingValue.from_dict(value_payload)
            ),
            snapshot=(
                None
                if snapshot_payload is None
                else {
                    str(key): SettingValue.from_dict(setting_value)
                    for key, setting_value in dict(snapshot_payload).items()
                }
            ),
        )
