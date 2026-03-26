from __future__ import annotations

import json
import time
from typing import Any

import zmq

from gtapilot.ipc.settings_registry import (
    INITIAL_SETTINGS_SOURCE,
    SETTINGS_RPC_PORT,
    SETTINGS_UPDATES_PORT,
    SETTINGS_UPDATES_TOPIC,
    build_settings_registry,
)
from gtapilot.ipc.settings_types import (
    SettingSpec,
    SettingValue,
    SettingsRequest,
    SettingsResponse,
)


class SettingsRuntime:
    def __init__(
        self,
        *,
        registry: dict[str, SettingSpec] | None = None,
        host: str = "127.0.0.1",
        updates_port: str = SETTINGS_UPDATES_PORT,
        rpc_port: str = SETTINGS_RPC_PORT,
        updates_topic: bytes = SETTINGS_UPDATES_TOPIC,
        rpc_timeout_ms: int = 1000,
    ):
        self.registry = (
            dict(registry) if registry is not None else build_settings_registry()
        )
        now_ns = time.time_ns()
        self._values: dict[str, SettingValue] = {
            key: SettingValue(
                key=spec.key,
                value=spec.default,
                revision=1,
                updated_timestamp_ns=now_ns,
                source=INITIAL_SETTINGS_SOURCE,
            )
            for key, spec in self.registry.items()
        }

        self.host = host
        self.updates_port = updates_port
        self.rpc_port = rpc_port
        self.updates_topic = updates_topic
        self.rpc_timeout_ms = rpc_timeout_ms

        self._running = False
        self._context = zmq.Context()
        self._updates_socket = self._context.socket(zmq.PUB)
        self._updates_socket.setsockopt(zmq.LINGER, 0)
        self._rpc_socket = self._context.socket(zmq.REP)
        self._rpc_socket.setsockopt(zmq.LINGER, 0)
        self._rpc_socket.setsockopt(zmq.RCVTIMEO, rpc_timeout_ms)

        self._updates_socket.bind(f"tcp://{self.host}:{self.updates_port}")
        self._rpc_socket.bind(f"tcp://{self.host}:{self.rpc_port}")
        time.sleep(0.1)

    def _shutdown_sockets(self) -> None:
        try:
            if not self._updates_socket.closed:
                self._updates_socket.close()
            if not self._rpc_socket.closed:
                self._rpc_socket.close()
            if not self._context.closed:
                self._context.term()
        except Exception:
            pass

    def _snapshot_copy(self) -> dict[str, SettingValue]:
        return {
            key: SettingValue.from_dict(value.to_dict())
            for key, value in self._values.items()
        }

    def _broadcast_update(self, setting_value: SettingValue) -> None:
        update_bytes = json.dumps(setting_value.to_dict()).encode("utf-8")
        self._updates_socket.send_multipart([self.updates_topic, update_bytes])

    def _set_value(
        self,
        *,
        key: str,
        value: Any,
        source: str,
    ) -> SettingsResponse:
        spec = self.registry.get(key)
        if spec is None:
            return SettingsResponse(ok=False, error=f"Unknown setting key '{key}'.")
        if not spec.mutable:
            return SettingsResponse(ok=False, error=f"Setting '{key}' is read-only.")
        if source not in spec.allowed_sources:
            return SettingsResponse(
                ok=False,
                error=f"Source '{source}' is not allowed to update '{key}'.",
            )

        try:
            normalized_value = spec.normalize(value)
        except (TypeError, ValueError) as error:
            return SettingsResponse(ok=False, error=str(error))

        current = self._values[key]
        updated_value = SettingValue(
            key=key,
            value=normalized_value,
            revision=current.revision + 1,
            updated_timestamp_ns=time.time_ns(),
            source=source,
        )
        self._values[key] = updated_value
        self._broadcast_update(updated_value)
        return SettingsResponse(ok=True, value=updated_value)

    def handle_request(self, request: SettingsRequest) -> SettingsResponse:
        if request.op == "get_snapshot":
            return SettingsResponse(ok=True, snapshot=self._snapshot_copy())

        if request.op == "get_value":
            if request.key is None:
                return SettingsResponse(ok=False, error="Missing setting key.")
            setting_value = self._values.get(request.key)
            if setting_value is None:
                return SettingsResponse(
                    ok=False,
                    error=f"Unknown setting key '{request.key}'.",
                )
            return SettingsResponse(
                ok=True,
                value=SettingValue.from_dict(setting_value.to_dict()),
            )

        if request.op == "set_value":
            if request.key is None:
                return SettingsResponse(ok=False, error="Missing setting key.")
            return self._set_value(
                key=request.key,
                value=request.value,
                source=request.source,
            )

        return SettingsResponse(
            ok=False,
            error=f"Unsupported settings operation '{request.op}'.",
        )

    def serve_forever(self) -> None:
        self._running = True
        try:
            while self._running:
                try:
                    request_payload = self._rpc_socket.recv_json()
                except zmq.Again:
                    continue
                except zmq.ContextTerminated:
                    break
                except zmq.ZMQError:
                    if not self._running:
                        break
                    continue

                response = self.handle_request(SettingsRequest.from_dict(request_payload))
                try:
                    self._rpc_socket.send_json(response.to_dict())
                except zmq.ZMQError:
                    if not self._running:
                        break
                    continue
        finally:
            self._shutdown_sockets()

    def close(self) -> None:
        self._running = False


def main() -> None:
    SettingsRuntime().serve_forever()
