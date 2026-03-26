from __future__ import annotations

import json
import threading
import time
from typing import Any

import zmq

from gtapilot.ipc.settings_registry import (
    SETTINGS_RPC_PORT,
    SETTINGS_UPDATES_PORT,
    SETTINGS_UPDATES_TOPIC,
)
from gtapilot.ipc.settings_types import SettingValue, SettingsRequest, SettingsResponse


class SettingsClient:
    def __init__(
        self,
        *,
        source_name: str,
        host: str = "127.0.0.1",
        updates_port: str = SETTINGS_UPDATES_PORT,
        rpc_port: str = SETTINGS_RPC_PORT,
        updates_topic: bytes = SETTINGS_UPDATES_TOPIC,
        rpc_timeout_ms: int = 2000,
        update_timeout_ms: int = 1000,
    ):
        self.source_name = source_name
        self.host = host
        self.updates_port = updates_port
        self.rpc_port = rpc_port
        self.updates_topic = updates_topic
        self.rpc_timeout_ms = rpc_timeout_ms
        self.update_timeout_ms = update_timeout_ms

        self._snapshot: dict[str, SettingValue] = {}
        self._lock = threading.Lock()
        self._request_lock = threading.Lock()
        self._running = False
        self._context: zmq.Context | None = None
        self._req_socket: zmq.Socket | None = None
        self._sub_socket: zmq.Socket | None = None
        self._listener_thread: threading.Thread | None = None

    def _merge_setting_value(self, update: SettingValue) -> None:
        with self._lock:
            current = self._snapshot.get(update.key)
            if current is None or update.revision >= current.revision:
                self._snapshot[update.key] = SettingValue.from_dict(update.to_dict())

    def _merge_snapshot(self, snapshot: dict[str, SettingValue]) -> None:
        for setting_value in snapshot.values():
            self._merge_setting_value(setting_value)

    def _connect_req_socket(self) -> None:
        if self._context is None:
            raise RuntimeError("SettingsClient context is not initialized.")
        self._req_socket = self._context.socket(zmq.REQ)
        self._req_socket.setsockopt(zmq.LINGER, 0)
        self._req_socket.setsockopt(zmq.RCVTIMEO, self.rpc_timeout_ms)
        self._req_socket.setsockopt(zmq.SNDTIMEO, self.rpc_timeout_ms)
        self._req_socket.connect(f"tcp://{self.host}:{self.rpc_port}")

    def _reset_req_socket(self) -> None:
        try:
            if self._req_socket is not None and not self._req_socket.closed:
                self._req_socket.close()
        except Exception:
            pass
        self._connect_req_socket()

    def start(self) -> None:
        if self._running:
            return

        self._context = zmq.Context()
        self._connect_req_socket()

        self._sub_socket = self._context.socket(zmq.SUB)
        self._sub_socket.setsockopt(zmq.LINGER, 0)
        self._sub_socket.setsockopt(zmq.RCVTIMEO, self.update_timeout_ms)
        self._sub_socket.connect(f"tcp://{self.host}:{self.updates_port}")
        self._sub_socket.subscribe(self.updates_topic)
        time.sleep(0.1)

        self._running = True
        self._listener_thread = threading.Thread(
            target=self._listen_for_updates,
            name=f"SettingsClient[{self.source_name}]",
            daemon=True,
        )
        self._listener_thread.start()
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            try:
                self.refresh_snapshot()
                return
            except TimeoutError:
                continue

        self.close()
        raise RuntimeError("Failed to load settings snapshot from SettingsRuntime.")

    def _request(self, request: SettingsRequest) -> SettingsResponse:
        if self._req_socket is None:
            raise RuntimeError("SettingsClient.start() must be called before requests.")

        payload = request.to_dict()
        payload["source"] = request.source or self.source_name
        with self._request_lock:
            try:
                self._req_socket.send_json(payload)
                response_payload = self._req_socket.recv_json()
            except zmq.Again as error:
                self._reset_req_socket()
                raise TimeoutError("Timed out waiting for settings RPC response.") from error

        return SettingsResponse.from_dict(response_payload)

    def _listen_for_updates(self) -> None:
        if self._sub_socket is None:
            return

        while self._running:
            try:
                topic, update_bytes = self._sub_socket.recv_multipart()
                if topic != self.updates_topic:
                    continue
                update = SettingValue.from_dict(json.loads(update_bytes.decode("utf-8")))
                self._merge_setting_value(update)
            except zmq.Again:
                continue
            except zmq.ContextTerminated:
                break
            except Exception:
                continue

    def refresh_snapshot(self) -> dict[str, SettingValue]:
        response = self._request(SettingsRequest(op="get_snapshot"))
        if not response.ok or response.snapshot is None:
            raise ValueError(response.error or "Failed to refresh settings snapshot.")

        self._merge_snapshot(response.snapshot)
        return self.snapshot()

    def fetch_setting_value(self, key: str) -> SettingValue | None:
        response = self._request(SettingsRequest(op="get_value", key=key))
        if not response.ok:
            raise ValueError(response.error or f"Failed to fetch setting '{key}'.")
        if response.value is None:
            return None

        self._merge_setting_value(response.value)
        return self.get_setting_value(key)

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            setting_value = self._snapshot.get(key)
            if setting_value is None:
                return default
            return setting_value.value

    def get_setting_value(self, key: str) -> SettingValue | None:
        with self._lock:
            setting_value = self._snapshot.get(key)
            if setting_value is None:
                return None
            return SettingValue.from_dict(setting_value.to_dict())

    def snapshot(self) -> dict[str, SettingValue]:
        with self._lock:
            return {
                key: SettingValue.from_dict(value.to_dict())
                for key, value in self._snapshot.items()
            }

    def set(self, key: str, value: Any, source: str | None = None) -> SettingValue:
        request = SettingsRequest(
            op="set_value",
            key=key,
            value=value,
            source=self.source_name if source is None else source,
        )
        response = self._request(request)
        if not response.ok or response.value is None:
            raise ValueError(response.error or f"Failed to set setting '{key}'.")

        self._merge_setting_value(response.value)
        return response.value

    def close(self) -> None:
        self._running = False
        if self._listener_thread is not None and self._listener_thread.is_alive():
            self._listener_thread.join(timeout=2.0)

        try:
            if self._sub_socket is not None and not self._sub_socket.closed:
                self._sub_socket.close()
            if self._req_socket is not None and not self._req_socket.closed:
                self._req_socket.close()
            if self._context is not None and not self._context.closed:
                self._context.term()
        except Exception:
            pass
