from __future__ import annotations

import collections
import copy
import json
import threading
import time
from typing import Any, Callable

import zmq

from gtapilot.ipc.types import (
    ChannelEnvelope,
    ChannelMessage,
    ChannelSpec,
    ChannelTransportEvent,
    ChannelTransportStats,
)

CHANNEL_ENVELOPE_VERSION = 1


class ChannelPublisher:
    def __init__(
        self,
        spec: ChannelSpec,
        *,
        host: str = "127.0.0.1",
        source_name: str,
        sndhwm: int | None = None,
    ):
        self.spec = spec
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.SNDHWM, sndhwm or spec.default_sndhwm)
        self.bind_addr = f"tcp://{host}:{spec.port}"
        self.source_name = source_name
        self._sequence_id = 1

        try:
            self.socket.bind(self.bind_addr)
            time.sleep(0.1)
            print(f"ChannelPublisher[{spec.name}] bound to {self.bind_addr}")
        except zmq.ZMQError:
            self.context.term()
            raise

    def publish(
        self,
        payload: Any,
        *,
        timestamp_ns: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        message_timestamp_ns = int(timestamp_ns or time.time_ns())
        publish_timestamp_ns = time.time_ns()
        payload_bytes, envelope_metadata = self.spec.codec.encode(payload, metadata)
        envelope = ChannelEnvelope(
            v=CHANNEL_ENVELOPE_VERSION,
            channel=self.spec.name,
            encoding=self.spec.codec.encoding_name,
            sequence_id=self._sequence_id,
            message_timestamp_ns=message_timestamp_ns,
            publish_timestamp_ns=publish_timestamp_ns,
            source=self.source_name,
            metadata=envelope_metadata,
        )
        self._sequence_id += 1

        envelope_bytes = json.dumps(envelope.to_dict()).encode("utf-8")
        try:
            self.socket.send_multipart([self.spec.topic, envelope_bytes, payload_bytes])
        except zmq.ZMQError as error:
            print(f"ChannelPublisher[{self.spec.name}] send error: {error}")

    def close(self) -> None:
        if hasattr(self, "socket") and not self.socket.closed:
            self.socket.close()
        if hasattr(self, "context") and not self.context.closed:
            self.context.term()


class ChannelTransportTracker:
    def __init__(
        self,
        channel_name: str,
        *,
        event_callback: Callable[[ChannelTransportEvent], None] | None = None,
    ):
        self._stats = ChannelTransportStats(channel=channel_name)
        self._event_callback = event_callback
        self._events: collections.deque[ChannelTransportEvent] = collections.deque(
            maxlen=2048
        )
        self._pending_callbacks: collections.deque[ChannelTransportEvent] = (
            collections.deque()
        )

    def _emit(self, event: ChannelTransportEvent) -> None:
        self._events.append(event)
        if self._event_callback is not None:
            self._pending_callbacks.append(event)

    def note_received(
        self,
        envelope: ChannelEnvelope,
        *,
        buffer_occupancy: int | None = None,
        timestamp_ns: int | None = None,
    ) -> None:
        self._stats.messages_received += 1
        if buffer_occupancy is not None:
            self._stats.max_buffer_occupancy = max(
                self._stats.max_buffer_occupancy,
                int(buffer_occupancy),
            )

        source = str(envelope.source)
        sequence_id = int(envelope.sequence_id)
        last_sequence_id = self._stats.last_sequence_by_source.get(source)
        if last_sequence_id is not None and sequence_id > last_sequence_id + 1:
            missing_count = int(sequence_id - last_sequence_id - 1)
            self._stats.sequence_gap_count += 1
            self._stats.missing_message_count += missing_count
            self._emit(
                ChannelTransportEvent(
                    kind="sequence_gap",
                    channel=self._stats.channel,
                    source=source,
                    timestamp_ns=int(timestamp_ns or time.time_ns()),
                    sequence_id_start=int(last_sequence_id + 1),
                    sequence_id_end=int(sequence_id - 1),
                    missing_count=missing_count,
                    buffer_occupancy=buffer_occupancy,
                )
            )
        if last_sequence_id is None or sequence_id > last_sequence_id:
            self._stats.last_sequence_by_source[source] = sequence_id

    def note_local_overflow(
        self,
        *,
        source: str,
        dropped_sequence_id: int,
        dropped_count: int = 1,
        buffer_occupancy: int | None = None,
        timestamp_ns: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        dropped_count = max(1, int(dropped_count))
        self._stats.local_overflow_count += 1
        self._stats.local_overflow_dropped_messages += dropped_count
        if buffer_occupancy is not None:
            self._stats.max_buffer_occupancy = max(
                self._stats.max_buffer_occupancy,
                int(buffer_occupancy),
            )
        self._emit(
            ChannelTransportEvent(
                kind="local_overflow",
                channel=self._stats.channel,
                source=str(source),
                timestamp_ns=int(timestamp_ns or time.time_ns()),
                sequence_id_start=int(dropped_sequence_id),
                sequence_id_end=int(dropped_sequence_id + dropped_count - 1),
                dropped_count=dropped_count,
                buffer_occupancy=buffer_occupancy,
                details=dict(details or {}),
            )
        )

    def snapshot(self) -> ChannelTransportStats:
        snapshot = copy.deepcopy(self._stats)
        snapshot.last_sequence_by_source = dict(snapshot.last_sequence_by_source)
        return snapshot

    def drain_events(self) -> list[ChannelTransportEvent]:
        events = list(self._events)
        self._events.clear()
        return events

    def dispatch_pending_callbacks(self) -> None:
        if self._event_callback is None:
            self._pending_callbacks.clear()
            return
        while self._pending_callbacks:
            self._event_callback(self._pending_callbacks.popleft())


class ChannelSubscriber:
    def __init__(
        self,
        spec: ChannelSpec,
        *,
        host: str = "127.0.0.1",
        buffer_size: int | None = None,
        latest_only: bool | None = None,
        socket_timeout_ms: int = 1000,
        rcvhwm: int | None = None,
        transport_event_callback: Callable[[ChannelTransportEvent], None] | None = None,
    ):
        resolved_latest_only = (
            spec.default_latest_only if latest_only is None else latest_only
        )
        resolved_buffer_size = spec.default_buffer_size if buffer_size is None else buffer_size
        if resolved_latest_only:
            resolved_buffer_size = 1
        if resolved_buffer_size <= 0:
            raise ValueError("buffer_size must be positive.")

        self.spec = spec
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.RCVTIMEO, socket_timeout_ms)
        self.socket.setsockopt(zmq.RCVHWM, rcvhwm or spec.default_rcvhwm)
        self.connect_addr = f"tcp://{host}:{spec.port}"
        self.socket.connect(self.connect_addr)
        self.socket.subscribe(spec.topic)

        self._running = True
        self._buffer: collections.deque[ChannelMessage[Any]] = collections.deque(
            maxlen=resolved_buffer_size
        )
        self._buffer_capacity = resolved_buffer_size
        self._lock = threading.Lock()
        self._available = threading.Condition(self._lock)
        self._transport_tracker = ChannelTransportTracker(
            spec.name,
            event_callback=transport_event_callback,
        )
        self._receive_thread = threading.Thread(
            target=self._continuously_receive, daemon=True
        )
        self._receive_thread.start()

    def _continuously_receive(self) -> None:
        while self._running:
            try:
                should_dispatch_callbacks = False
                topic, envelope_bytes, payload_bytes = self.socket.recv_multipart()
                subscriber_received_timestamp_ns = time.time_ns()
                if topic != self.spec.topic:
                    continue

                envelope = ChannelEnvelope.from_dict(
                    json.loads(envelope_bytes.decode("utf-8"))
                )
                if envelope.channel != self.spec.name:
                    continue
                if envelope.encoding != self.spec.codec.encoding_name:
                    continue

                message = ChannelMessage(
                    envelope=envelope,
                    payload=self.spec.codec.decode(payload_bytes, envelope),
                    subscriber_received_timestamp_ns=subscriber_received_timestamp_ns,
                )
                with self._lock:
                    if len(self._buffer) == self._buffer_capacity and self._buffer:
                        dropped_message = self._buffer[0]
                        self._transport_tracker.note_local_overflow(
                            source=dropped_message.envelope.source,
                            dropped_sequence_id=dropped_message.envelope.sequence_id,
                            buffer_occupancy=self._buffer_capacity,
                            timestamp_ns=subscriber_received_timestamp_ns,
                        )
                    self._buffer.append(message)
                    self._transport_tracker.note_received(
                        envelope,
                        buffer_occupancy=len(self._buffer),
                        timestamp_ns=subscriber_received_timestamp_ns,
                    )
                    self._available.notify_all()
                    should_dispatch_callbacks = True
                if should_dispatch_callbacks:
                    self._transport_tracker.dispatch_pending_callbacks()
            except zmq.Again:
                continue
            except zmq.ContextTerminated:
                break
            except Exception:
                time.sleep(0.001)

    def receive(
        self, blocking: bool = True, timeout_sec: float | None = None
    ) -> ChannelMessage[Any] | None:
        with self._lock:
            if not blocking and not self._buffer:
                return None

            if blocking:
                deadline = (
                    None if timeout_sec is None else time.monotonic() + timeout_sec
                )
                while not self._buffer and self._running:
                    remaining = (
                        None
                        if deadline is None
                        else max(0.0, deadline - time.monotonic())
                    )
                    if not self._available.wait(timeout=remaining):
                        return None

            if not self._buffer:
                return None
            return self._buffer.popleft()

    def get_latest(
        self, before_timestamp_ns: int | None = None
    ) -> ChannelMessage[Any] | None:
        with self._lock:
            if not self._buffer:
                return None
            if before_timestamp_ns is None:
                return self._buffer[-1]

            for message in reversed(self._buffer):
                if message.envelope.message_timestamp_ns <= before_timestamp_ns:
                    return message
            return None

    def drain(self) -> list[ChannelMessage[Any]]:
        with self._lock:
            messages = list(self._buffer)
            self._buffer.clear()
            return messages

    def transport_stats_snapshot(self) -> ChannelTransportStats:
        with self._lock:
            return self._transport_tracker.snapshot()

    def drain_transport_events(self) -> list[ChannelTransportEvent]:
        with self._lock:
            return self._transport_tracker.drain_events()

    def close(self) -> None:
        self._running = False
        with self._lock:
            self._available.notify_all()

        if self._receive_thread.is_alive():
            self._receive_thread.join(timeout=2.0)

        try:
            if hasattr(self, "socket") and not self.socket.closed:
                self.socket.close()
            if hasattr(self, "context") and not self.context.closed:
                self.context.term()
        except Exception:
            pass
