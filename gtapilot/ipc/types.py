from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, TypeVar

if TYPE_CHECKING:
    from gtapilot.ipc.codecs import ChannelCodec

T = TypeVar("T")


@dataclass(slots=True, frozen=True)
class ChannelSpec:
    name: str
    port: str
    topic: bytes
    codec: ChannelCodec
    default_buffer_size: int
    default_latest_only: bool
    default_sndhwm: int
    default_rcvhwm: int


@dataclass(slots=True)
class ChannelEnvelope:
    v: int
    channel: str
    encoding: str
    sequence_id: int
    message_timestamp_ns: int
    publish_timestamp_ns: int
    source: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "v": int(self.v),
            "channel": self.channel,
            "encoding": self.encoding,
            "sequence_id": int(self.sequence_id),
            "message_timestamp_ns": int(self.message_timestamp_ns),
            "publish_timestamp_ns": int(self.publish_timestamp_ns),
            "source": self.source,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ChannelEnvelope":
        return cls(
            v=int(payload["v"]),
            channel=str(payload["channel"]),
            encoding=str(payload["encoding"]),
            sequence_id=int(payload["sequence_id"]),
            message_timestamp_ns=int(payload["message_timestamp_ns"]),
            publish_timestamp_ns=int(payload["publish_timestamp_ns"]),
            source=str(payload["source"]),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(slots=True)
class ChannelMessage(Generic[T]):
    envelope: ChannelEnvelope
    payload: T
    subscriber_received_timestamp_ns: int | None = None


@dataclass(slots=True)
class ChannelTransportEvent:
    kind: str
    channel: str
    source: str
    timestamp_ns: int
    sequence_id_start: int | None = None
    sequence_id_end: int | None = None
    missing_count: int = 0
    dropped_count: int = 0
    buffer_occupancy: int | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "channel": self.channel,
            "source": self.source,
            "timestamp_ns": int(self.timestamp_ns),
            "sequence_id_start": (
                None
                if self.sequence_id_start is None
                else int(self.sequence_id_start)
            ),
            "sequence_id_end": (
                None if self.sequence_id_end is None else int(self.sequence_id_end)
            ),
            "missing_count": int(self.missing_count),
            "dropped_count": int(self.dropped_count),
            "buffer_occupancy": (
                None if self.buffer_occupancy is None else int(self.buffer_occupancy)
            ),
            "details": dict(self.details),
        }


@dataclass(slots=True)
class ChannelTransportStats:
    channel: str
    messages_received: int = 0
    sequence_gap_count: int = 0
    missing_message_count: int = 0
    local_overflow_count: int = 0
    local_overflow_dropped_messages: int = 0
    max_buffer_occupancy: int = 0
    last_sequence_by_source: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "messages_received": int(self.messages_received),
            "sequence_gap_count": int(self.sequence_gap_count),
            "missing_message_count": int(self.missing_message_count),
            "local_overflow_count": int(self.local_overflow_count),
            "local_overflow_dropped_messages": int(
                self.local_overflow_dropped_messages
            ),
            "max_buffer_occupancy": int(self.max_buffer_occupancy),
            "last_sequence_by_source": {
                str(source): int(sequence_id)
                for source, sequence_id in self.last_sequence_by_source.items()
            },
        }
