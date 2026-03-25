from __future__ import annotations

from dataclasses import dataclass
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
