from __future__ import annotations

import json
from dataclasses import dataclass, field, is_dataclass
from typing import Any, Generic, Protocol, TypeVar

import numpy as np

from gtapilot.ipc.shared_frames import (
    SharedFrameDescriptor,
    SharedFrameWriter,
    close_shared_frame_reader_cache,
    read_shared_frame,
)
from gtapilot.ipc.types import ChannelEnvelope

T = TypeVar("T")


class ChannelCodec(Protocol):
    encoding_name: str

    def encode(
        self, payload: Any, metadata: dict[str, Any] | None = None
    ) -> tuple[bytes | memoryview, dict[str, Any]]: ...

    def decode(self, payload_bytes: bytes, envelope: ChannelEnvelope) -> Any: ...


@dataclass(slots=True)
class SharedMemoryFrameCodec:
    channel_name: str
    slot_count: int = 8
    encoding_name: str = "shm_rgb_v1"
    _writer: SharedFrameWriter | None = field(default=None, init=False, repr=False)
    _slot_bytes: int | None = field(default=None, init=False, repr=False)

    def _ensure_writer(self, frame_nbytes: int) -> SharedFrameWriter:
        required_slot_bytes = int(frame_nbytes)
        if (
            self._writer is None
            or self._slot_bytes is None
            or self._slot_bytes < required_slot_bytes
        ):
            if self._writer is not None:
                self._writer.close()
            self._slot_bytes = required_slot_bytes
            self._writer = SharedFrameWriter(
                channel_name=self.channel_name,
                slot_count=self.slot_count,
                slot_bytes=required_slot_bytes,
            )
        return self._writer

    def encode(
        self, payload: Any, metadata: dict[str, Any] | None = None
    ) -> tuple[bytes, dict[str, Any]]:
        if not isinstance(payload, np.ndarray):
            raise TypeError("SharedMemoryFrameCodec expects a numpy.ndarray payload.")

        frame = payload
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                "SharedMemoryFrameCodec expects an RGB uint8 HxWx3 frame."
            )
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8, copy=False)
        if not frame.flags["C_CONTIGUOUS"]:
            frame = np.ascontiguousarray(frame)

        writer = self._ensure_writer(frame.nbytes)
        descriptor = writer.write(memoryview(frame).cast("B"))

        height, width, channels = frame.shape
        envelope_metadata = dict(metadata or {})
        envelope_metadata.update(
            {
                "w": int(width),
                "h": int(height),
                "channels": int(channels),
                "dtype": "uint8",
                "shm_name": descriptor.shm_name,
                "slot_bytes": int(descriptor.slot_bytes),
                "slot_index": int(descriptor.slot_index),
                "slot_generation": int(descriptor.slot_generation),
                "frame_bytes": int(descriptor.frame_bytes),
            }
        )
        return b"", envelope_metadata

    def decode(self, payload_bytes: bytes, envelope: ChannelEnvelope) -> Any:
        metadata = envelope.metadata
        width = int(metadata["w"])
        height = int(metadata["h"])
        channels = int(metadata["channels"])
        dtype_name = str(metadata["dtype"])
        if dtype_name != "uint8":
            raise ValueError(f"Unsupported shared frame dtype '{dtype_name}'.")
        if channels != 3:
            raise ValueError(f"Unsupported shared frame channel count '{channels}'.")
        descriptor = SharedFrameDescriptor(
            shm_name=str(metadata["shm_name"]),
            slot_bytes=int(metadata["slot_bytes"]),
            slot_index=int(metadata["slot_index"]),
            slot_generation=int(metadata.get("slot_generation", 0)),
            frame_bytes=int(metadata["frame_bytes"]),
        )
        arr = np.frombuffer(read_shared_frame(descriptor), dtype=np.uint8).copy()
        expected = width * height * channels
        if arr.size != expected:
            raise ValueError(
                f"Frame payload size mismatch: expected {expected}, got {arr.size}."
            )
        return arr.reshape(height, width, channels)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        self._slot_bytes = None
        close_shared_frame_reader_cache()


class JsonDataclassCodec(Generic[T]):
    encoding_name = "json_dataclass_v1"

    def __init__(self, payload_type: type[T]):
        self.payload_type = payload_type

    def encode(
        self, payload: Any, metadata: dict[str, Any] | None = None
    ) -> tuple[bytes, dict[str, Any]]:
        if hasattr(payload, "to_dict"):
            payload_dict = payload.to_dict()
        elif is_dataclass(payload):
            payload_dict = dict(payload.__dict__)
        else:
            raise TypeError(
                f"{self.payload_type.__name__} payload must provide to_dict()."
            )
        return json.dumps(payload_dict).encode("utf-8"), dict(metadata or {})

    def decode(self, payload_bytes: bytes, envelope: ChannelEnvelope) -> T:
        payload_dict = json.loads(payload_bytes.decode("utf-8"))
        if not hasattr(self.payload_type, "from_dict"):
            raise TypeError(
                f"{self.payload_type.__name__} payload type must provide from_dict()."
            )
        return self.payload_type.from_dict(payload_dict)
