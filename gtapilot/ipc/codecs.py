from __future__ import annotations

import json
from dataclasses import is_dataclass
from typing import Any, Generic, Protocol, TypeVar

import numpy as np

from gtapilot.ipc.types import ChannelEnvelope

T = TypeVar("T")


class ChannelCodec(Protocol):
    encoding_name: str

    def encode(
        self, payload: Any, metadata: dict[str, Any] | None = None
    ) -> tuple[bytes | memoryview, dict[str, Any]]: ...

    def decode(self, payload_bytes: bytes, envelope: ChannelEnvelope) -> Any: ...


class RawRGBFrameCodec:
    encoding_name = "raw_rgb_v1"

    def encode(
        self, payload: Any, metadata: dict[str, Any] | None = None
    ) -> tuple[bytes | memoryview, dict[str, Any]]:
        if not isinstance(payload, np.ndarray):
            raise TypeError("RawRGBFrameCodec expects a numpy.ndarray payload.")

        frame = payload
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("RawRGBFrameCodec expects an RGB uint8 HxWx3 frame.")

        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8, copy=False)
        if not frame.flags["C_CONTIGUOUS"]:
            frame = np.ascontiguousarray(frame)

        height, width, channels = frame.shape
        envelope_metadata = dict(metadata or {})
        envelope_metadata.update(
            {
                "w": int(width),
                "h": int(height),
                "channels": int(channels),
                "dtype": "uint8",
            }
        )
        return memoryview(frame).cast("B"), envelope_metadata

    def decode(self, payload_bytes: bytes, envelope: ChannelEnvelope) -> np.ndarray:
        metadata = envelope.metadata
        width = int(metadata["w"])
        height = int(metadata["h"])
        channels = int(metadata["channels"])
        dtype_name = str(metadata["dtype"])
        if dtype_name != "uint8":
            raise ValueError(f"Unsupported raw RGB dtype '{dtype_name}'.")
        if channels != 3:
            raise ValueError(f"Unsupported raw RGB channel count '{channels}'.")

        arr = np.frombuffer(payload_bytes, dtype=np.uint8)
        expected = width * height * channels
        if arr.size != expected:
            raise ValueError(
                f"Frame payload size mismatch: expected {expected}, got {arr.size}."
            )
        return arr.reshape(height, width, channels)


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
