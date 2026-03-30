from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from multiprocessing import shared_memory


@dataclass(slots=True, frozen=True)
class SharedFrameDescriptor:
    shm_name: str
    slot_bytes: int
    slot_index: int
    slot_generation: int
    frame_bytes: int


class SharedFrameWriter:
    def __init__(self, *, channel_name: str, slot_count: int, slot_bytes: int):
        if slot_count <= 0:
            raise ValueError("slot_count must be positive.")
        if slot_bytes <= 0:
            raise ValueError("slot_bytes must be positive.")
        self.channel_name = str(channel_name)
        self.slot_count = int(slot_count)
        self.slot_bytes = int(slot_bytes)
        sanitized_channel = self.channel_name.replace(".", "_")
        self.shm_name = (
            f"gtapilot_{sanitized_channel}_{uuid.uuid4().hex}_{self.slot_bytes}"
        )
        self._shm = shared_memory.SharedMemory(
            name=self.shm_name,
            create=True,
            size=self.slot_count * self.slot_bytes,
        )
        self._slot_generations = [0] * self.slot_count
        self._next_slot_index = 0
        self._lock = threading.Lock()

    def write(self, frame_bytes: bytes | memoryview) -> SharedFrameDescriptor:
        view = memoryview(frame_bytes).cast("B")
        frame_size = len(view)
        if frame_size > self.slot_bytes:
            raise ValueError(
                f"Frame payload size {frame_size} exceeds slot capacity {self.slot_bytes}."
            )
        with self._lock:
            slot_index = self._next_slot_index
            self._next_slot_index = (self._next_slot_index + 1) % self.slot_count
            self._slot_generations[slot_index] += 1
            generation = self._slot_generations[slot_index]
            start = slot_index * self.slot_bytes
            end = start + frame_size
            self._shm.buf[start:end] = view
            return SharedFrameDescriptor(
                shm_name=self.shm_name,
                slot_bytes=self.slot_bytes,
                slot_index=slot_index,
                slot_generation=generation,
                frame_bytes=frame_size,
            )

    def close(self) -> None:
        try:
            self._shm.close()
        finally:
            try:
                self._shm.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                # On Windows the segment is removed automatically when the last
                # handle is closed.
                pass


class SharedFrameReaderCache:
    def __init__(self):
        self._lock = threading.Lock()
        self._segments: dict[str, shared_memory.SharedMemory] = {}

    def read(self, descriptor: SharedFrameDescriptor) -> memoryview:
        with self._lock:
            segment = self._segments.get(descriptor.shm_name)
            if segment is None:
                segment = shared_memory.SharedMemory(name=descriptor.shm_name)
                self._segments[descriptor.shm_name] = segment
            start = descriptor.slot_index * descriptor.slot_bytes
            end = start + descriptor.frame_bytes
            return segment.buf[start:end]

    def close(self) -> None:
        with self._lock:
            segments = list(self._segments.values())
            self._segments.clear()
        for segment in segments:
            try:
                segment.close()
            except Exception:
                pass


_reader_cache = SharedFrameReaderCache()


def read_shared_frame(descriptor: SharedFrameDescriptor) -> memoryview:
    return _reader_cache.read(descriptor)


def close_shared_frame_reader_cache() -> None:
    _reader_cache.close()
