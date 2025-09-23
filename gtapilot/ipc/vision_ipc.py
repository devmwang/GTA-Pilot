from collections import deque
from dataclasses import dataclass
import json
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np
import zmq

DEFAULT_VIPC_ZMQ_HOST_PUB = "127.0.0.1"  # Publisher binds to all interfaces
DEFAULT_VIPC_ZMQ_HOST_SUB = "127.0.0.1"  # Subscriber connects to localhost by default
DEFAULT_VIPC_ZMQ_PORT = "55550"
DEFAULT_VIPC_ZMQ_TOPIC_CPU = b"frames_cpu"  # Topic for CPU frame metadata & frame data
DEFAULT_VIPC_ZMQ_TOPIC_GPU = b"frames_gpu"  # Topic for GPU frame metadata


class VisionIpcPublisher:
    """
    Publishes JPEG frames (NumPy arrays) over ZeroMQ.
    This acts as the single producer in the system.
    """

    def __init__(
        self,
        host=DEFAULT_VIPC_ZMQ_HOST_PUB,
        port=DEFAULT_VIPC_ZMQ_PORT,
        topic=DEFAULT_VIPC_ZMQ_TOPIC_CPU,
        resize_to: Optional[Tuple[int, int]] = (1920, 1080),
        sndhwm: int = 1,  # Drop excess when subscriber is slow
    ):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.SNDHWM, sndhwm)
        self.bind_addr = f"tcp://{host}:{port}"
        self._topic_bytes = topic if isinstance(topic, bytes) else topic.encode("utf-8")
        self.resize_to = resize_to
        self._frame_id = 1

        try:
            self.socket.bind(self.bind_addr)
            # Allow time for connection to establish
            time.sleep(0.1)
            print(f"VisionIPCPublisher bound to {self.bind_addr}")

        except zmq.ZMQError as e:
            print(f"Error binding publisher socket to {self.bind_addr}: {e}")
            # Clean up context if bind fails in constructor
            self.context.term()
            raise

    def publish_frame(self, frame: np.ndarray):
        if frame.ndim != 3 or frame.shape[2] != 3:
            print("Error: Frame to publish is not a RGB np.uint8 HxWx3 array.")
            return

        if self.resize_to is not None:
            w, h = int(self.resize_to[0]), int(self.resize_to[1])
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            h, w = frame.shape[:2]

        # Ensure contiguous RGB uint8
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8, copy=False)

        if not frame.flags["C_CONTIGUOUS"]:
            frame = np.ascontiguousarray(frame)

        # metadata = {"dtype": str(frame.dtype), "shape": frame.shape, "frame_id": self.frame_id}
        metadata = {
            "encoding": "raw",
            "w": int(w),
            "h": int(h),
            "channels": 3,
            "dtype": "uint8",
            "frame_id": int(self._frame_id),
        }
        self._frame_id += 1

        metadata_bytes = json.dumps(metadata).encode("utf-8")

        try:
            self.socket.send_multipart(
                [self._topic_bytes, metadata_bytes, memoryview(frame).cast("B")]
            )

        except zmq.ZMQError as e:
            print(f"Error sending frame: {e}")

        except Exception as e:
            print(f"Unexpected error publishing frame: {e}")

    def close(self):
        print("Closing VisionIPCPublisher.")

        if hasattr(self, "socket") and not self.socket.closed:
            self.socket.close()

        if hasattr(self, "context") and not self.context.closed:
            self.context.term()

        print("VisionIPCPublisher closed.")


class VisionIpcCpuSubscriber:
    """
    CPU subscriber for frames published by the native D3D11 capture or VisionIPCPublisher (display override).
    Topic payload is [topic, json_meta, raw_bytes].
      meta = {"encoding":"raw","w":1920,"h":1080,"channels":3,"dtype":"uint8","frame_id":<int>}
    Output frames are RGB np.uint8 HxWx3
    """

    def __init__(
        self,
        host=DEFAULT_VIPC_ZMQ_HOST_SUB,
        port=DEFAULT_VIPC_ZMQ_PORT,
        topic=DEFAULT_VIPC_ZMQ_TOPIC_CPU,
        buffer_size=10,
        socket_timeout_ms=1000,
        rcvhwm=1,
        latestOnly=False,
    ):
        if buffer_size <= 0:
            raise ValueError("buffer_size must be positive.")
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.RCVTIMEO, socket_timeout_ms)
        self.socket.setsockopt(zmq.RCVHWM, rcvhwm)
        if latestOnly:
            # Can't use ZMQ_CONFLATE with multipart messages; it corrupts
            # message boundaries and trigger libzmq assertions.
            # Emulate "latest only" behavior via a maxlen=1 application buffer.
            buffer_size = 1
        self.connect_addr = f"tcp://{host}:{port}"
        self._topic_bytes = topic if isinstance(topic, bytes) else topic.encode("utf-8")

        self.socket.connect(self.connect_addr)
        self.socket.subscribe(self._topic_bytes)

        self._buffer = deque(maxlen=buffer_size)
        self._buffer_lock = threading.Lock()
        self._frame_available_condition = threading.Condition(self._buffer_lock)

        self._running = True
        self._receive_thread = threading.Thread(
            target=self._continuously_receive, daemon=True
        )
        self._receive_thread.start()

    def _continuously_receive(self):
        while self._running:
            # print("Running")
            try:
                topic, metadata_bytes, raw_bytes = self.socket.recv_multipart()

                if topic != self._topic_bytes:
                    continue

                metadata = json.loads(metadata_bytes.decode("utf-8"))
                w, h = int(metadata["w"]), int(metadata["h"])
                arr = np.frombuffer(raw_bytes, dtype=np.uint8)

                if arr is None or arr.size != w * h * 3:
                    continue

                rgb = arr.reshape(h, w, 3)

                with self._buffer_lock:
                    self._buffer.append(rgb)
                    self._frame_available_condition.notify_all()

            except zmq.Again:
                continue

            except zmq.ContextTerminated:
                break

            except Exception:
                time.sleep(0.001)
                continue

    def receive_frame(self, blocking=True, timeout_sec=None):
        with self._buffer_lock:
            if not blocking and not self._buffer:
                return None

            if blocking:
                end = None if timeout_sec is None else time.time() + timeout_sec

                while not self._buffer and self._running:
                    rem = None if end is None else max(0.0, end - time.time())

                    if not self._frame_available_condition.wait(timeout=rem):
                        return None

            if self._buffer:
                return self._buffer.popleft()

            return None

    def close(self):
        self._running = False

        with self._buffer_lock:
            self._frame_available_condition.notify_all()

        if self._receive_thread and self._receive_thread.is_alive():
            self._receive_thread.join(timeout=2.0)

        try:
            if hasattr(self, "socket") and not self.socket.closed:
                self.socket.close()

            if hasattr(self, "context") and not self.context.closed:
                self.context.term()

        except Exception:
            pass


class VisionIpcGpuSubscriber:
    """
    GPU subscriber for frame metadata published by the native D3D11 capture or VisionIPCPublisher (display override).
    Topic payload is [topic, json_meta].
      meta = {"slot":<int>, "handle":<uint64_t>, "w":<int>, "h":<int>, "format":<str>, "frame_id":<int>, "qpc":<uint64_t>}
    Frame data is stored in GPU memory to avoid unnecessary GPU->CPU->GPU copies.
    """

    def __init__(
        self,
        host=DEFAULT_VIPC_ZMQ_HOST_SUB,
        port=DEFAULT_VIPC_ZMQ_PORT,
        topic=DEFAULT_VIPC_ZMQ_TOPIC_GPU,
        buffer_size=10,
        socket_timeout_ms=1000,
        rcvhwm=1,
        latestOnly=False,
    ):
        if buffer_size <= 0:
            raise ValueError("buffer_size must be positive.")
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.RCVTIMEO, socket_timeout_ms)
        self.socket.setsockopt(zmq.RCVHWM, rcvhwm)
        if latestOnly:
            # Can't use ZMQ_CONFLATE with multipart messages; it corrupts
            # message boundaries and trigger libzmq assertions.
            # Emulate "latest only" behavior via a maxlen=1 application buffer.
            buffer_size = 1
        self.connect_addr = f"tcp://{host}:{port}"
        self._topic_bytes = topic if isinstance(topic, bytes) else topic.encode("utf-8")

        self.socket.connect(self.connect_addr)
        self.socket.subscribe(self._topic_bytes)

        self._buffer: deque[GpuFrameMetadata] = deque(maxlen=buffer_size)
        self._buffer_lock = threading.Lock()
        self._data_available_condition = threading.Condition(self._buffer_lock)

        self._running = True
        self._receive_thread = threading.Thread(
            target=self._continuously_receive, daemon=True
        )
        self._receive_thread.start()

    def _continuously_receive(self):
        while self._running:
            # print("Running")
            try:
                topic, metadata_bytes = self.socket.recv_multipart()

                if topic != self._topic_bytes:
                    continue

                metadata = json.loads(metadata_bytes.decode("utf-8"))

                latest_data = GpuFrameMetadata(
                    slot=int(metadata["slot"]),
                    handle=int(metadata["handle"]),
                    w=int(metadata["w"]),
                    h=int(metadata["h"]),
                    frame_id=int(metadata["frame_id"]),
                )

                with self._buffer_lock:
                    self._buffer.append(latest_data)
                    self._data_available_condition.notify_all()

            except zmq.Again:
                continue

            except zmq.ContextTerminated:
                break

            except Exception:
                time.sleep(0.001)
                continue

    def receive_frame_metadata(self, blocking=True, timeout_sec=None):
        with self._buffer_lock:
            if not blocking and not self._buffer:
                return None

            if blocking:
                end = None if timeout_sec is None else time.time() + timeout_sec

                while not self._buffer and self._running:
                    rem = None if end is None else max(0.0, end - time.time())

                    if not self._data_available_condition.wait(timeout=rem):
                        return None

            if self._buffer:
                return self._buffer.popleft()

            return None

    def close(self):
        self._running = False

        with self._buffer_lock:
            self._data_available_condition.notify_all()

        if self._receive_thread and self._receive_thread.is_alive():
            self._receive_thread.join(timeout=2.0)

        try:
            if hasattr(self, "socket") and not self.socket.closed:
                self.socket.close()

            if hasattr(self, "context") and not self.context.closed:
                self.context.term()

        except Exception:
            pass


@dataclass
class GpuFrameMetadata:
    slot: int
    handle: int
    w: int
    h: int
    frame_id: int
