import zmq
import numpy as np
import pickle
import threading
import collections
import time

DEFAULT_VIPC_ZMQ_HOST_PUB = "127.0.0.1"  # Publisher binds to all interfaces
DEFAULT_VIPC_ZMQ_HOST_SUB = "127.0.0.1"  # Subscriber connects to localhost by default
DEFAULT_VIPC_ZMQ_PORT = "55550"
DEFAULT_VIPC_ZMQ_TOPIC = b"frames"  # Topic for frame data


class VisionIPCPublisher:
    """
    Publishes frames (NumPy arrays) over ZeroMQ.
    This acts as the single producer in the system.
    """

    def __init__(self, host=DEFAULT_VIPC_ZMQ_HOST_PUB, port=DEFAULT_VIPC_ZMQ_PORT):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.bind_addr = f"tcp://{host}:{port}"

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

    def publish_frame(self, frame: np.ndarray, topic=DEFAULT_VIPC_ZMQ_TOPIC):
        if not isinstance(frame, np.ndarray):
            print("Error: Frame to publish is not a NumPy array.")
            return

        metadata = {"dtype": str(frame.dtype), "shape": frame.shape}
        # Using a specific pickle protocol can be useful for cross-version compatibility
        metadata_bytes = pickle.dumps(metadata, protocol=pickle.HIGHEST_PROTOCOL)
        frame_bytes = frame.tobytes()

        try:
            self.socket.send_multipart([topic, metadata_bytes, frame_bytes])
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


class VisionIPCSubscriber:
    """
    Subscribes to frames published by VisionIPCPublisher.
    Maintains an internal buffer of recently received frames.
    """

    def __init__(
        self,
        host=DEFAULT_VIPC_ZMQ_HOST_SUB,
        port=DEFAULT_VIPC_ZMQ_PORT,
        topic=DEFAULT_VIPC_ZMQ_TOPIC,
        conflate=False,
        buffer_size=10,
        socket_timeout_ms=1000,
    ):  # Timeout for ZMQ socket receive

        if buffer_size <= 0:
            raise ValueError("buffer_size must be positive for VisionIPCSubscriber.")

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)

        if conflate:
            self.socket.setsockopt(zmq.CONFLATE, 1)

        self.socket.setsockopt(
            zmq.LINGER, 0
        )  # Don't wait for pending messages on close
        self.socket.setsockopt(zmq.RCVTIMEO, socket_timeout_ms)  # Timeout for recv

        self.connect_addr = f"tcp://{host}:{port}"
        self._topic_bytes = topic if isinstance(topic, bytes) else topic.encode("utf-8")

        try:
            self.socket.connect(self.connect_addr)
            self.socket.subscribe(self._topic_bytes)
            print(
                f"VisionIPCSubscriber connected to {self.connect_addr} on topic '{self._topic_bytes.decode()}'"
            )
        except zmq.ZMQError as e:
            print(
                f"Error connecting/subscribing subscriber socket to {self.connect_addr}: {e}"
            )
            self.context.term()
            raise

        self._buffer = collections.deque(maxlen=buffer_size)
        self._buffer_lock = threading.Lock()
        self._frame_available_condition = threading.Condition(self._buffer_lock)

        self._running = True
        self._receive_thread = threading.Thread(
            target=self._continuously_receive, daemon=True
        )
        self._receive_thread.start()

    def _deserialize_frame(self, metadata_bytes, frame_bytes):
        try:
            metadata = pickle.loads(metadata_bytes)
            dtype = np.dtype(metadata["dtype"])
            shape = metadata["shape"]
            if frame_bytes is None or not isinstance(frame_bytes, bytes):
                print("Error: frame_bytes is invalid during deserialization.")
                return None
            return np.frombuffer(frame_bytes, dtype=dtype).reshape(shape)
        except Exception as e:
            print(f"Error deserializing frame: {e}")
            return None

    def _continuously_receive(self):
        print(
            f"Subscriber receive thread started for topic '{self._topic_bytes.decode()}'."
        )
        while self._running:
            try:
                # RCVTIMEO on socket makes recv_multipart non-blocking indefinitely
                topic, metadata_bytes, frame_bytes = self.socket.recv_multipart()

                # We only subscribed to one topic, but good practice to check if more were added
                if topic == self._topic_bytes:
                    frame = self._deserialize_frame(metadata_bytes, frame_bytes)
                    if frame is not None:
                        with self._buffer_lock:
                            self._buffer.append(frame)
                            self._frame_available_condition.notify_all()
            except zmq.Again:  # Timeout (RCVTIMEO)
                continue  # Loop again to check self._running
            except zmq.ContextTerminated:
                print("Subscriber context terminated, stopping receive thread.")
                break
            except zmq.ZMQError as e:
                if self._running:  # Only log if not part of a normal shutdown
                    print(f"ZMQError in subscriber receive thread: {e}")
                break
            except Exception as e:
                if self._running:
                    print(f"Unexpected error in subscriber receive thread: {e}")
                    time.sleep(0.1)  # Avoid tight loop on unexpected persistent errors
        print(
            f"Subscriber receive thread stopped for topic '{self._topic_bytes.decode()}'."
        )

    def receive_frame(self, blocking=True, timeout_sec=None):
        """
        Retrieves a single frame (oldest) from the internal buffer.
        If blocking is True, waits until a frame is available or timeout occurs.
        """
        with self._buffer_lock:
            if not blocking and not self._buffer:
                return None

            if blocking:
                # Wait only if buffer is empty and thread is supposed to be running
                while not self._buffer and self._running:
                    wait_success = self._frame_available_condition.wait(
                        timeout=timeout_sec
                    )
                    if not wait_success and timeout_sec is not None:  # Timeout occurred
                        return None
                    # If woken up but still no buffer and not running, exit
                    if not self._running and not self._buffer:
                        return None

            if self._buffer:
                return self._buffer.popleft()
            return None

    def get_latest_frames(self, count=1):
        """
        Returns up to 'count' latest frames from the internal buffer.
        The frames are returned as a list, with the most recent frame at the end.
        """
        if count <= 0:
            return []
        with self._buffer_lock:
            num_to_get = min(count, len(self._buffer))
            if num_to_get == 0:
                return []
            # Convert deque to list and take the last 'num_to_get' elements
            return list(self._buffer)[-num_to_get:]

    def close(self):
        print(
            f"Attempting to close VisionIPCSubscriber for topic '{self._topic_bytes.decode()}'."
        )
        self._running = False

        with self._buffer_lock:
            self._frame_available_condition.notify_all()  # Wake up waiting threads

        if self._receive_thread and self._receive_thread.is_alive():
            print("Joining subscriber receive thread...")
            self._receive_thread.join(
                timeout=(
                    max(2.0, (int(self.socket.rcvtimeo) / 1000.0) * 2)
                    if hasattr(self.socket, "rcvtimeo")
                    else 2.0
                )
            )
            if self._receive_thread.is_alive():
                print("Warning: Subscriber receive thread did not terminate cleanly.")

        print("Closing subscriber ZMQ socket and context.")
        if hasattr(self, "socket") and not self.socket.closed:
            try:
                # Unsubscribe before closing if not done automatically
                # self.socket.unsubscribe(self._topic_bytes) # Usually not needed
                self.socket.close()
            except Exception as e:
                print(f"Exception during ZMQ socket close: {e}")

        if hasattr(self, "context") and not self.context.closed:
            try:
                self.context.term()
            except Exception as e:
                print(f"Exception during ZMQ context term: {e}")
        print(f"VisionIPCSubscriber for topic '{self._topic_bytes.decode()}' closed.")
