import zmq
import pickle
import threading
import time
from typing import Any, Dict, Optional


DEFAULT_MSG_ZMQ_HOST_PUB = "*"  # Publisher binds to all interfaces
DEFAULT_MSG_ZMQ_HOST_SUB = "127.0.0.1"  # Subscriber connects to localhost by default
DEFAULT_MSG_ZMQ_PORT = "5556"


class Message:
    def __init__(self, topic: str, data: Any):
        self.topic = topic
        self.data = data


class PubMaster:
    def __init__(self, host=DEFAULT_MSG_ZMQ_HOST_PUB, port=DEFAULT_MSG_ZMQ_PORT):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(f"tcp://{host}:{port}")

    def publish(self, topic: str, message: Message):
        try:
            serialized = pickle.dumps(message)
            self.socket.send_multipart([topic.encode(), serialized])
        except (zmq.ZMQError, pickle.PickleError) as e:
            print(f"Error publishing message: {e}")

    def close(self):
        self.socket.close()
        self.context.term()


class SubMaster:
    def __init__(self, ports: Optional[list[str]] = None):
        if ports is None:
            ports = [DEFAULT_MSG_ZMQ_PORT]

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.topics: Dict[str, Optional[Message]] = {}
        self.callbacks: Dict[str, list] = {}
        self.running = True

        for port in ports:
            self.socket.connect(f"tcp://{DEFAULT_MSG_ZMQ_HOST_SUB}:{port}")

        self._start_receiver()

    def add_topic(self, topic: str, callback=None):
        self.topics[topic] = None
        self.socket.subscribe(topic.encode())

        if callback:
            if topic not in self.callbacks:
                self.callbacks[topic] = []
            self.callbacks[topic].append(callback)

    def _receive_loop(self):
        while self.running:
            try:
                topic_bytes, serialized = self.socket.recv_multipart(flags=zmq.NOBLOCK)
                topic = topic_bytes.decode()
                message = pickle.loads(serialized)

                if topic in self.topics:
                    self.topics[topic] = message

                    # Handle callbacks
                    if topic in self.callbacks:
                        for callback in self.callbacks[topic]:
                            callback(message)

            except zmq.Again:
                time.sleep(0.001)  # Prevent CPU spinning
            except Exception as e:
                print(f"Error in receive loop: {e}")

    def _start_receiver(self):
        self.receiver_thread = threading.Thread(target=self._receive_loop)
        self.receiver_thread.daemon = True
        self.receiver_thread.start()

    def close(self):
        self.running = False
        if hasattr(self, "receiver_thread"):
            self.receiver_thread.join(timeout=1.0)
        self.socket.close()
        self.context.term()


class MessagingSystem:
    def __init__(self, pub_port: str = "5556"):
        self.pub_master = PubMaster(pub_port)
        self.sub_master = SubMaster([pub_port])

    def publish(self, topic: str, data: Any):
        message = Message(topic, data)
        self.pub_master.publish(topic, message)

    def subscribe(self, topic: str, callback):
        self.sub_master.add_topic(topic, callback)

    def close(self):
        self.pub_master.close()
        self.sub_master.close()
