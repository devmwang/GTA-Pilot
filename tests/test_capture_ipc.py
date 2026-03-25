from __future__ import annotations

import json
import tarfile
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import zmq

from gtapilot.blackbox.blackbox import BlackboxRecorder
from gtapilot.ipc.channel import ChannelPublisher, ChannelSubscriber
from gtapilot.ipc.channels import (
    INPUT_ACTIONS_CHANNEL,
    VISION_FRAMES_CHANNEL,
    ActionPacket,
)


def _vision_spec(port: str):
    return replace(VISION_FRAMES_CHANNEL, port=port)


def _action_spec(port: str):
    return replace(INPUT_ACTIONS_CHANNEL, port=port)


def test_raw_rgb_generic_channel_roundtrip_metadata() -> None:
    spec = _vision_spec("55650")
    publisher = ChannelPublisher(spec, source_name="test_vision_source")
    subscriber = ChannelSubscriber(spec, latest_only=True)

    try:
        time.sleep(0.2)
        capture_timestamp_ns = 123456789
        frame = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
        publisher.publish(
            frame,
            timestamp_ns=capture_timestamp_ns,
            metadata={
                "frame_id": 42,
                "capture_timestamp_ns": capture_timestamp_ns,
                "is_repeat": False,
            },
        )

        message = None
        deadline = time.time() + 2.0
        while time.time() < deadline:
            message = subscriber.get_latest()
            if message is not None:
                break
            time.sleep(0.05)
        assert message is not None
        assert message.payload.shape == (4, 5, 3)
        assert message.envelope.v == 1
        assert message.envelope.channel == "vision.frames"
        assert message.envelope.encoding == "raw_rgb_v1"
        assert message.envelope.source == "test_vision_source"
        assert message.envelope.message_timestamp_ns == capture_timestamp_ns
        assert message.envelope.metadata["frame_id"] == 42
        assert message.envelope.metadata["capture_timestamp_ns"] == capture_timestamp_ns
        assert message.envelope.metadata["w"] == 5
        assert message.envelope.metadata["h"] == 4
        assert message.envelope.metadata["dtype"] == "uint8"
    finally:
        subscriber.close()
        publisher.close()


def test_latest_only_behavior_on_vision_channel() -> None:
    spec = _vision_spec("55651")
    publisher = ChannelPublisher(spec, source_name="test_vision_latest")
    subscriber = ChannelSubscriber(spec, latest_only=True)

    try:
        time.sleep(0.2)
        for frame_id in range(1, 4):
            publisher.publish(
                np.full((2, 3, 3), frame_id, dtype=np.uint8),
                timestamp_ns=1000 + frame_id,
                metadata={
                    "frame_id": frame_id,
                    "capture_timestamp_ns": 1000 + frame_id,
                    "is_repeat": False,
                },
            )
            time.sleep(0.03)

        latest = subscriber.get_latest()
        assert latest is not None
        assert latest.envelope.metadata["frame_id"] == 3
        drained = subscriber.drain()
        assert len(drained) == 1
        assert drained[0].envelope.metadata["frame_id"] == 3
    finally:
        subscriber.close()
        publisher.close()


def test_action_json_roundtrip_and_drain() -> None:
    spec = _action_spec("55652")
    publisher = ChannelPublisher(spec, source_name="test_action_source")
    subscriber = ChannelSubscriber(spec)

    try:
        time.sleep(0.2)
        publisher.publish(
            ActionPacket(
                steer=-1.0,
                throttle=1.0,
                brake=0.0,
                handbrake=0.0,
                reverse=0.0,
                pilot_active=0.0,
                raw_inputs={"w": True, "a": True},
            ),
            timestamp_ns=999,
        )

        message = subscriber.receive(blocking=True, timeout_sec=2.0)
        assert message is not None
        assert message.envelope.source == "test_action_source"
        assert message.envelope.message_timestamp_ns == 999
        assert message.payload.vector == [-1.0, 1.0, 0.0, 0.0, 0.0, 0.0]

        time.sleep(0.1)
        publisher.publish(
            ActionPacket(
                steer=0.0,
                throttle=0.0,
                brake=1.0,
                handbrake=0.0,
                reverse=0.0,
                pilot_active=1.0,
                raw_inputs={"s": True},
            ),
            timestamp_ns=1001,
        )
        time.sleep(0.1)
        publisher.publish(
            ActionPacket(
                steer=1.0,
                throttle=0.0,
                brake=0.0,
                handbrake=1.0,
                reverse=0.0,
                pilot_active=1.0,
                raw_inputs={"d": True, "space": True},
            ),
            timestamp_ns=1002,
        )
        time.sleep(0.3)

        drained = subscriber.drain()
        assert [message.envelope.message_timestamp_ns for message in drained] == [1001, 1002]
        assert drained[-1].payload.raw_inputs["space"] is True
    finally:
        subscriber.close()
        publisher.close()


def test_get_latest_before_timestamp_alignment() -> None:
    spec = _action_spec("55653")
    publisher = ChannelPublisher(spec, source_name="test_action_alignment")
    subscriber = ChannelSubscriber(spec)

    try:
        time.sleep(0.2)
        for timestamp_ns, steer in ((1000, -1.0), (2000, 0.0), (3000, 1.0)):
            publisher.publish(
                ActionPacket(
                    steer=steer,
                    throttle=0.0,
                    brake=0.0,
                    handbrake=0.0,
                    reverse=0.0,
                    pilot_active=0.0,
                    raw_inputs={},
                ),
                timestamp_ns=timestamp_ns,
            )
            time.sleep(0.03)

        time.sleep(0.2)
        aligned = subscriber.get_latest(before_timestamp_ns=2500)
        assert aligned is not None
        assert aligned.envelope.message_timestamp_ns == 2000
        assert subscriber.get_latest(before_timestamp_ns=500) is None
    finally:
        subscriber.close()
        publisher.close()


def test_blackbox_records_generic_channels() -> None:
    vision_spec = _vision_spec("55654")
    action_spec = _action_spec("55655")
    vision_publisher = ChannelPublisher(vision_spec, source_name="test_blackbox_frames")
    action_publisher = ChannelPublisher(action_spec, source_name="test_blackbox_actions")

    with tempfile.TemporaryDirectory() as tmp_dir:
        recorder = BlackboxRecorder(
            output_dir=tmp_dir,
            vision_channel=vision_spec,
            action_channel=action_spec,
        )
        try:
            time.sleep(0.2)
            action_publisher.publish(
                ActionPacket(
                    steer=0.25,
                    throttle=1.0,
                    brake=0.0,
                    handbrake=0.0,
                    reverse=0.0,
                    pilot_active=0.0,
                    raw_inputs={"w": True},
                ),
                timestamp_ns=1000,
            )
            time.sleep(0.05)
            vision_publisher.publish(
                np.full((8, 8, 3), 127, dtype=np.uint8),
                timestamp_ns=1100,
                metadata={
                    "frame_id": 1,
                    "capture_timestamp_ns": 1100,
                    "is_repeat": False,
                },
            )
            assert recorder.record_next_frame(timeout_sec=2.0) is True
        finally:
            recorder.close()
            action_publisher.close()
            vision_publisher.close()

        output_dir = Path(tmp_dir)
        tar_paths = sorted(output_dir.glob("capture_*_frames.tar"))
        metadata_paths = sorted(output_dir.glob("capture_*_metadata.json"))
        assert len(tar_paths) == 1
        assert len(metadata_paths) == 1

        manifest = json.loads(metadata_paths[0].read_text(encoding="utf-8"))
        assert manifest["schema_version"] == 3
        assert manifest["frame_count"] == 1
        assert manifest["action_count"] == 1
        assert len(manifest["frames"]) == 1
        assert len(manifest["actions"]) == 1
        assert manifest["frames"][0]["action"]["throttle"] == 1.0
        assert manifest["frames"][0]["action_vector"] == [0.25, 1.0, 0.0, 0.0, 0.0, 0.0]
        assert manifest["actions"][0]["payload"]["raw_inputs"]["w"] is True

        with tarfile.open(tar_paths[0], "r") as archive:
            names = archive.getnames()
        assert names == ["frame_000001.bmp"]


def test_native_envelope_contract_decodes_without_adapter() -> None:
    spec = _vision_spec("55666")
    subscriber = ChannelSubscriber(spec, latest_only=True)
    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    publisher.setsockopt(zmq.LINGER, 0)
    publisher.bind(f"tcp://127.0.0.1:{spec.port}")

    try:
        time.sleep(0.5)
        frame = np.arange(3 * 4 * 3, dtype=np.uint8).reshape(3, 4, 3)
        envelope = {
            "v": 1,
            "channel": "vision.frames",
            "encoding": "raw_rgb_v1",
            "sequence_id": 7,
            "message_timestamp_ns": 777,
            "publish_timestamp_ns": 888,
            "source": "display_capture_dx11",
            "metadata": {
                "w": 4,
                "h": 3,
                "channels": 3,
                "dtype": "uint8",
                "frame_id": 7,
                "capture_timestamp_ns": 777,
                "is_repeat": False,
            },
        }
        for _ in range(10):
            publisher.send_multipart(
                [
                    spec.topic,
                    json.dumps(envelope).encode("utf-8"),
                    memoryview(frame).cast("B"),
                ]
            )
            time.sleep(0.1)

        message = None
        deadline = time.time() + 2.0
        while time.time() < deadline:
            message = subscriber.get_latest()
            if message is not None:
                break
            time.sleep(0.05)
        assert message is not None
        assert message.envelope.source == "display_capture_dx11"
        assert message.envelope.sequence_id == 7
        assert message.payload.shape == (3, 4, 3)
        assert int(message.payload[2, 3, 2]) == int(frame[2, 3, 2])
    finally:
        publisher.close()
        context.term()
        subscriber.close()
