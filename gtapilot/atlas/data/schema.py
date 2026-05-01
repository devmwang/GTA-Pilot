from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class LaneSegment3D:
    lane_id: int
    centerline: np.ndarray
    left_boundary: np.ndarray
    right_boundary: np.ndarray
    lane_semantic: str
    direction: str
    left_kind: str
    left_color: str
    left_pattern: str
    left_continuity: str
    right_kind: str
    right_color: str
    right_pattern: str
    right_continuity: str
    predecessors: list[int] = field(default_factory=list)
    successors: list[int] = field(default_factory=list)
    left_adjacent: int | None = None
    right_adjacent: int | None = None
    traffic_element_ids: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("centerline", "left_boundary", "right_boundary"):
            payload[key] = np.asarray(payload[key]).tolist()
        return payload


@dataclass
class MapElement3D:
    elem_id: int
    elem_type: str
    polyline: np.ndarray
    bound_lane_ids: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["polyline"] = np.asarray(payload["polyline"]).tolist()
        return payload


@dataclass
class ConvertedLaneSample:
    lane_segments: list[LaneSegment3D]
    map_elements: list[MapElement3D]
    valid_mask: dict[str, bool]
    source_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BlackboxFrameRecord:
    sequence_id: int
    video_frame_index: int
    capture_timestamp_ns: int
    publish_timestamp_ns: int
    frame_id: int
    capture_frame_id: int
    is_repeat: bool
    subscriber_received_timestamp_ns: int | None = None
    writer_committed_timestamp_ns: int | None = None
    subscriber_queue_latency_ns: int | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BlackboxFrameRecord":
        return cls(
            sequence_id=int(payload.get("sequence_id", -1)),
            video_frame_index=int(payload["video_frame_index"]),
            capture_timestamp_ns=int(payload["capture_timestamp_ns"]),
            publish_timestamp_ns=int(payload["publish_timestamp_ns"]),
            frame_id=int(payload.get("frame_id", -1)),
            capture_frame_id=int(
                payload.get("capture_frame_id", payload.get("frame_id", -1))
            ),
            is_repeat=bool(payload.get("is_repeat", False)),
            subscriber_received_timestamp_ns=None
            if payload.get("subscriber_received_timestamp_ns") is None
            else int(payload["subscriber_received_timestamp_ns"]),
            writer_committed_timestamp_ns=None
            if payload.get("writer_committed_timestamp_ns") is None
            else int(payload["writer_committed_timestamp_ns"]),
            subscriber_queue_latency_ns=None
            if payload.get("subscriber_queue_latency_ns") is None
            else int(payload["subscriber_queue_latency_ns"]),
        )


@dataclass
class BlackboxFrameActionRecord:
    video_frame_index: int
    capture_timestamp_ns: int
    frame_id: int
    capture_frame_id: int
    action_vector: np.ndarray
    action_message_timestamp_ns: int | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BlackboxFrameActionRecord":
        return cls(
            video_frame_index=int(payload["video_frame_index"]),
            capture_timestamp_ns=int(payload["capture_timestamp_ns"]),
            frame_id=int(payload.get("frame_id", -1)),
            capture_frame_id=int(
                payload.get("capture_frame_id", payload.get("frame_id", -1))
            ),
            action_vector=np.asarray(payload["action_vector"], dtype=np.float32),
            action_message_timestamp_ns=(
                None
                if payload.get("action_message_timestamp_ns") is None
                else int(payload["action_message_timestamp_ns"])
            ),
        )


@dataclass
class BlackboxActionRecord:
    sequence_id: int
    message_timestamp_ns: int
    publish_timestamp_ns: int
    payload: dict[str, Any]
    subscriber_received_timestamp_ns: int | None = None
    subscriber_queue_latency_ns: int | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BlackboxActionRecord":
        raw_payload = payload.get("payload")
        return cls(
            sequence_id=int(payload.get("sequence_id", -1)),
            message_timestamp_ns=int(payload.get("message_timestamp_ns", 0)),
            publish_timestamp_ns=int(payload.get("publish_timestamp_ns", 0)),
            payload={} if raw_payload is None else dict(raw_payload),
            subscriber_received_timestamp_ns=(
                None
                if payload.get("subscriber_received_timestamp_ns") is None
                else int(payload["subscriber_received_timestamp_ns"])
            ),
            subscriber_queue_latency_ns=(
                None
                if payload.get("subscriber_queue_latency_ns") is None
                else int(payload["subscriber_queue_latency_ns"])
            ),
        )

    @property
    def timestamp_ns(self) -> int:
        return self.message_timestamp_ns

    @property
    def action_vector(self) -> np.ndarray:
        return np.asarray(
            [
                self.payload.get("steer", 0.0),
                self.payload.get("throttle", 0.0),
                self.payload.get("brake", 0.0),
                self.payload.get("handbrake", 0.0),
                self.payload.get("reverse", 0.0),
                self.payload.get("pilot_active", 0.0),
            ],
            dtype=np.float32,
        )


@dataclass
class AtlasTemporalClipIndex:
    clip_id: str
    metadata_path: str
    actions_path: str
    video_path: str
    anchor_timestamp_ns: int
    action_source: str
    nominal_fps: float
    frame_source: str
    privileged_dir: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AtlasTemporalClipIndex":
        return cls(
            clip_id=str(payload["clip_id"]),
            metadata_path=str(payload["metadata_path"]),
            actions_path=str(payload["actions_path"]),
            video_path=str(payload["video_path"]),
            anchor_timestamp_ns=int(payload["anchor_timestamp_ns"]),
            action_source=str(payload["action_source"]),
            nominal_fps=float(payload["nominal_fps"]),
            frame_source=str(payload["frame_source"]),
            privileged_dir=(
                None if payload.get("privileged_dir") is None else str(payload["privileged_dir"])
            ),
        )

    @property
    def metadata_file(self) -> Path:
        return Path(self.metadata_path)

    @property
    def video_file(self) -> Path:
        return Path(self.video_path)

    @property
    def actions_file(self) -> Path:
        return Path(self.actions_path)

    @property
    def privileged_path(self) -> Path | None:
        return None if self.privileged_dir is None else Path(self.privileged_dir)
