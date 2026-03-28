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
    video_frame_index: int
    capture_timestamp_ns: int
    publish_timestamp_ns: int
    frame_id: int
    capture_frame_id: int
    frame_source: str
    frame_metadata: dict[str, Any]
    action: dict[str, Any] | None
    action_vector: np.ndarray
    action_message_timestamp_ns: int | None = None
    subscriber_received_timestamp_ns: int | None = None
    writer_committed_timestamp_ns: int | None = None
    subscriber_queue_latency_ns: int | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BlackboxFrameRecord":
        action_envelope = payload.get("action_envelope") or {}
        return cls(
            video_frame_index=int(payload["video_frame_index"]),
            capture_timestamp_ns=int(payload["capture_timestamp_ns"]),
            publish_timestamp_ns=int(payload["publish_timestamp_ns"]),
            frame_id=int(payload.get("frame_id", -1)),
            capture_frame_id=int(
                payload.get("capture_frame_id", payload.get("frame_id", -1))
            ),
            frame_source=str(payload.get("frame_source", "")),
            frame_metadata=dict(payload.get("frame_metadata", {})),
            action=None if payload.get("action") is None else dict(payload["action"]),
            action_vector=np.asarray(payload["action_vector"], dtype=np.float32),
            action_message_timestamp_ns=None
            if action_envelope.get("message_timestamp_ns") is None
            else int(action_envelope["message_timestamp_ns"]),
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
class BlackboxActionRecord:
    message_timestamp_ns: int
    publish_timestamp_ns: int
    action_vector: np.ndarray
    envelope: dict[str, Any]
    payload: dict[str, Any]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BlackboxActionRecord":
        envelope = dict(payload.get("envelope", {}))
        raw_payload = payload.get("payload")
        vector_payload = payload.get("action_vector")
        if vector_payload is None and isinstance(raw_payload, dict):
            vector_payload = [
                raw_payload.get("steer", 0.0),
                raw_payload.get("throttle", 0.0),
                raw_payload.get("brake", 0.0),
                raw_payload.get("handbrake", 0.0),
                raw_payload.get("reverse", 0.0),
                raw_payload.get("pilot_active", 0.0),
            ]
        return cls(
            message_timestamp_ns=int(envelope.get("message_timestamp_ns", 0)),
            publish_timestamp_ns=int(envelope.get("publish_timestamp_ns", 0)),
            action_vector=np.asarray(vector_payload or np.zeros(6, dtype=np.float32), dtype=np.float32),
            envelope=envelope,
            payload={} if raw_payload is None else dict(raw_payload),
        )

    @property
    def timestamp_ns(self) -> int:
        return self.message_timestamp_ns


@dataclass
class AtlasTemporalClipIndex:
    clip_id: str
    metadata_path: str
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
    def privileged_path(self) -> Path | None:
        return None if self.privileged_dir is None else Path(self.privileged_dir)
