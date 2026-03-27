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
    frame_source: str
    frame_metadata: dict[str, Any]
    action: dict[str, Any] | None
    action_vector: np.ndarray

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BlackboxFrameRecord":
        return cls(
            video_frame_index=int(payload["video_frame_index"]),
            capture_timestamp_ns=int(payload["capture_timestamp_ns"]),
            publish_timestamp_ns=int(payload["publish_timestamp_ns"]),
            frame_id=int(payload.get("frame_id", -1)),
            frame_source=str(payload.get("frame_source", "")),
            frame_metadata=dict(payload.get("frame_metadata", {})),
            action=None if payload.get("action") is None else dict(payload["action"]),
            action_vector=np.asarray(payload["action_vector"], dtype=np.float32),
        )


@dataclass
class AtlasTemporalClipIndex:
    clip_id: str
    metadata_path: str
    video_path: str
    target_frame_index: int
    recent_frame_indices: list[int]
    older_frame_indices: list[int]
    mid_frame_indices: list[int]
    action_frame_indices: list[int]
    nominal_fps: float
    frame_source: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def metadata_file(self) -> Path:
        return Path(self.metadata_path)

    @property
    def video_file(self) -> Path:
        return Path(self.video_path)
