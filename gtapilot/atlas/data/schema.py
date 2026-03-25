from __future__ import annotations

from dataclasses import asdict, dataclass, field
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
class LoggedStep:
    episode_id: str
    frame_idx: int
    timestamp_ms: int

    rgb_front_path: str
    action: np.ndarray
    dt_s: float

    route_polyline: np.ndarray | None = None
    nav_cmd: np.ndarray | None = None

    gt_pose_local: np.ndarray | None = None
    gt_kinematics: np.ndarray | None = None
    gt_actors: dict[str, Any] | None = None
    gt_occ_state: np.ndarray | None = None
    gt_occ_sem: np.ndarray | None = None
    gt_bev_lite: np.ndarray | None = None
    gt_provenance: np.ndarray | None = None
    gt_lane_segments: list[LaneSegment3D] | None = None
    gt_map_elements: list[MapElement3D] | None = None
    gt_teacher_trajs: np.ndarray | None = None
    gt_teacher_costs: np.ndarray | None = None
    gt_teacher_best: int | None = None
    valid_mask: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in (
            "action",
            "route_polyline",
            "nav_cmd",
            "gt_pose_local",
            "gt_kinematics",
            "gt_occ_state",
            "gt_occ_sem",
            "gt_bev_lite",
            "gt_provenance",
            "gt_teacher_trajs",
            "gt_teacher_costs",
        ):
            value = payload.get(key)
            if value is not None:
                payload[key] = np.asarray(value).tolist()
        if payload.get("gt_lane_segments") is not None:
            payload["gt_lane_segments"] = [
                segment.to_dict() for segment in self.gt_lane_segments or []
            ]
        if payload.get("gt_map_elements") is not None:
            payload["gt_map_elements"] = [
                element.to_dict() for element in self.gt_map_elements or []
            ]
        return payload
