from __future__ import annotations

import numpy as np

from ..schema import LaneSegment3D
from .road_graph_extract import RoadGraph


def fit_lane_segments(road_graph: RoadGraph, default_lane_width_m: float = 3.5) -> list[LaneSegment3D]:
    lane_segments: list[LaneSegment3D] = []
    lane_id = 0
    for edge in road_graph.edges:
        center = np.asarray(edge.centerline, dtype=np.float32)
        if center.ndim != 2 or center.shape[0] == 0:
            continue
        lateral = np.zeros_like(center)
        if center.shape[1] >= 2:
            lateral[:, 1] = default_lane_width_m / 2.0
        for offset_idx in range(max(edge.lane_count, 1)):
            shift = offset_idx - (edge.lane_count - 1) / 2.0
            left = center + lateral * (shift + 0.5)
            right = center + lateral * (shift - 0.5)
            lane_segments.append(
                LaneSegment3D(
                    lane_id=lane_id,
                    centerline=center + lateral * shift,
                    left_boundary=left,
                    right_boundary=right,
                    lane_semantic="driving",
                    direction=edge.direction,
                    left_kind="paint",
                    left_color="white",
                    left_pattern="solid",
                    left_continuity="continuous",
                    right_kind="paint",
                    right_color="white",
                    right_pattern="solid",
                    right_continuity="continuous",
                )
            )
            lane_id += 1
    return lane_segments
