from __future__ import annotations

from ..schema import LaneSegment3D


def attach_lane_topology(lane_segments: list[LaneSegment3D]) -> list[LaneSegment3D]:
    for idx, lane in enumerate(lane_segments):
        lane.predecessors = [lane_segments[idx - 1].lane_id] if idx > 0 else []
        lane.successors = [lane_segments[idx + 1].lane_id] if idx + 1 < len(lane_segments) else []
        lane.left_adjacent = lane_segments[idx - 1].lane_id if idx > 0 else None
        lane.right_adjacent = lane_segments[idx + 1].lane_id if idx + 1 < len(lane_segments) else None
    return lane_segments
