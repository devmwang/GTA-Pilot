from __future__ import annotations

from typing import Any

from .common import default_lane, empty_lane_sample, make_polyline


def convert_openlane_record(record: dict[str, Any]) -> dict[str, Any]:
    sample = empty_lane_sample(source_id=record.get("file_path"))
    for lane_id, lane in enumerate(record.get("lane_lines", [])):
        centerline = make_polyline(lane.get("xyz", []), dims=3)
        lane_obj = default_lane(
            lane_id=lane_id,
            centerline=centerline,
            lane_semantic=str(lane.get("category", "driving")),
            direction=str(lane.get("visibility", "unknown")),
        )
        sample.lane_segments.append(lane_obj)
    sample.valid_mask = {
        "centerline": True,
        "left_boundary": False,
        "right_boundary": False,
        "lane_semantic": True,
        "direction": False,
        "topology": False,
    }
    return {"sample": sample}
