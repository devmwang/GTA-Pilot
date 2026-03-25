from __future__ import annotations

from typing import Any

from .common import default_lane, empty_lane_sample, make_polyline


def convert_once_3dlanes_record(record: dict[str, Any]) -> dict[str, Any]:
    sample = empty_lane_sample(source_id=record.get("sample_idx"))
    for lane_id, lane in enumerate(record.get("lanes", [])):
        centerline = make_polyline(lane.get("xyz", []), dims=3)
        sample.lane_segments.append(default_lane(lane_id=lane_id, centerline=centerline))
    sample.valid_mask = {
        "centerline": True,
        "left_boundary": False,
        "right_boundary": False,
        "lane_semantic": False,
        "direction": False,
        "topology": False,
    }
    return {"sample": sample}
