from __future__ import annotations

from typing import Any

from .common import default_lane, empty_lane_sample, make_polyline


def convert_apolloscape_record(record: dict[str, Any]) -> dict[str, Any]:
    sample = empty_lane_sample(source_id=record.get("image_key"))
    for lane_id, lane in enumerate(record.get("lanes", [])):
        polyline = make_polyline(lane.get("points", []), dims=3)
        sample.lane_segments.append(default_lane(lane_id=lane_id, centerline=polyline))
    sample.valid_mask = {
        "centerline": True,
        "left_boundary": False,
        "right_boundary": False,
        "lane_semantic": False,
        "direction": False,
        "topology": False,
    }
    return {"sample": sample}
