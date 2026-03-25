from __future__ import annotations

from typing import Any

from .common import default_lane, empty_lane_sample, make_polyline


def convert_culane_record(record: dict[str, Any]) -> dict[str, Any]:
    sample = empty_lane_sample(source_id=record.get("id"))
    for lane_id, lane_points in enumerate(record.get("lanes", [])):
        centerline = make_polyline(lane_points, dims=2)
        sample.lane_segments.append(
            default_lane(
                lane_id=lane_id,
                centerline=make_polyline(centerline, dims=3),
                lane_semantic="intersection_imputed"
                if record.get("occluded")
                else "driving",
            )
        )
    sample.valid_mask = {
        "centerline": True,
        "left_boundary": False,
        "right_boundary": False,
        "lane_semantic": False,
        "direction": False,
        "topology": False,
    }
    return {"sample": sample}
