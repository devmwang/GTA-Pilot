from __future__ import annotations

from typing import Any

from .common import default_lane, default_map_element, empty_lane_sample, make_polyline


_TYPE_MAP = {
    "single white": ("driving", "white", "solid"),
    "single yellow": ("driving", "yellow", "solid"),
    "double white": ("driving", "white", "double"),
    "double yellow": ("driving", "yellow", "double"),
    "road curb": ("other", "other", "solid"),
    "crosswalk": ("other", "white", "solid"),
}


def convert_bdd100k_record(record: dict[str, Any]) -> dict[str, Any]:
    sample = empty_lane_sample(source_id=record.get("name"))
    labels = record.get("labels", [])
    lane_id = 0
    elem_id = 0
    for label in labels:
        category = label.get("category", "").lower()
        poly2d = label.get("poly2d", [])
        if not poly2d:
            continue
        polyline = make_polyline(poly2d[0].get("vertices", []), dims=3)
        if "lane" in category or "curb" in category:
            lane = default_lane(lane_id=lane_id, centerline=polyline)
            lane.left_kind = "curb" if "curb" in category else "paint"
            _, lane.left_color, lane.left_pattern = _TYPE_MAP.get(category, ("driving", "unknown", "unknown"))
            lane.right_kind = lane.left_kind
            lane.right_color = lane.left_color
            lane.right_pattern = lane.left_pattern
            sample.lane_segments.append(lane)
            lane_id += 1
        elif "crosswalk" in category or "stop" in category:
            elem_type = "crosswalk_boundary" if "crosswalk" in category else "stop_line"
            sample.map_elements.append(default_map_element(elem_id, elem_type, polyline))
            elem_id += 1
    sample.valid_mask = {
        "centerline": True,
        "left_boundary": False,
        "right_boundary": False,
        "lane_semantic": True,
        "direction": True,
        "topology": False,
        "map_elements": True,
    }
    return {"sample": sample}
