from __future__ import annotations

from typing import Any

from .common import default_lane, default_map_element, empty_lane_sample, make_polyline


def convert_openlane_v2_record(record: dict[str, Any]) -> dict[str, Any]:
    sample = empty_lane_sample(source_id=record.get("segment_id"))
    topology = record.get("topology", {})
    for lane_id, lane in enumerate(record.get("lane_segments", [])):
        centerline = make_polyline(lane.get("centerline", []), dims=3)
        left_boundary = make_polyline(lane.get("left_boundary", centerline), dims=3)
        right_boundary = make_polyline(lane.get("right_boundary", centerline), dims=3)
        lane_obj = default_lane(
            lane_id=lane_id,
            centerline=centerline,
            left_boundary=left_boundary,
            right_boundary=right_boundary,
            lane_semantic=str(lane.get("type", "driving")),
            direction=str(lane.get("direction", "unknown")),
        )
        lane_obj.predecessors = list(topology.get("predecessors", {}).get(str(lane_id), []))
        lane_obj.successors = list(topology.get("successors", {}).get(str(lane_id), []))
        sample.lane_segments.append(lane_obj)
    for elem_id, element in enumerate(record.get("traffic_elements", [])):
        sample.map_elements.append(
            default_map_element(
                elem_id,
                str(element.get("type", "keepout_other")),
                make_polyline(element.get("polyline", []), dims=3),
            )
        )
    sample.valid_mask = {
        "centerline": True,
        "left_boundary": True,
        "right_boundary": True,
        "lane_semantic": True,
        "direction": True,
        "topology": True,
        "map_elements": True,
    }
    return {"sample": sample}
