from __future__ import annotations

from ..schema import LaneSegment3D, MapElement3D


def summarize_autolabel_bundle(
    lane_segments: list[LaneSegment3D], map_elements: list[MapElement3D]
) -> dict[str, int]:
    return {
        "lane_segments": len(lane_segments),
        "map_elements": len(map_elements),
        "stop_lines": sum(1 for elem in map_elements if elem.elem_type == "stop_line"),
        "crosswalk_boundaries": sum(
            1 for elem in map_elements if elem.elem_type == "crosswalk_boundary"
        ),
    }
