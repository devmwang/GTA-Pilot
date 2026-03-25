from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..schema import ConvertedLaneSample, LaneSegment3D, MapElement3D


def make_polyline(points: list[list[float]] | np.ndarray, dims: int = 3) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("Expected polyline points with shape [N, C]")
    if arr.shape[1] < dims:
        pad = np.zeros((arr.shape[0], dims - arr.shape[1]), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=1)
    return arr[:, :dims]


def empty_lane_sample(source_id: str | None = None) -> ConvertedLaneSample:
    return ConvertedLaneSample(
        lane_segments=[],
        map_elements=[],
        valid_mask={},
        source_id=source_id,
    )


def default_lane(
    lane_id: int,
    centerline: np.ndarray,
    left_boundary: np.ndarray | None = None,
    right_boundary: np.ndarray | None = None,
    lane_semantic: str = "driving",
    direction: str = "unknown",
) -> LaneSegment3D:
    left = centerline if left_boundary is None else left_boundary
    right = centerline if right_boundary is None else right_boundary
    return LaneSegment3D(
        lane_id=lane_id,
        centerline=centerline,
        left_boundary=left,
        right_boundary=right,
        lane_semantic=lane_semantic,
        direction=direction,
        left_kind="unknown",
        left_color="unknown",
        left_pattern="unknown",
        left_continuity="unknown",
        right_kind="unknown",
        right_color="unknown",
        right_pattern="unknown",
        right_continuity="unknown",
    )


def default_map_element(
    elem_id: int,
    elem_type: str,
    polyline: np.ndarray,
    bound_lane_ids: list[int] | None = None,
) -> MapElement3D:
    return MapElement3D(
        elem_id=elem_id,
        elem_type=elem_type,
        polyline=polyline,
        bound_lane_ids=bound_lane_ids or [],
    )
