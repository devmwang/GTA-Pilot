from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class RoadGraphEdge:
    edge_id: int
    centerline: np.ndarray
    lane_count: int
    direction: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RoadGraph:
    edges: list[RoadGraphEdge]
    intersections: list[np.ndarray] = field(default_factory=list)
    stop_lines: list[np.ndarray] = field(default_factory=list)
    crosswalks: list[np.ndarray] = field(default_factory=list)


def build_road_graph(raw_records: list[dict[str, Any]]) -> RoadGraph:
    edges = []
    for edge_id, record in enumerate(raw_records):
        edges.append(
            RoadGraphEdge(
                edge_id=edge_id,
                centerline=np.asarray(record.get("centerline", []), dtype=np.float32),
                lane_count=int(record.get("lane_count", 1)),
                direction=str(record.get("direction", "unknown")),
                metadata=dict(record),
            )
        )
    return RoadGraph(edges=edges)
