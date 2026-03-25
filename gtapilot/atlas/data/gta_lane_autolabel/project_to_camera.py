from __future__ import annotations

import numpy as np

from ..schema import LaneSegment3D


def project_lane_segments_to_camera(
    lane_segments: list[LaneSegment3D],
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
) -> list[dict[str, np.ndarray]]:
    projected = []
    for lane in lane_segments:
        points = np.asarray(lane.centerline, dtype=np.float32)
        if points.shape[1] == 2:
            points = np.concatenate(
                [points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=1
            )
        points_h = np.concatenate(
            [points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=1
        )
        cam = (extrinsics @ points_h.T).T[:, :3]
        cam[:, 2] = np.clip(cam[:, 2], 1e-3, None)
        uvw = (intrinsics @ cam.T).T
        projected.append({"lane_id": lane.lane_id, "uv": uvw[:, :2] / uvw[:, 2:3]})
    return projected
