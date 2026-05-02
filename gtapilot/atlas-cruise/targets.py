from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import AtlasCruiseConfig


@dataclass(slots=True)
class EgoTelemetrySeries:
    timestamps_ns: np.ndarray
    pos_xy: np.ndarray
    yaw_rad: np.ndarray
    speed_mps: np.ndarray
    valid: np.ndarray
    raw: dict[str, np.ndarray]


def wrap_angle(value: np.ndarray | float) -> np.ndarray | float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def _as_float_array(payload: Any) -> np.ndarray:
    return np.asarray(payload, dtype=np.float32)


def _extract_timestamp(payload: dict[str, Any]) -> int:
    for key in ("host_qpc_ns", "timestamp_ns", "capture_timestamp_ns", "receive_time_ns"):
        if payload.get(key) is not None:
            return int(payload[key])
    raise ValueError(f"Ego telemetry row is missing a timestamp: {payload}")


def _extract_yaw(payload: dict[str, Any]) -> float:
    if payload.get("ego_heading_yaw") is not None:
        yaw = float(payload["ego_heading_yaw"])
    elif payload.get("heading_yaw") is not None:
        yaw = float(payload["heading_yaw"])
    elif payload.get("rot_yaw") is not None:
        yaw = float(payload["rot_yaw"])
    else:
        rot = payload.get("ego_rot_world")
        if rot is None:
            raise ValueError(f"Ego telemetry row is missing yaw: {payload}")
        yaw = float(rot[-1])
    if abs(yaw) > 2.0 * math.pi + 1e-3:
        yaw = math.radians(yaw)
    return float(wrap_angle(yaw))


def _extract_pos_xy(payload: dict[str, Any]) -> tuple[float, float]:
    if payload.get("ego_pos_world") is not None:
        pos = payload["ego_pos_world"]
        return float(pos[0]), float(pos[1])
    if payload.get("pos_x") is not None and payload.get("pos_y") is not None:
        return float(payload["pos_x"]), float(payload["pos_y"])
    raise ValueError(f"Ego telemetry row is missing position: {payload}")


def _extract_speed(payload: dict[str, Any]) -> float:
    for key in ("ego_speed_mps", "speed_mps", "ego_speed_forward_mps", "forward_speed_mps"):
        if payload.get(key) is not None:
            return float(payload[key])
    vel = payload.get("ego_velocity_world") or payload.get("velocity_world")
    if vel is not None:
        return float(np.linalg.norm(np.asarray(vel, dtype=np.float32)))
    if payload.get("vel_x") is not None and payload.get("vel_y") is not None:
        return float(math.hypot(float(payload["vel_x"]), float(payload["vel_y"])))
    return 0.0


def _load_ego_json(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(row) for row in payload]
    for key in ("ego", "samples", "telemetry"):
        if isinstance(payload.get(key), list):
            return [dict(row) for row in payload[key]]
    raise ValueError(f"Unsupported ego JSON payload: {path}")


def _load_ego_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_ego_telemetry_source(path: str | Path) -> EgoTelemetrySeries:
    source = Path(path)
    if source.suffix == ".jsonl":
        rows = _load_ego_jsonl(source)
        return ego_series_from_rows(rows)
    if source.suffix == ".json":
        rows = _load_ego_json(source)
        return ego_series_from_rows(rows)
    if source.suffix == ".npz":
        payload = np.load(source)
        return ego_series_from_arrays({key: payload[key] for key in payload.files})
    raise ValueError(f"Unsupported ego telemetry source suffix {source.suffix!r}.")


def ego_series_from_rows(rows: list[dict[str, Any]]) -> EgoTelemetrySeries:
    timestamps: list[int] = []
    pos: list[tuple[float, float]] = []
    yaw: list[float] = []
    speed: list[float] = []
    valid: list[bool] = []
    for row in rows:
        timestamps.append(_extract_timestamp(row))
        pos.append(_extract_pos_xy(row))
        yaw.append(_extract_yaw(row))
        speed.append(_extract_speed(row))
        valid.append(bool(row.get("valid", row.get("vehicle_valid", True))))
    return ego_series_from_arrays(
        {
            "timestamps_ns": np.asarray(timestamps, dtype=np.int64),
            "pos_xy": np.asarray(pos, dtype=np.float32),
            "yaw_rad": np.asarray(yaw, dtype=np.float32),
            "speed_mps": np.asarray(speed, dtype=np.float32),
            "valid": np.asarray(valid, dtype=bool),
        }
    )


def ego_series_from_arrays(arrays: dict[str, np.ndarray]) -> EgoTelemetrySeries:
    if "pose_xyyaw_world" in arrays:
        pose = _as_float_array(arrays["pose_xyyaw_world"])
        pos_xy = pose[:, :2]
        yaw_rad = pose[:, 2]
    else:
        if "pos_xy" in arrays:
            pos_xy = _as_float_array(arrays["pos_xy"])
        elif "ego_pos_world" in arrays:
            pos_xy = _as_float_array(arrays["ego_pos_world"])[:, :2]
        elif "pos_x" in arrays and "pos_y" in arrays:
            pos_xy = np.stack([arrays["pos_x"], arrays["pos_y"]], axis=-1).astype(np.float32)
        else:
            raise ValueError("Ego arrays must include pose_xyyaw_world, pos_xy, ego_pos_world, or pos_x/pos_y.")

        if "yaw_rad" in arrays:
            yaw_rad = _as_float_array(arrays["yaw_rad"])
        elif "ego_heading_yaw" in arrays:
            yaw_rad = _as_float_array(arrays["ego_heading_yaw"])
        elif "heading_yaw" in arrays:
            yaw_rad = _as_float_array(arrays["heading_yaw"])
        else:
            raise ValueError("Ego arrays must include yaw_rad, ego_heading_yaw, or heading_yaw.")

    if "timestamps_ns" in arrays:
        timestamps_ns = np.asarray(arrays["timestamps_ns"], dtype=np.int64)
    elif "host_qpc_ns" in arrays:
        timestamps_ns = np.asarray(arrays["host_qpc_ns"], dtype=np.int64)
    elif "capture_timestamps_ns" in arrays:
        timestamps_ns = np.asarray(arrays["capture_timestamps_ns"], dtype=np.int64)
    else:
        raise ValueError("Ego arrays must include timestamps_ns, host_qpc_ns, or capture_timestamps_ns.")

    if "speed_mps" in arrays:
        speed_mps = _as_float_array(arrays["speed_mps"])
    elif "ego_speed_mps" in arrays:
        speed_mps = _as_float_array(arrays["ego_speed_mps"])
    elif "kinematics" in arrays:
        speed_mps = _as_float_array(arrays["kinematics"])[:, 0]
    else:
        speed_mps = np.zeros((timestamps_ns.shape[0],), dtype=np.float32)

    valid = np.asarray(arrays.get("valid", arrays.get("ego_valid", np.ones_like(timestamps_ns, dtype=bool))), dtype=bool)
    yaw_rad = _as_float_array(yaw_rad)
    if np.nanmax(np.abs(yaw_rad)) > 2.0 * math.pi + 1e-3:
        yaw_rad = np.deg2rad(yaw_rad)
    order = np.argsort(timestamps_ns, kind="stable")
    timestamps_ns = timestamps_ns[order]
    pos_xy = _as_float_array(pos_xy)[order]
    yaw_rad = yaw_rad[order]
    speed_mps = _as_float_array(speed_mps)[order]
    valid = valid[order]
    unique = np.concatenate([[True], np.diff(timestamps_ns) > 0])
    return EgoTelemetrySeries(
        timestamps_ns=timestamps_ns[unique],
        pos_xy=pos_xy[unique],
        yaw_rad=np.unwrap(yaw_rad[unique]).astype(np.float32),
        speed_mps=np.maximum(speed_mps[unique], 0.0).astype(np.float32),
        valid=valid[unique],
        raw={key: np.asarray(value)[order][unique] for key, value in arrays.items() if np.asarray(value).shape[:1] == order.shape},
    )


def smooth_ego_series(series: EgoTelemetrySeries, *, alpha: float = 0.25) -> EgoTelemetrySeries:
    def smooth(values: np.ndarray) -> np.ndarray:
        out = np.asarray(values, dtype=np.float32).copy()
        for idx in range(1, out.shape[0]):
            out[idx] = alpha * out[idx] + (1.0 - alpha) * out[idx - 1]
        for idx in range(out.shape[0] - 2, -1, -1):
            out[idx] = alpha * out[idx] + (1.0 - alpha) * out[idx + 1]
        return out

    return EgoTelemetrySeries(
        timestamps_ns=series.timestamps_ns,
        pos_xy=smooth(series.pos_xy),
        yaw_rad=smooth(series.yaw_rad),
        speed_mps=smooth(series.speed_mps),
        valid=series.valid,
        raw=series.raw,
    )


def _segment_bad(series: EgoTelemetrySeries, cfg: AtlasCruiseConfig) -> np.ndarray:
    ts = series.timestamps_ns
    if ts.shape[0] <= 1:
        return np.ones((0,), dtype=bool)
    dt_s = np.diff(ts).astype(np.float64) / 1_000_000_000.0
    delta = np.linalg.norm(np.diff(series.pos_xy.astype(np.float64), axis=0), axis=1)
    implied_speed = delta / np.clip(dt_s, 1e-6, None)
    return (
        (dt_s <= 0.0)
        | (dt_s > cfg.max_ego_interp_gap_s)
        | (delta > cfg.max_ego_teleport_m)
        | (implied_speed > cfg.max_ego_speed_mps)
        | (~series.valid[:-1])
        | (~series.valid[1:])
    )


def interpolate_ego(
    series: EgoTelemetrySeries,
    query_timestamps_ns: np.ndarray,
    cfg: AtlasCruiseConfig,
) -> tuple[np.ndarray, np.ndarray]:
    query = np.asarray(query_timestamps_ns, dtype=np.int64)
    out = np.zeros((query.shape[0], 4), dtype=np.float32)
    valid = np.zeros((query.shape[0],), dtype=bool)
    ts = series.timestamps_ns
    if ts.shape[0] < 2:
        return out, valid
    segment_bad = _segment_bad(series, cfg)
    right = np.searchsorted(ts, query, side="right")
    left = right - 1
    inside = (left >= 0) & (right < ts.shape[0])
    valid[inside] = True
    for out_idx in np.where(inside)[0].tolist():
        l_idx = int(left[out_idx])
        r_idx = int(right[out_idx])
        if bool(segment_bad[l_idx]):
            valid[out_idx] = False
            continue
        denom = max(1, int(ts[r_idx] - ts[l_idx]))
        frac = float(query[out_idx] - ts[l_idx]) / float(denom)
        frac = max(0.0, min(1.0, frac))
        pos = (1.0 - frac) * series.pos_xy[l_idx] + frac * series.pos_xy[r_idx]
        yaw = (1.0 - frac) * series.yaw_rad[l_idx] + frac * series.yaw_rad[r_idx]
        speed = (1.0 - frac) * series.speed_mps[l_idx] + frac * series.speed_mps[r_idx]
        out[out_idx] = [float(pos[0]), float(pos[1]), float(yaw), float(speed)]
    return out, valid


def build_target_trajectory_for_timestamp(
    anchor_timestamp_ns: int,
    ego: EgoTelemetrySeries,
    cfg: AtlasCruiseConfig,
) -> tuple[np.ndarray, np.ndarray]:
    base_ns = int(anchor_timestamp_ns + round(cfg.target_latency_s * 1_000_000_000.0))
    step_ns = int(round(cfg.traj_dt_s * 1_000_000_000.0))
    target_ts = np.asarray([base_ns + idx * step_ns for idx in range(cfg.traj_points)], dtype=np.int64)
    poses, valid = interpolate_ego(ego, target_ts, cfg)
    traj = np.zeros((cfg.traj_points, 4), dtype=np.float32)
    if not bool(valid[0]):
        return traj, np.zeros_like(valid)
    origin_x, origin_y, origin_yaw, _ = poses[0]
    cos_yaw = math.cos(float(origin_yaw))
    sin_yaw = math.sin(float(origin_yaw))
    dx = poses[:, 0] - origin_x
    dy = poses[:, 1] - origin_y
    traj[:, 0] = cos_yaw * dx + sin_yaw * dy
    traj[:, 1] = -sin_yaw * dx + cos_yaw * dy
    traj[:, 2] = wrap_angle(poses[:, 2] - origin_yaw)
    traj[:, 3] = np.maximum(poses[:, 3], 0.0)
    return traj, valid


def build_ego_target_for_timestamp(
    anchor_timestamp_ns: int,
    ego: EgoTelemetrySeries,
    cfg: AtlasCruiseConfig,
) -> tuple[np.ndarray, bool]:
    center_ns = int(anchor_timestamp_ns + round(cfg.target_latency_s * 1_000_000_000.0))
    sample_step_ns = int(round(0.10 * 1_000_000_000.0))
    query = np.asarray([center_ns - sample_step_ns, center_ns, center_ns + sample_step_ns], dtype=np.int64)
    poses, valid = interpolate_ego(ego, query, cfg)
    if not bool(valid.all()):
        return np.zeros((4,), dtype=np.float32), False
    dt_s = max(1e-3, (query[2] - query[0]) / 1_000_000_000.0)
    speed = float(poses[1, 3])
    a_long = float((poses[2, 3] - poses[0, 3]) / dt_s)
    yaw_rate = float(wrap_angle(poses[2, 2] - poses[0, 2]) / dt_s)
    curvature = yaw_rate / max(speed, 0.5)
    return np.asarray([speed, a_long, yaw_rate, curvature], dtype=np.float32), True


def _latest_indices_before(source_timestamps_ns: np.ndarray, query_timestamps_ns: np.ndarray) -> np.ndarray:
    return np.searchsorted(source_timestamps_ns, query_timestamps_ns, side="right") - 1


def build_control_aux_targets(
    anchor_timestamps_ns: np.ndarray,
    action_timestamps_ns: np.ndarray,
    action_vectors: np.ndarray,
    cfg: AtlasCruiseConfig,
) -> np.ndarray:
    output = np.zeros((anchor_timestamps_ns.shape[0], cfg.control_horizon_steps, 3), dtype=np.float32)
    if action_timestamps_ns.size == 0:
        return output
    step_ns = int(round(cfg.control_dt_s * 1_000_000_000.0))
    for row_idx, anchor_ns in enumerate(anchor_timestamps_ns.tolist()):
        query = np.asarray([int(anchor_ns) + idx * step_ns for idx in range(cfg.control_horizon_steps)], dtype=np.int64)
        indices = _latest_indices_before(action_timestamps_ns, query)
        valid = indices >= 0
        output[row_idx, valid] = action_vectors[indices[valid], :3]
    return output


def build_slow_or_brake_targets(
    target_traj: np.ndarray,
    target_valid: np.ndarray,
    action_timestamps_ns: np.ndarray,
    action_vectors: np.ndarray,
    anchor_timestamps_ns: np.ndarray,
    cfg: AtlasCruiseConfig,
) -> np.ndarray:
    out = np.zeros((anchor_timestamps_ns.shape[0], 1), dtype=np.float32)
    if action_timestamps_ns.size > 0:
        indices = _latest_indices_before(action_timestamps_ns, anchor_timestamps_ns)
        valid = indices >= 0
        out[valid, 0] = (action_vectors[indices[valid], 2] > 0.15).astype(np.float32)
    one_s_step = max(1, int(round(1.0 / cfg.traj_dt_s)))
    for idx in range(anchor_timestamps_ns.shape[0]):
        if target_valid[idx, 0] and target_valid[idx, min(one_s_step, cfg.traj_points - 1)]:
            v0 = float(target_traj[idx, 0, 3])
            v1 = float(target_traj[idx, min(one_s_step, cfg.traj_points - 1), 3])
            if v1 < v0 - 3.0:
                out[idx, 0] = 1.0
    return out


def build_cruise_target_arrays(
    anchor_timestamps_ns: np.ndarray,
    ego_series: EgoTelemetrySeries,
    cfg: AtlasCruiseConfig,
    *,
    action_timestamps_ns: np.ndarray | None = None,
    action_vectors: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    cfg.validate()
    anchors = np.asarray(anchor_timestamps_ns, dtype=np.int64)
    ego = smooth_ego_series(ego_series)
    target_traj = np.zeros((anchors.shape[0], cfg.traj_points, 4), dtype=np.float32)
    target_traj_valid = np.zeros((anchors.shape[0], cfg.traj_points), dtype=bool)
    target_ego = np.zeros((anchors.shape[0], 4), dtype=np.float32)
    target_ego_valid = np.zeros((anchors.shape[0],), dtype=bool)
    for idx, timestamp_ns in enumerate(anchors.tolist()):
        traj, valid = build_target_trajectory_for_timestamp(int(timestamp_ns), ego, cfg)
        ego_target, ego_valid = build_ego_target_for_timestamp(int(timestamp_ns), ego, cfg)
        target_traj[idx] = traj
        target_traj_valid[idx] = valid
        target_ego[idx] = ego_target
        target_ego_valid[idx] = ego_valid
    actions_ts = np.asarray([] if action_timestamps_ns is None else action_timestamps_ns, dtype=np.int64)
    actions = np.zeros((actions_ts.shape[0], cfg.action_dim), dtype=np.float32)
    if action_vectors is not None and actions_ts.size:
        actions = np.asarray(action_vectors, dtype=np.float32)
    target_slow = build_slow_or_brake_targets(target_traj, target_traj_valid, actions_ts, actions, anchors, cfg)
    target_fallback = (
        (target_traj_valid.sum(axis=1) < cfg.min_valid_traj_points) | (~target_ego_valid)
    ).astype(np.float32)[:, None]
    target_control_aux = build_control_aux_targets(anchors, actions_ts, actions, cfg)
    return {
        "anchor_timestamps_ns": anchors,
        "target_traj": target_traj,
        "target_traj_valid": target_traj_valid,
        "target_ego": target_ego,
        "target_ego_valid": target_ego_valid,
        "target_slow_or_brake": target_slow,
        "target_fallback": target_fallback,
        "target_control_aux": target_control_aux,
    }
