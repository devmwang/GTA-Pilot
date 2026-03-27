from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..config import AtlasConfig


@dataclass
class Stage1BPrivilegedTargets:
    pose_delta_local: np.ndarray
    kinematics: np.ndarray
    ego_valid: np.ndarray
    depth_8x_m: np.ndarray
    depth_valid_8x: np.ndarray
    dynamic_mask_8x: np.ndarray
    track_lag_indices: np.ndarray
    track_target_sparse: np.ndarray
    track_valid_sparse: np.ndarray


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def compute_pose_delta_local(
    pose_xyyaw_world: np.ndarray,
) -> np.ndarray:
    pose = np.asarray(pose_xyyaw_world, dtype=np.float32)
    delta = np.zeros_like(pose)
    if pose.shape[0] <= 1:
        return delta
    world_delta = pose[1:] - pose[:-1]
    prev_yaw = pose[:-1, 2]
    cos_yaw = np.cos(prev_yaw)
    sin_yaw = np.sin(prev_yaw)
    dx_local = cos_yaw * world_delta[:, 0] + sin_yaw * world_delta[:, 1]
    dy_local = -sin_yaw * world_delta[:, 0] + cos_yaw * world_delta[:, 1]
    delta[1:, 0] = dx_local
    delta[1:, 1] = dy_local
    delta[1:, 2] = wrap_angle(world_delta[:, 2])
    return delta


def compute_kinematics(
    pose_delta_local: np.ndarray,
    dt_s: np.ndarray,
) -> np.ndarray:
    dt_s = np.asarray(dt_s, dtype=np.float32).reshape(-1)
    pose_delta_local = np.asarray(pose_delta_local, dtype=np.float32)
    kin = np.zeros((pose_delta_local.shape[0], 4), dtype=np.float32)
    safe_dt = np.clip(dt_s, 1e-4, None)
    speed = np.linalg.norm(pose_delta_local[:, :2], axis=-1) / safe_dt
    kin[:, 0] = speed
    if pose_delta_local.shape[0] > 1:
        kin[1:, 1] = (speed[1:] - speed[:-1]) / safe_dt[1:]
    kin[:, 2] = pose_delta_local[:, 2] / safe_dt
    kin[:, 3] = np.abs(pose_delta_local[:, 1]) / safe_dt
    return kin


def downsample_depth_native_to_8x(depth_native_m: np.ndarray) -> np.ndarray:
    depth_native_m = np.asarray(depth_native_m, dtype=np.float32)
    if depth_native_m.ndim != 3:
        raise ValueError("depth_native_m must have shape [T, H, W].")
    t, h, w = depth_native_m.shape
    if h % 8 != 0 or w % 8 != 0:
        raise ValueError("depth_native_m spatial shape must be divisible by 8.")
    depth = depth_native_m.reshape(t, h // 8, 8, w // 8, 8)
    valid = np.isfinite(depth) & (depth > 0.0)
    summed = np.where(valid, depth, 0.0).sum(axis=(2, 4))
    counts = valid.sum(axis=(2, 4)).clip(min=1)
    return (summed / counts).astype(np.float32)


def build_rigid_track_targets(
    *,
    cfg: AtlasConfig,
    depth_8x_m: np.ndarray,
    pose_xyyaw_world: np.ndarray,
    dynamic_mask_8x: np.ndarray,
    lag_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    depth_8x_m = np.asarray(depth_8x_m, dtype=np.float32)
    pose_xyyaw_world = np.asarray(pose_xyyaw_world, dtype=np.float32)
    dynamic_mask_8x = np.asarray(dynamic_mask_8x, dtype=bool)
    lag_indices = np.asarray(lag_indices, dtype=np.int64)
    t, h, w = depth_8x_m.shape
    track_target = np.zeros((t, lag_indices.shape[0], 2, h, w), dtype=np.float32)
    track_valid = np.zeros((t, lag_indices.shape[0], h, w), dtype=bool)
    yy, xx = np.meshgrid(
        np.arange(h, dtype=np.float32),
        np.arange(w, dtype=np.float32),
        indexing="ij",
    )
    scale_x = float(cfg.image.padded_width) / float(w)
    scale_y = float(cfg.image.padded_height) / float(h)
    u = (xx + 0.5) * scale_x
    v = (yy + 0.5) * scale_y
    fx = float(cfg.image.fx)
    fy = float(cfg.image.fy)
    cx = float(cfg.image.padded_width) * 0.5
    cy = float(cfg.image.padded_height) * 0.5

    for time_idx in range(t):
        current_depth = depth_8x_m[time_idx]
        current_valid = (
            np.isfinite(current_depth)
            & (current_depth >= cfg.image.depth_min_m)
            & (current_depth <= cfg.image.depth_max_m)
            & (~dynamic_mask_8x[time_idx])
        )
        if not np.any(current_valid):
            continue
        x_cam = ((u - cx) / fx) * current_depth
        y_cam = ((v - cy) / fy) * current_depth
        z_cam = current_depth
        forward_cur = z_cam
        left_cur = -x_cam
        yaw_cur = float(pose_xyyaw_world[time_idx, 2])
        cos_cur = math.cos(yaw_cur)
        sin_cur = math.sin(yaw_cur)
        world_x = float(pose_xyyaw_world[time_idx, 0]) + cos_cur * forward_cur - sin_cur * left_cur
        world_y = float(pose_xyyaw_world[time_idx, 1]) + sin_cur * forward_cur + cos_cur * left_cur

        for lag_idx, lag in enumerate(lag_indices.tolist()):
            prev_idx = time_idx - lag
            if prev_idx < 0:
                continue
            yaw_prev = float(pose_xyyaw_world[prev_idx, 2])
            cos_prev = math.cos(yaw_prev)
            sin_prev = math.sin(yaw_prev)
            dx_world = world_x - float(pose_xyyaw_world[prev_idx, 0])
            dy_world = world_y - float(pose_xyyaw_world[prev_idx, 1])
            forward_prev = cos_prev * dx_world + sin_prev * dy_world
            left_prev = -sin_prev * dx_world + cos_prev * dy_world
            x_prev = -left_prev
            z_prev = forward_prev
            sane_depth = z_prev > max(cfg.image.depth_min_m, 1e-3)
            u_prev = fx * (x_prev / np.maximum(z_prev, 1e-3)) + cx
            v_prev = fy * (y_cam / np.maximum(z_prev, 1e-3)) + cy
            target_x = (u_prev / scale_x) - 0.5
            target_y = (v_prev / scale_y) - 0.5
            in_bounds = (
                (target_x >= 0.0)
                & (target_x <= (w - 1))
                & (target_y >= 0.0)
                & (target_y <= (h - 1))
            )
            finite_target = np.isfinite(target_x) & np.isfinite(target_y)
            valid = (
                current_valid
                & sane_depth
                & finite_target
                & in_bounds
                & (~dynamic_mask_8x[prev_idx])
            )
            track_target[time_idx, lag_idx, 0] = target_x - xx
            track_target[time_idx, lag_idx, 1] = target_y - yy
            track_valid[time_idx, lag_idx] = valid
    return track_target, track_valid


def build_stage1b_targets(
    *,
    cfg: AtlasConfig,
    pose_xyyaw_world: np.ndarray,
    dt_s: np.ndarray,
    depth_native_m: np.ndarray | None = None,
    depth_8x_m: np.ndarray | None = None,
    dynamic_mask_8x: np.ndarray | None = None,
    ego_valid: np.ndarray | None = None,
    track_lag_indices: np.ndarray | None = None,
) -> Stage1BPrivilegedTargets:
    if depth_8x_m is None:
        if depth_native_m is None:
            raise ValueError("Either depth_native_m or depth_8x_m is required.")
        depth_8x_m = downsample_depth_native_to_8x(depth_native_m)
    depth_8x_m = np.asarray(depth_8x_m, dtype=np.float32)
    if dynamic_mask_8x is None:
        dynamic_mask_8x = np.zeros_like(depth_8x_m, dtype=bool)
    if ego_valid is None:
        ego_valid = np.ones((depth_8x_m.shape[0],), dtype=bool)
    if track_lag_indices is None:
        track_lag_indices = np.asarray(cfg.geometry.track_lag_indices, dtype=np.int64)

    pose_delta_local = compute_pose_delta_local(pose_xyyaw_world)
    kinematics = compute_kinematics(pose_delta_local, dt_s)
    depth_valid_8x = (
        np.isfinite(depth_8x_m)
        & (depth_8x_m >= cfg.image.depth_min_m)
        & (depth_8x_m <= cfg.image.depth_max_m)
    )
    depth_8x_m = np.clip(depth_8x_m, cfg.image.depth_min_m, cfg.image.depth_max_m)
    track_target_sparse, track_valid_sparse = build_rigid_track_targets(
        cfg=cfg,
        depth_8x_m=depth_8x_m,
        pose_xyyaw_world=pose_xyyaw_world,
        dynamic_mask_8x=dynamic_mask_8x,
        lag_indices=track_lag_indices,
    )
    return Stage1BPrivilegedTargets(
        pose_delta_local=pose_delta_local.astype(np.float32),
        kinematics=kinematics.astype(np.float32),
        ego_valid=np.asarray(ego_valid, dtype=bool),
        depth_8x_m=depth_8x_m.astype(np.float32),
        depth_valid_8x=depth_valid_8x.astype(bool),
        dynamic_mask_8x=np.asarray(dynamic_mask_8x, dtype=bool),
        track_lag_indices=np.asarray(track_lag_indices, dtype=np.int64),
        track_target_sparse=track_target_sparse.astype(np.float32),
        track_valid_sparse=track_valid_sparse.astype(bool),
    )
