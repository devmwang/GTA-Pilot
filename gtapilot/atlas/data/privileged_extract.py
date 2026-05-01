from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from ..config import build_atlas_config
from .privileged_schema import PrivilegedClipManifest
from .privileged_targets import build_stage1b_targets


def write_privileged_clip(
    *,
    output_dir: str | Path,
    clip_id: str,
    source_video_file: str | None,
    source_metadata_file: str | Path | None,
    source_frame_ids: np.ndarray | None,
    source_capture_timestamps_ns: np.ndarray | None,
    source_video_frame_indices: np.ndarray | None = None,
    targets,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "pose_delta_local": "pose_delta_local.npy",
        "kinematics": "kinematics.npy",
        "ego_valid": "ego_valid.npy",
        "depth_8x_m": "depth_8x_m.npy",
        "depth_valid_8x": "depth_valid_8x.npy",
        "dynamic_mask_8x": "dynamic_mask_8x.npy",
        "track_lag_indices": "track_lag_indices.npy",
        "track_target_sparse": "track_target_sparse.npy",
        "track_valid_sparse": "track_valid_sparse.npy",
    }
    np.save(output_dir / files["pose_delta_local"], targets.pose_delta_local)
    np.save(output_dir / files["kinematics"], targets.kinematics)
    np.save(output_dir / files["ego_valid"], targets.ego_valid)
    np.save(output_dir / files["depth_8x_m"], targets.depth_8x_m)
    np.save(output_dir / files["depth_valid_8x"], targets.depth_valid_8x)
    np.save(output_dir / files["dynamic_mask_8x"], targets.dynamic_mask_8x)
    np.save(output_dir / files["track_lag_indices"], targets.track_lag_indices)
    np.save(output_dir / files["track_target_sparse"], targets.track_target_sparse)
    np.save(output_dir / files["track_valid_sparse"], targets.track_valid_sparse)
    frame_ids_file = None
    capture_timestamps_ns_file = None
    video_frame_indices_file = None
    source_metadata_sha1 = None
    if source_frame_ids is not None:
        frame_ids_file = "frame_ids.npy"
        np.save(output_dir / frame_ids_file, np.asarray(source_frame_ids, dtype=np.int64))
    if source_capture_timestamps_ns is not None:
        capture_timestamps_ns_file = "capture_timestamps_ns.npy"
        np.save(
            output_dir / capture_timestamps_ns_file,
            np.asarray(source_capture_timestamps_ns, dtype=np.int64),
        )
    if source_video_frame_indices is not None:
        video_frame_indices_file = "video_frame_indices.npy"
        np.save(
            output_dir / video_frame_indices_file,
            np.asarray(source_video_frame_indices, dtype=np.int64),
        )
    if source_metadata_file is not None:
        source_metadata_path = Path(source_metadata_file)
        source_metadata_sha1 = hashlib.sha1(
            source_metadata_path.read_bytes()
        ).hexdigest()
        source_metadata_file = str(source_metadata_path.resolve())

    manifest = PrivilegedClipManifest(
        clip_id=clip_id,
        source_video_file=source_video_file,
        source_metadata_file=None if source_metadata_file is None else str(source_metadata_file),
        source_metadata_sha1=source_metadata_sha1,
        frame_count=int(targets.depth_8x_m.shape[0]),
        grid_height_8x=int(targets.depth_8x_m.shape[1]),
        grid_width_8x=int(targets.depth_8x_m.shape[2]),
        track_lag_indices=targets.track_lag_indices.tolist(),
        frame_ids_file=frame_ids_file,
        capture_timestamps_ns_file=capture_timestamps_ns_file,
        video_frame_indices_file=video_frame_indices_file,
        files=files,
    )
    manifest_path = output_dir / "manifest.json"
    manifest.save(manifest_path)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Stage 1B privileged targets from raw aligned arrays.",
    )
    parser.add_argument("--variant", default="student")
    parser.add_argument("--metadata-path", required=True)
    parser.add_argument("--source-video-file")
    parser.add_argument("--frame-ids")
    parser.add_argument("--capture-timestamps-ns")
    parser.add_argument("--video-frame-indices")
    parser.add_argument("--pose-xyyaw-world", required=True)
    parser.add_argument("--dt-seconds", required=True)
    parser.add_argument("--depth-8x")
    parser.add_argument("--depth-native")
    parser.add_argument("--dynamic-mask-8x")
    parser.add_argument("--ego-valid")
    args = parser.parse_args()

    cfg = build_atlas_config(args.variant)
    pose_xyyaw_world = np.load(args.pose_xyyaw_world)
    dt_seconds = np.load(args.dt_seconds)
    frame_ids = None if args.frame_ids is None else np.load(args.frame_ids)
    capture_timestamps_ns = (
        None if args.capture_timestamps_ns is None else np.load(args.capture_timestamps_ns)
    )
    video_frame_indices = (
        None if args.video_frame_indices is None else np.load(args.video_frame_indices)
    )
    depth_8x = None if args.depth_8x is None else np.load(args.depth_8x)
    depth_native = None if args.depth_native is None else np.load(args.depth_native)
    dynamic_mask = None if args.dynamic_mask_8x is None else np.load(args.dynamic_mask_8x)
    ego_valid = None if args.ego_valid is None else np.load(args.ego_valid)

    metadata_path = Path(args.metadata_path)
    clip_id = metadata_path.parent.name
    output_dir = metadata_path.parent / "privileged"

    targets = build_stage1b_targets(
        cfg=cfg,
        pose_xyyaw_world=pose_xyyaw_world,
        dt_s=dt_seconds,
        depth_native_m=depth_native,
        depth_8x_m=depth_8x,
        dynamic_mask_8x=dynamic_mask,
        ego_valid=ego_valid,
    )
    manifest_path = write_privileged_clip(
        output_dir=output_dir,
        clip_id=clip_id,
        source_video_file=args.source_video_file,
        source_metadata_file=metadata_path,
        source_frame_ids=frame_ids,
        source_capture_timestamps_ns=capture_timestamps_ns,
        source_video_frame_indices=video_frame_indices,
        targets=targets,
    )
    print(f"wrote_privileged_manifest={manifest_path}")


if __name__ == "__main__":
    main()
