from __future__ import annotations

import argparse
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

    manifest = PrivilegedClipManifest(
        clip_id=clip_id,
        source_video_file=source_video_file,
        frame_count=int(targets.depth_8x_m.shape[0]),
        grid_height_8x=int(targets.depth_8x_m.shape[1]),
        grid_width_8x=int(targets.depth_8x_m.shape[2]),
        track_lag_indices=targets.track_lag_indices.tolist(),
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
    depth_8x = None if args.depth_8x is None else np.load(args.depth_8x)
    depth_native = None if args.depth_native is None else np.load(args.depth_native)
    dynamic_mask = None if args.dynamic_mask_8x is None else np.load(args.dynamic_mask_8x)
    ego_valid = None if args.ego_valid is None else np.load(args.ego_valid)

    metadata_path = Path(args.metadata_path)
    clip_id = metadata_path.stem.replace("_metadata", "")
    output_dir = metadata_path.with_name(f"{clip_id}_privileged")

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
        targets=targets,
    )
    print(f"wrote_privileged_manifest={manifest_path}")


if __name__ == "__main__":
    main()
