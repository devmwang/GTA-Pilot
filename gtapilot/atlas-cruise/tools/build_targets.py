from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..config import AtlasCruiseConfig
from ..dataset import _load_timeline
from ..targets import build_cruise_target_arrays, load_ego_telemetry_source


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Atlas-Cruise target package for a blackbox clip.")
    parser.add_argument("--metadata-path", required=True)
    parser.add_argument("--ego-source", required=True, help="Ego telemetry .jsonl/.json/.npz with timestamps, pose, yaw, and speed.")
    parser.add_argument("--output-dir", help="Defaults to capture_<timestamp>/privileged.")
    parser.add_argument("--no-control-aux", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    cfg = AtlasCruiseConfig(predict_control_aux=not args.no_control_aux)
    cfg.validate()
    metadata_path = Path(args.metadata_path).resolve()
    timeline = _load_timeline(metadata_path)
    ego = load_ego_telemetry_source(args.ego_source)
    targets = build_cruise_target_arrays(
        timeline.frame_timestamps_ns,
        ego,
        cfg,
        action_timestamps_ns=timeline.action_timestamps_ns,
        action_vectors=timeline.action_vectors,
    )
    output_dir = Path(args.output_dir).resolve() if args.output_dir else metadata_path.parent / "privileged"
    output_dir.mkdir(parents=True, exist_ok=True)
    package_path = output_dir / cfg.target_package_name
    np.savez_compressed(package_path, **targets)
    manifest = {
        "schema_version": 1,
        "kind": "atlas_cruise_targets",
        "clip_id": timeline.clip_id,
        "source_metadata_file": str(metadata_path),
        "source_actions_file": str(timeline.actions_path),
        "ego_source": str(Path(args.ego_source).resolve()),
        "target_package": cfg.target_package_name,
        "frame_count": int(timeline.frame_timestamps_ns.shape[0]),
        "traj_points": cfg.traj_points,
        "traj_horizon_s": cfg.traj_horizon_s,
        "target_latency_s": cfg.target_latency_s,
        "target_valid_fraction": float(np.asarray(targets["target_traj_valid"]).mean()),
        "fallback_positive_fraction": float(np.asarray(targets["target_fallback"]).mean()),
        "slow_or_brake_positive_fraction": float(np.asarray(targets["target_slow_or_brake"]).mean()),
    }
    manifest_path = output_dir / cfg.target_manifest_name
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote_target_package={package_path}")
    print(f"wrote_target_manifest={manifest_path}")


if __name__ == "__main__":
    main()
