from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..config import build_atlas_config
from .privileged_extract import write_privileged_clip
from .privileged_targets import build_stage1b_targets
from .schema import BlackboxFrameRecord


def _load_source_frames(metadata_path: Path) -> tuple[dict, list[BlackboxFrameRecord]]:
    manifest = json.loads(metadata_path.read_text(encoding="utf-8"))
    if int(manifest.get("schema_version", 0)) != 2:
        raise RuntimeError(f"Unsupported blackbox manifest schema: {metadata_path}")
    frames = [
        BlackboxFrameRecord.from_dict(frame_payload)
        for frame_payload in manifest.get("frames", [])
    ]
    if not frames:
        raise RuntimeError(f"No frames found in source metadata: {metadata_path}")
    return manifest, frames


def _validate_raw_length(name: str, array: np.ndarray, expected: int) -> None:
    if int(array.shape[0]) != int(expected):
        raise ValueError(
            f"Raw privileged array '{name}' has length {array.shape[0]}, expected {expected}."
        )


def _alignment_indices(
    *,
    source_frame_ids: np.ndarray,
    source_capture_timestamps_ns: np.ndarray,
    raw_frame_ids: np.ndarray | None,
    raw_capture_timestamps_ns: np.ndarray | None,
) -> np.ndarray:
    if raw_frame_ids is not None:
        frame_id_to_index: dict[int, int] = {}
        for raw_index, frame_id in enumerate(np.asarray(raw_frame_ids, dtype=np.int64).tolist()):
            if frame_id in frame_id_to_index:
                raise ValueError(f"Duplicate raw privileged frame id detected: {frame_id}")
            frame_id_to_index[int(frame_id)] = int(raw_index)
        try:
            aligned = np.asarray(
                [frame_id_to_index[int(frame_id)] for frame_id in source_frame_ids.tolist()],
                dtype=np.int64,
            )
        except KeyError as exc:
            raise ValueError(
                f"Missing privileged frame id for source frame {int(exc.args[0])}."
            ) from exc
        if raw_capture_timestamps_ns is not None:
            selected_timestamps = np.asarray(raw_capture_timestamps_ns, dtype=np.int64)[aligned]
            if not np.array_equal(selected_timestamps, source_capture_timestamps_ns):
                raise ValueError(
                    "Privileged timestamps do not match the source metadata after frame-id alignment."
                )
        return aligned

    if raw_capture_timestamps_ns is not None:
        timestamp_to_index: dict[int, int] = {}
        for raw_index, timestamp_ns in enumerate(
            np.asarray(raw_capture_timestamps_ns, dtype=np.int64).tolist()
        ):
            if timestamp_ns in timestamp_to_index:
                raise ValueError(
                    f"Duplicate raw privileged capture timestamp detected: {timestamp_ns}"
                )
            timestamp_to_index[int(timestamp_ns)] = int(raw_index)
        try:
            return np.asarray(
                [
                    timestamp_to_index[int(timestamp_ns)]
                    for timestamp_ns in source_capture_timestamps_ns.tolist()
                ],
                dtype=np.int64,
            )
        except KeyError as exc:
            raise ValueError(
                f"Missing privileged capture timestamp for source frame {int(exc.args[0])}."
            ) from exc

    raise ValueError(
        "Privileged alignment requires either --raw-frame-ids or --raw-capture-timestamps-ns."
    )


def _aligned_dt_seconds(
    *,
    raw_dt_seconds: np.ndarray | None,
    aligned_indices: np.ndarray,
    aligned_capture_timestamps_ns: np.ndarray,
) -> np.ndarray:
    if raw_dt_seconds is not None:
        return np.asarray(raw_dt_seconds, dtype=np.float32)[aligned_indices]

    dt = np.zeros((aligned_capture_timestamps_ns.shape[0],), dtype=np.float32)
    if aligned_capture_timestamps_ns.shape[0] <= 1:
        dt[0] = 1.0 / 60.0
        return dt
    deltas = np.diff(aligned_capture_timestamps_ns).astype(np.float64) / 1_000_000_000.0
    dt[1:] = deltas.astype(np.float32)
    dt[0] = dt[1]
    return dt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Align raw GTA privileged arrays to a blackbox clip and build Stage 1B targets.",
    )
    parser.add_argument("--variant", default="student")
    parser.add_argument("--metadata-path", required=True)
    parser.add_argument("--source-video-file")
    parser.add_argument("--raw-frame-ids")
    parser.add_argument("--raw-capture-timestamps-ns")
    parser.add_argument("--raw-pose-xyyaw-world", required=True)
    parser.add_argument("--raw-dt-seconds")
    parser.add_argument("--raw-depth-8x")
    parser.add_argument("--raw-depth-native")
    parser.add_argument("--raw-dynamic-mask-8x")
    parser.add_argument("--raw-ego-valid")
    args = parser.parse_args()

    metadata_path = Path(args.metadata_path)
    manifest, frames = _load_source_frames(metadata_path)
    cfg = build_atlas_config(args.variant)

    source_frame_ids = np.asarray([frame.frame_id for frame in frames], dtype=np.int64)
    source_capture_timestamps_ns = np.asarray(
        [frame.capture_timestamp_ns for frame in frames],
        dtype=np.int64,
    )
    source_video_frame_indices = np.asarray(
        [frame.video_frame_index for frame in frames],
        dtype=np.int64,
    )

    raw_pose_xyyaw_world = np.load(args.raw_pose_xyyaw_world)
    raw_dt_seconds = None if args.raw_dt_seconds is None else np.load(args.raw_dt_seconds)
    raw_frame_ids = None if args.raw_frame_ids is None else np.load(args.raw_frame_ids)
    raw_capture_timestamps_ns = (
        None
        if args.raw_capture_timestamps_ns is None
        else np.load(args.raw_capture_timestamps_ns)
    )
    raw_depth_8x = None if args.raw_depth_8x is None else np.load(args.raw_depth_8x)
    raw_depth_native = None if args.raw_depth_native is None else np.load(args.raw_depth_native)
    raw_dynamic_mask_8x = (
        None if args.raw_dynamic_mask_8x is None else np.load(args.raw_dynamic_mask_8x)
    )
    raw_ego_valid = None if args.raw_ego_valid is None else np.load(args.raw_ego_valid)

    raw_length = int(raw_pose_xyyaw_world.shape[0])
    for name, array in (
        ("raw_dt_seconds", raw_dt_seconds),
        ("raw_frame_ids", raw_frame_ids),
        ("raw_capture_timestamps_ns", raw_capture_timestamps_ns),
        ("raw_depth_8x", raw_depth_8x),
        ("raw_depth_native", raw_depth_native),
        ("raw_dynamic_mask_8x", raw_dynamic_mask_8x),
        ("raw_ego_valid", raw_ego_valid),
    ):
        if array is not None:
            _validate_raw_length(name, np.asarray(array), raw_length)

    aligned_indices = _alignment_indices(
        source_frame_ids=source_frame_ids,
        source_capture_timestamps_ns=source_capture_timestamps_ns,
        raw_frame_ids=None if raw_frame_ids is None else np.asarray(raw_frame_ids),
        raw_capture_timestamps_ns=None
        if raw_capture_timestamps_ns is None
        else np.asarray(raw_capture_timestamps_ns),
    )

    aligned_pose_xyyaw_world = np.asarray(raw_pose_xyyaw_world, dtype=np.float32)[aligned_indices]
    aligned_dt_seconds = _aligned_dt_seconds(
        raw_dt_seconds=None if raw_dt_seconds is None else np.asarray(raw_dt_seconds),
        aligned_indices=aligned_indices,
        aligned_capture_timestamps_ns=source_capture_timestamps_ns,
    )
    aligned_depth_8x = (
        None
        if raw_depth_8x is None
        else np.asarray(raw_depth_8x, dtype=np.float32)[aligned_indices]
    )
    aligned_depth_native = (
        None
        if raw_depth_native is None
        else np.asarray(raw_depth_native, dtype=np.float32)[aligned_indices]
    )
    aligned_dynamic_mask_8x = (
        None
        if raw_dynamic_mask_8x is None
        else np.asarray(raw_dynamic_mask_8x, dtype=bool)[aligned_indices]
    )
    aligned_ego_valid = (
        None
        if raw_ego_valid is None
        else np.asarray(raw_ego_valid, dtype=bool)[aligned_indices]
    )

    targets = build_stage1b_targets(
        cfg=cfg,
        pose_xyyaw_world=aligned_pose_xyyaw_world,
        dt_s=aligned_dt_seconds,
        depth_native_m=aligned_depth_native,
        depth_8x_m=aligned_depth_8x,
        dynamic_mask_8x=aligned_dynamic_mask_8x,
        ego_valid=aligned_ego_valid,
    )

    clip_id = metadata_path.stem.replace("_metadata", "")
    source_video_file = (
        args.source_video_file
        or (manifest.get("video_session") or {}).get("file_name")
    )
    output_dir = metadata_path.with_name(f"{clip_id}_privileged")
    manifest_path = write_privileged_clip(
        output_dir=output_dir,
        clip_id=clip_id,
        source_video_file=source_video_file,
        source_metadata_file=metadata_path,
        source_frame_ids=source_frame_ids,
        source_capture_timestamps_ns=source_capture_timestamps_ns,
        source_video_frame_indices=source_video_frame_indices,
        targets=targets,
    )
    print(f"wrote_privileged_manifest={manifest_path}")


if __name__ == "__main__":
    main()
