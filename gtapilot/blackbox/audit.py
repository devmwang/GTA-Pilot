from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * max(0.0, min(100.0, percentile)) / 100.0
    lower_index = int(position)
    upper_index = min(len(ordered) - 1, lower_index + 1)
    blend = position - lower_index
    lower = ordered[lower_index]
    upper = ordered[upper_index]
    return lower + (upper - lower) * blend


def _delta_ms(values_ns: list[int]) -> dict[str, float]:
    if not values_ns:
        return {"p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    values_ms = [float(value) / 1_000_000.0 for value in values_ns]
    return {
        "p50_ms": _percentile(values_ms, 50.0),
        "p95_ms": _percentile(values_ms, 95.0),
        "max_ms": max(values_ms),
    }


def _sequence_gap_summary(values: list[int]) -> dict[str, int]:
    gap_count = 0
    gap_total = 0
    gap_max = 0
    last_value: int | None = None
    for value in values:
        if last_value is not None and value > last_value + 1:
            gap = value - last_value - 1
            gap_count += 1
            gap_total += gap
            gap_max = max(gap_max, gap)
        last_value = value
    return {
        "gap_count": int(gap_count),
        "gap_total": int(gap_total),
        "gap_max": int(gap_max),
    }


def _fresh_capture_gap_summary(values: list[int]) -> dict[str, int]:
    fresh_values: list[int] = []
    last_value: int | None = None
    for value in values:
        if last_value is None or value != last_value:
            fresh_values.append(value)
            last_value = value
    return _sequence_gap_summary(fresh_values)


def _format_float(value: float) -> str:
    return f"{value:.3f}"


def _load_manifest(metadata_path: Path) -> dict[str, Any]:
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit blackbox clip integrity and cadence from capture metadata.",
    )
    parser.add_argument("--metadata-path", required=True)
    args = parser.parse_args()

    metadata_path = Path(args.metadata_path).resolve()
    manifest = _load_manifest(metadata_path)
    frames = list(manifest.get("frames", []))
    frame_ids = [int(frame.get("frame_id", -1)) for frame in frames if frame.get("frame_id") is not None]
    capture_frame_ids = [
        int(frame.get("capture_frame_id", frame.get("frame_id", -1)))
        for frame in frames
    ]
    capture_timestamps_ns = [
        int(frame.get("capture_timestamp_ns", 0))
        for frame in frames
        if frame.get("capture_timestamp_ns") is not None
    ]
    publish_timestamps_ns = [
        int(frame.get("publish_timestamp_ns", 0))
        for frame in frames
        if frame.get("publish_timestamp_ns") is not None
    ]
    fresh_capture_timestamps_ns: list[int] = []
    fresh_last_capture_frame_id: int | None = None
    for frame in frames:
        capture_frame_id = int(frame.get("capture_frame_id", frame.get("frame_id", -1)))
        capture_timestamp_ns = int(frame.get("capture_timestamp_ns", 0))
        if fresh_last_capture_frame_id is None or capture_frame_id != fresh_last_capture_frame_id:
            fresh_capture_timestamps_ns.append(capture_timestamp_ns)
            fresh_last_capture_frame_id = capture_frame_id

    published_dt_ns = [
        later - earlier
        for earlier, later in zip(publish_timestamps_ns, publish_timestamps_ns[1:])
        if later >= earlier
    ]
    fresh_dt_ns = [
        later - earlier
        for earlier, later in zip(
            fresh_capture_timestamps_ns,
            fresh_capture_timestamps_ns[1:],
        )
        if later >= earlier
    ]
    repeat_frame_count = sum(
        1
        for frame in frames
        if bool((frame.get("frame_metadata") or {}).get("is_repeat", False))
    )
    writer_lag_ns = [
        int(frame.get("writer_committed_timestamp_ns", 0))
        - int(frame.get("subscriber_received_timestamp_ns", 0))
        for frame in frames
        if frame.get("writer_committed_timestamp_ns") is not None
        and frame.get("subscriber_received_timestamp_ns") is not None
    ]
    transport_stats = dict(manifest.get("transport_stats", {}))
    writer_stats = dict(manifest.get("writer_stats", {}))
    native_pipeline = dict((manifest.get("session_stats") or {}).get("native_pipeline", {}))

    published_gap_summary = _sequence_gap_summary(frame_ids)
    fresh_gap_summary = _fresh_capture_gap_summary(capture_frame_ids)
    published_dt = _delta_ms(published_dt_ns)
    fresh_dt = _delta_ms(fresh_dt_ns)
    writer_lag = _delta_ms(writer_lag_ns)

    print(f"metadata_path={metadata_path}")
    print(f"schema_version={manifest.get('schema_version')}")
    print(
        f"session_integrity_status="
        f"{(manifest.get('session_integrity') or {}).get('status', 'unknown')}"
    )
    print(f"frame_count={len(frames)}")
    print(f"action_count={len(manifest.get('actions', []))}")
    print(f"repeat_frame_count={repeat_frame_count}")
    print(
        "published_dt_ms="
        f"p50={_format_float(published_dt['p50_ms'])} "
        f"p95={_format_float(published_dt['p95_ms'])} "
        f"max={_format_float(published_dt['max_ms'])}"
    )
    print(
        "fresh_capture_dt_ms="
        f"p50={_format_float(fresh_dt['p50_ms'])} "
        f"p95={_format_float(fresh_dt['p95_ms'])} "
        f"max={_format_float(fresh_dt['max_ms'])}"
    )
    print(
        "published_frame_id_gaps="
        f"count={published_gap_summary['gap_count']} "
        f"total={published_gap_summary['gap_total']} "
        f"max={published_gap_summary['gap_max']}"
    )
    print(
        "capture_frame_id_gaps="
        f"count={fresh_gap_summary['gap_count']} "
        f"total={fresh_gap_summary['gap_total']} "
        f"max={fresh_gap_summary['gap_max']}"
    )
    print(
        "writer_lag_ms="
        f"p50={_format_float(writer_lag['p50_ms'])} "
        f"p95={_format_float(writer_lag['p95_ms'])} "
        f"max={_format_float(writer_lag['max_ms'])}"
    )
    print(
        "vision_transport="
        + json.dumps(transport_stats.get("vision", {}), sort_keys=True)
    )
    print(
        "action_transport="
        + json.dumps(transport_stats.get("actions", {}), sort_keys=True)
    )
    print(
        "writer_stats="
        + json.dumps(writer_stats, sort_keys=True)
    )
    print(
        "native_pipeline="
        + json.dumps(native_pipeline, sort_keys=True)
    )
    print(f"drop_event_count={len(manifest.get('drop_events', []))}")


if __name__ == "__main__":
    main()
