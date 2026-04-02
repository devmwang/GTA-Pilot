from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TIMING_KEYS = (
    ("frame_arrival_wait_ns", "native_frame_arrival_wait_ms"),
    ("gpu_readback_ns", "native_gpu_readback_ms"),
    ("cpu_convert_ns", "native_cpu_convert_ms"),
    ("publish_deadline_lag_ns", "publish_deadline_lag_ms"),
)


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


def _summary_ms(values_ns: list[int]) -> dict[str, float]:
    if not values_ns:
        return {"p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    values_ms = [float(value) / 1_000_000.0 for value in values_ns]
    return {
        "p50_ms": _percentile(values_ms, 50.0),
        "p95_ms": _percentile(values_ms, 95.0),
        "max_ms": max(values_ms),
    }


def _bucket_ms(bucket: dict[str, Any] | None) -> dict[str, float]:
    payload = dict(bucket or {})
    return {
        "p50_ms": float(payload.get("p50", 0)) / 1_000_000.0,
        "p95_ms": float(payload.get("p95", 0)) / 1_000_000.0,
        "max_ms": float(payload.get("max", 0)) / 1_000_000.0,
    }


def _gap_summary(values: list[int], *, dedupe: bool = False) -> dict[str, int]:
    gap_count = 0
    gap_total = 0
    gap_max = 0
    last_value: int | None = None
    for value in values:
        if dedupe and value == last_value:
            continue
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


def _timestamp_deltas(values_ns: list[int]) -> list[int]:
    return [
        later - earlier
        for earlier, later in zip(values_ns, values_ns[1:])
        if later >= earlier
    ]


def _fresh_capture_timestamps_ns(frames: list[dict[str, Any]]) -> list[int]:
    timestamps_ns: list[int] = []
    last_capture_frame_id: int | None = None
    for frame in frames:
        capture_frame_id = int(frame.get("capture_frame_id", frame.get("frame_id", -1)))
        if capture_frame_id == last_capture_frame_id:
            continue
        timestamps_ns.append(int(frame.get("capture_timestamp_ns", 0)))
        last_capture_frame_id = capture_frame_id
    return timestamps_ns


def _native_timing_summary(
    frames: list[dict[str, Any]],
    native_capture_perf: dict[str, Any],
) -> dict[str, dict[str, float]]:
    if native_capture_perf:
        return {
            metric_name: _bucket_ms(native_capture_perf.get(key))
            for key, metric_name in TIMING_KEYS
        }

    sample_capture_frame_id: int | None = None
    samples: dict[str, list[int]] = {key: [] for key, _ in TIMING_KEYS}
    for frame in frames:
        pipeline_stats = (frame.get("frame_metadata") or {}).get("pipeline_stats")
        if not isinstance(pipeline_stats, dict):
            continue
        current_sample_capture_frame_id = int(
            pipeline_stats.get(
                "sample_capture_frame_id",
                frame.get("capture_frame_id", frame.get("frame_id", -1)),
            )
        )
        if sample_capture_frame_id == current_sample_capture_frame_id:
            continue
        sample_capture_frame_id = current_sample_capture_frame_id
        for key in samples:
            value = pipeline_stats.get(key)
            if value is not None:
                samples[key].append(int(value))

    return {
        metric_name: _summary_ms(samples[key])
        for key, metric_name in TIMING_KEYS
    }


def _format_ms(summary: dict[str, float]) -> str:
    return (
        f"p50={summary['p50_ms']:.3f} "
        f"p95={summary['p95_ms']:.3f} "
        f"max={summary['max_ms']:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit blackbox clip integrity and cadence from capture metadata.",
    )
    parser.add_argument("--metadata-path", required=True)
    args = parser.parse_args()

    metadata_path = Path(args.metadata_path).resolve()
    manifest = json.loads(metadata_path.read_text(encoding="utf-8"))
    frames = list(manifest.get("frames", []))
    session_integrity = dict(manifest.get("session_integrity", {}))
    session_stats = dict(manifest.get("session_stats", {}))
    performance_stats = dict(manifest.get("performance_stats", {}))
    native_capture_perf = dict(performance_stats.get("native_capture", {}))

    frame_ids = [
        int(frame.get("frame_id", -1))
        for frame in frames
        if frame.get("frame_id") is not None
    ]
    capture_frame_ids = [
        int(frame.get("capture_frame_id", frame.get("frame_id", -1)))
        for frame in frames
    ]
    publish_timestamps_ns = [
        int(frame.get("publish_timestamp_ns", 0))
        for frame in frames
        if frame.get("publish_timestamp_ns") is not None
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

    native_timing_summary = _native_timing_summary(frames, native_capture_perf)
    published_gap_summary = _gap_summary(frame_ids)
    fresh_gap_summary = _gap_summary(capture_frame_ids, dedupe=True)
    published_dt = _summary_ms(_timestamp_deltas(publish_timestamps_ns))
    fresh_dt = _summary_ms(_timestamp_deltas(_fresh_capture_timestamps_ns(frames)))
    writer_lag = _summary_ms(writer_lag_ns)

    print(f"metadata_path={metadata_path}")
    print(f"schema_version={manifest.get('schema_version')}")
    print(
        "session_integrity_status="
        f"{session_integrity.get('status', 'unknown')}"
    )
    print(
        "session_integrity_ignored_drop_event_count="
        f"{int(session_integrity.get('ignored_drop_event_count', 0))}"
    )
    print(
        "session_integrity_edge_grace_window_seconds="
        f"{float(session_integrity.get('edge_grace_window_seconds', 0.0)):.1f}"
    )
    print(f"frame_count={len(frames)}")
    print(f"action_count={len(manifest.get('actions', []))}")
    print(f"repeat_frame_count={repeat_frame_count}")
    print(f"published_dt_ms={_format_ms(published_dt)}")
    print(f"fresh_capture_dt_ms={_format_ms(fresh_dt)}")
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
    print(f"writer_lag_ms={_format_ms(writer_lag)}")
    for _key, metric_name in TIMING_KEYS:
        print(f"{metric_name}={_format_ms(native_timing_summary[metric_name])}")
    print(
        "vision_transport="
        + json.dumps(dict((manifest.get("transport_stats") or {}).get("vision", {})), sort_keys=True)
    )
    print(
        "action_transport="
        + json.dumps(dict((manifest.get("transport_stats") or {}).get("actions", {})), sort_keys=True)
    )
    print(
        "writer_stats=" + json.dumps(dict(manifest.get("writer_stats", {})), sort_keys=True)
    )
    print(
        "native_pipeline="
        + json.dumps(dict(session_stats.get("native_pipeline", {})), sort_keys=True)
    )
    print(
        "blackbox_ingest="
        + json.dumps(dict(performance_stats.get("blackbox_ingest", {})), sort_keys=True)
    )
    print(f"drop_event_count={len(manifest.get('drop_events', []))}")
    print(f"ignored_drop_event_count={len(manifest.get('ignored_drop_events', []))}")


if __name__ == "__main__":
    main()
