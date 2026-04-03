from __future__ import annotations

from typing import Any

VISION_FRAMES_CHANNEL_NAME = "vision.frames"
INPUT_ACTIONS_CHANNEL_NAME = "input.actions"


def _native_pipeline_samples(
    frame_payloads: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    last_sample_capture_frame_id: int | None = None
    for frame_payload in frame_payloads:
        pipeline_stats = frame_payload.pop("pipeline_stats", None)
        if not isinstance(pipeline_stats, dict):
            continue
        current_capture_frame_id = int(
            pipeline_stats.get(
                "sample_capture_frame_id",
                frame_payload.get("capture_frame_id", frame_payload.get("frame_id", -1)),
            )
        )
        if current_capture_frame_id == last_sample_capture_frame_id:
            continue
        last_sample_capture_frame_id = current_capture_frame_id
        samples.append(
            {
                "capture_frame_id": current_capture_frame_id,
                "capture_timestamp_ns": int(frame_payload.get("capture_timestamp_ns", 0)),
                "stats": dict(pipeline_stats),
            }
        )
    return samples


def _active_devices_seen(action_payloads: list[dict[str, Any]]) -> list[str]:
    devices = {
        str((action_payload.get("payload") or {}).get("active_device", "none"))
        for action_payload in action_payloads
    }
    return sorted(device for device in devices if device)


def _percentile_ns(values: list[int], percentile: float) -> int:
    if not values:
        return 0
    ordered = sorted(int(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * max(0.0, min(100.0, float(percentile))) / 100.0
    lower_index = int(position)
    upper_index = min(len(ordered) - 1, lower_index + 1)
    blend = position - lower_index
    lower = ordered[lower_index]
    upper = ordered[upper_index]
    return int(round(lower + (upper - lower) * blend))


def _timing_summary_ns(values: list[int]) -> dict[str, int]:
    if not values:
        return {"count": 0, "p50": 0, "p95": 0, "max": 0}
    normalized = [max(0, int(value)) for value in values]
    return {
        "count": int(len(normalized)),
        "p50": _percentile_ns(normalized, 50.0),
        "p95": _percentile_ns(normalized, 95.0),
        "max": max(normalized),
    }


def _frame_gap_stats(frame_payloads: list[dict[str, Any]]) -> dict[str, int]:
    published_gap_count = 0
    published_gap_total = 0
    capture_gap_count = 0
    capture_gap_total = 0
    last_frame_id: int | None = None
    last_capture_frame_id: int | None = None

    for frame_payload in frame_payloads:
        frame_id = int(frame_payload.get("frame_id", -1))
        capture_frame_id = int(frame_payload.get("capture_frame_id", frame_id))
        if last_frame_id is not None and frame_id > last_frame_id + 1:
            published_gap_count += 1
            published_gap_total += frame_id - last_frame_id - 1
        last_frame_id = frame_id

        if last_capture_frame_id is None:
            last_capture_frame_id = capture_frame_id
            continue
        if capture_frame_id == last_capture_frame_id:
            continue
        if capture_frame_id > last_capture_frame_id + 1:
            capture_gap_count += 1
            capture_gap_total += capture_frame_id - last_capture_frame_id - 1
        last_capture_frame_id = capture_frame_id

    return {
        "published_gap_count": int(published_gap_count),
        "published_gap_total": int(published_gap_total),
        "capture_gap_count": int(capture_gap_count),
        "capture_gap_total": int(capture_gap_total),
    }


def _fresh_capture_frame_count(frame_payloads: list[dict[str, Any]]) -> int:
    fresh_capture_frame_count = 0
    last_capture_frame_id: int | None = None
    for frame_payload in frame_payloads:
        capture_frame_id = int(
            frame_payload.get("capture_frame_id", frame_payload.get("frame_id", -1))
        )
        if last_capture_frame_id is None or capture_frame_id != last_capture_frame_id:
            fresh_capture_frame_count += 1
            last_capture_frame_id = capture_frame_id
    return int(fresh_capture_frame_count)


def _repeat_frame_count(frame_payloads: list[dict[str, Any]]) -> int:
    return int(
        sum(1 for frame_payload in frame_payloads if bool(frame_payload.get("is_repeat", False)))
    )


def _session_data_window_ns(
    *,
    frame_payloads: list[dict[str, Any]],
    action_payloads: list[dict[str, Any]],
    created_timestamp_ns: int,
) -> tuple[int, int]:
    timestamps_ns: list[int] = []
    for frame_payload in frame_payloads:
        timestamp_ns = int(
            frame_payload.get(
                "capture_timestamp_ns",
                frame_payload.get("publish_timestamp_ns", 0),
            )
            or 0
        )
        if timestamp_ns > 0:
            timestamps_ns.append(timestamp_ns)

    for action_payload in action_payloads:
        timestamp_ns = int(
            action_payload.get(
                "message_timestamp_ns",
                action_payload.get("subscriber_received_timestamp_ns", 0),
            )
            or 0
        )
        if timestamp_ns > 0:
            timestamps_ns.append(timestamp_ns)

    if not timestamps_ns:
        fallback_timestamp_ns = max(0, int(created_timestamp_ns))
        return fallback_timestamp_ns, fallback_timestamp_ns

    return min(timestamps_ns), max(timestamps_ns)


def _is_grace_filterable_drop_event(event: dict[str, Any]) -> bool:
    kind = str(event.get("kind", ""))
    channel = str(event.get("channel", ""))
    if kind in {"writer_frame_queue_overflow", "writer_action_queue_overflow"}:
        return True
    if kind == "native_overload":
        return True
    if kind in {"sequence_gap", "local_overflow"} and channel in {
        VISION_FRAMES_CHANNEL_NAME,
        INPUT_ACTIONS_CHANNEL_NAME,
    }:
        return True
    return False


def _split_drop_events_for_integrity(
    *,
    drop_events: list[dict[str, Any]],
    session_start_timestamp_ns: int,
    session_end_timestamp_ns: int,
    grace_window_ns: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not drop_events:
        return [], []

    effective_drop_events: list[dict[str, Any]] = []
    ignored_drop_events: list[dict[str, Any]] = []
    grace_window_ns = max(0, int(grace_window_ns))
    for drop_event in drop_events:
        if grace_window_ns <= 0 or not _is_grace_filterable_drop_event(drop_event):
            effective_drop_events.append(drop_event)
            continue

        timestamp_ns = int(drop_event.get("timestamp_ns", 0) or 0)
        if timestamp_ns <= 0:
            effective_drop_events.append(drop_event)
            continue

        if (
            timestamp_ns <= session_start_timestamp_ns + grace_window_ns
            or timestamp_ns >= session_end_timestamp_ns - grace_window_ns
        ):
            ignored_drop_events.append(dict(drop_event))
            continue

        effective_drop_events.append(drop_event)

    return effective_drop_events, ignored_drop_events


def _native_capture_timing_summary(
    native_pipeline_samples: list[dict[str, Any]],
) -> dict[str, Any]:
    latest_pipeline_stats: dict[str, Any] = {}
    frame_arrival_wait_ns: list[int] = []
    gpu_readback_ns: list[int] = []
    cpu_convert_ns: list[int] = []
    publish_deadline_lag_ns: list[int] = []

    for sample in native_pipeline_samples:
        pipeline_stats = sample.get("stats")
        if not isinstance(pipeline_stats, dict):
            continue
        latest_pipeline_stats = dict(pipeline_stats)

        for bucket, key in (
            (frame_arrival_wait_ns, "frame_arrival_wait_ns"),
            (gpu_readback_ns, "gpu_readback_ns"),
            (cpu_convert_ns, "cpu_convert_ns"),
            (publish_deadline_lag_ns, "publish_deadline_lag_ns"),
        ):
            value = pipeline_stats.get(key)
            if value is not None:
                bucket.append(int(value))

    return {
        "sample_count": int(len(frame_arrival_wait_ns)),
        "frame_arrival_wait_ns": _timing_summary_ns(frame_arrival_wait_ns),
        "gpu_readback_ns": _timing_summary_ns(gpu_readback_ns),
        "cpu_convert_ns": _timing_summary_ns(cpu_convert_ns),
        "publish_deadline_lag_ns": _timing_summary_ns(publish_deadline_lag_ns),
        "latest_pipeline_stats": latest_pipeline_stats,
    }


def _session_integrity_payload(
    *,
    frame_payloads: list[dict[str, Any]],
    drop_events: list[dict[str, Any]],
    ignored_drop_events: list[dict[str, Any]],
    edge_grace_window_seconds: float,
) -> dict[str, Any]:
    return {
        "status": "degraded" if drop_events else "ok",
        "drop_event_count": int(len(drop_events)),
        "ignored_drop_event_count": int(len(ignored_drop_events)),
        "edge_grace_window_seconds": float(edge_grace_window_seconds),
        "repeat_frame_count": _repeat_frame_count(frame_payloads),
        "fresh_capture_frame_count": _fresh_capture_frame_count(frame_payloads),
        **_frame_gap_stats(frame_payloads),
    }
