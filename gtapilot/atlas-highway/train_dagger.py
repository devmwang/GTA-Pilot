from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HA-D DAgger-lite failure mining queue builder.")
    parser.add_argument("--failure-log", required=True, help="JSONL log from live_highway_assist or replay_highway.")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--max-items", type=int, default=10000)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    input_path = Path(args.failure_log)
    rows = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    interesting = []
    failure_keys = (
        "lane_departure",
        "collision",
        "unsafe_lane_change_attempt",
        "lane_change_abort",
        "phone_ui_event",
        "stale_frame_burst",
        "minimum_risk_activation",
        "takeover_requested",
    )
    for row in rows:
        if len(interesting) >= args.max_items:
            break
        if any(bool(row.get(key, False)) for key in failure_keys):
            interesting.append(row)
    output_path = Path(args.output_jsonl)
    output_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in interesting),
        encoding="utf-8",
    )
    print(f"wrote {len(interesting)} DAgger-lite queue items to {output_path}")


if __name__ == "__main__":
    main()
