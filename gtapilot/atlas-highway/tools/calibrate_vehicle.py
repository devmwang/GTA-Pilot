from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ..actuator_calibration import ActuatorCalibration, estimate_steer_gain


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit a conservative Atlas-HA actuator calibration from CSV samples.")
    parser.add_argument("--csv", help="CSV with columns command,radius_m,speed_mps for steering runs.")
    parser.add_argument("--output-json", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    calibration = ActuatorCalibration()
    if args.csv:
        gains = []
        with Path(args.csv).open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                gains.append(
                    estimate_steer_gain(
                        float(row["radius_m"]),
                        float(row.get("speed_mps", 0.0)),
                        float(row["command"]),
                    )
                )
        if gains:
            calibration.steer_gain = sum(gains) / len(gains)
    calibration.to_json_file(args.output_json)
    print(f"wrote calibration to {args.output_json}")


if __name__ == "__main__":
    main()
