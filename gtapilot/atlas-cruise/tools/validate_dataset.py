from __future__ import annotations

import argparse
from collections import Counter

from ..config import AtlasCruiseConfig
from ..dataset import AtlasCruiseBlackboxDataset, AtlasCruiseJsonlDataset


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate Atlas-Cruise dataset shape and target contracts.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--recordings-root")
    source.add_argument("--samples-jsonl")
    parser.add_argument("--split-file")
    parser.add_argument("--metadata-path", action="append", dest="metadata_paths")
    parser.add_argument("--max-samples", type=int, default=100)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    cfg = AtlasCruiseConfig()
    if args.samples_jsonl:
        dataset = AtlasCruiseJsonlDataset(args.samples_jsonl, cfg, validate=True)
    else:
        dataset = AtlasCruiseBlackboxDataset(
            args.recordings_root,
            cfg,
            metadata_paths=args.metadata_paths,
            split_file=args.split_file,
            validate=True,
        )
    valid_points = Counter()
    slow_positive = 0
    fallback_positive = 0
    count = min(len(dataset), max(0, int(args.max_samples)))
    for idx in range(count):
        sample = dataset[idx]
        valid_points[int(sample.targets["target_traj_valid"].sum().item())] += 1
        slow_positive += int(sample.targets["target_slow_or_brake"].item() >= 0.5)
        fallback_positive += int(sample.targets["target_fallback"].item() >= 0.5)
    print(f"dataset_samples={len(dataset)}")
    print(f"validated_samples={count}")
    print(f"valid_point_histogram={dict(sorted(valid_points.items()))}")
    print(f"slow_or_brake_positive={slow_positive}")
    print(f"fallback_positive={fallback_positive}")


if __name__ == "__main__":
    main()
