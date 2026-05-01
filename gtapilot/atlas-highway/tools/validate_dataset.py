from __future__ import annotations

import argparse
from collections import Counter

from ..config import AtlasHAConfig
from ..dataset import AtlasHAJsonlDataset


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate Atlas-HA JSONL dataset shape contracts.")
    parser.add_argument("--samples-jsonl", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    dataset = AtlasHAJsonlDataset(args.samples_jsonl, AtlasHAConfig(), validate=True)
    candidates = Counter()
    for sample in dataset:
        candidates[int(sample.targets["target_candidate"].reshape(-1)[0].item())] += 1
    print(f"validated_samples={len(dataset)}")
    print(f"candidate_histogram={dict(sorted(candidates.items()))}")


if __name__ == "__main__":
    main()
