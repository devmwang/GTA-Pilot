from __future__ import annotations

import argparse

from .runtime import run_training_stage
from .train_config import load_trainer_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Atlas Stage 1A camera JEPA.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--recordings-root", type=str, default=None)
    parser.add_argument("--train-split", type=str, default=None)
    parser.add_argument("--val-split", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    cfg = load_trainer_config("stage1a", args.config)
    if args.variant:
        cfg.model_variant = args.variant
    if args.device:
        cfg.device = args.device
    if args.max_steps is not None:
        cfg.max_steps = args.max_steps
    if args.recordings_root:
        cfg.train.recordings_root = args.recordings_root
        cfg.val.recordings_root = args.recordings_root
    if args.train_split:
        cfg.train.split_file = args.train_split
    if args.val_split:
        cfg.val.split_file = args.val_split
    if args.resume:
        cfg.checkpoint.resume_from = args.resume

    result = run_training_stage("stage1a", cfg)
    print(f"stage1a step={result['step']} checkpoint_dir={result['checkpoint_dir']}")


if __name__ == "__main__":
    main()
