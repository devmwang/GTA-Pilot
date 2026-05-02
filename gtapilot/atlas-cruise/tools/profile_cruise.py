from __future__ import annotations

import argparse
import time

import torch

from ..config import AtlasCruiseConfig
from ..model import AtlasCruise


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile Atlas-Cruise dummy forward latency.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    cfg = AtlasCruiseConfig(pretrained_backbone=not args.no_pretrained)
    device = torch.device(args.device)
    model = AtlasCruise(cfg).to(device).eval()
    rgb = torch.zeros(1, cfg.num_visual_frames, 3, cfg.input_h, cfg.input_w, device=device)
    actions = torch.zeros(1, cfg.action_history_steps, cfg.action_dim, device=device)
    dt = torch.full((1, cfg.action_history_steps, 1), 1.0 / cfg.action_sample_hz, device=device)
    freshness = torch.zeros(1, cfg.num_visual_frames, 2, device=device)
    with torch.no_grad():
        for _ in range(args.warmup):
            model(rgb, actions, dt, frame_freshness=freshness)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(args.iters):
            model(rgb, actions, dt, frame_freshness=freshness)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    print(f"iters={args.iters}")
    print(f"avg_forward_ms={(elapsed / max(1, args.iters)) * 1000.0:.3f}")


if __name__ == "__main__":
    main()
