from __future__ import annotations

import argparse
import time

import torch

from ..config import AtlasHAConfig, AtlasHAStretchConfig
from ..model import AtlasHA


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile an Atlas-HA forward pass.")
    parser.add_argument("--stretch", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--input-h", type=int)
    parser.add_argument("--input-w", type=int)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    cfg = AtlasHAStretchConfig() if args.stretch else AtlasHAConfig()
    if args.no_pretrained:
        cfg.pretrained_backbone = False
    if args.input_h:
        cfg.input_h = args.input_h
    if args.input_w:
        cfg.input_w = args.input_w
    device = torch.device(args.device)
    model = AtlasHA(cfg).to(device).eval()
    scene = torch.rand(1, 3, cfg.input_h, cfg.input_w, device=device)
    actions = torch.zeros(1, cfg.action_history_steps, cfg.action_dim, device=device)
    dt_hist = torch.full((1, cfg.action_history_steps, 1), 1.0 / cfg.action_sample_hz, device=device)
    state = model.init_state(1, device=device)
    with torch.no_grad():
        for _ in range(args.warmup):
            _, state = model(scene, actions, dt_hist, state=state)
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for _ in range(args.iters):
            outputs, state = model(scene, actions, dt_hist, state=state)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    print(f"latency_ms={(elapsed / max(1, args.iters)) * 1000.0:.2f}")
    if device.type == "cuda":
        print(f"peak_vram_mib={torch.cuda.max_memory_allocated() / (1024 * 1024):.1f}")
    for key in (
        "traj_candidates",
        "candidate_logits",
        "lane_lat_pred",
        "lead_state",
        "ego_kinematics",
        "full_ctx_tokens",
        "summary_ctx_tokens",
        "action_tokens",
    ):
        value = outputs[key]
        print(f"{key}: {tuple(value.shape)}")


if __name__ == "__main__":
    main()
