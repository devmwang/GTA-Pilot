from __future__ import annotations

import argparse

import torch

from ..config import AtlasCruiseConfig
from ..model import AtlasCruise


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Print Atlas-Cruise model parameter counts and optional dry-run shapes.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    cfg = AtlasCruiseConfig(pretrained_backbone=not args.no_pretrained)
    model = AtlasCruise(cfg).to(args.device).eval()
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"parameters_total={total}")
    print(f"parameters_trainable={trainable}")
    if args.dry_run:
        with torch.no_grad():
            rgb = torch.zeros(1, cfg.num_visual_frames, 3, cfg.input_h, cfg.input_w, device=args.device)
            actions = torch.zeros(1, cfg.action_history_steps, cfg.action_dim, device=args.device)
            dt = torch.full((1, cfg.action_history_steps, 1), 1.0 / cfg.action_sample_hz, device=args.device)
            freshness = torch.zeros(1, cfg.num_visual_frames, 2, device=args.device)
            outputs = model(rgb, actions, dt, frame_freshness=freshness)
        for key in ("traj", "ego_kinematics", "traj_conf_logit", "slow_or_brake_logit", "fallback_logit", "control_aux"):
            if key in outputs:
                print(f"{key}_shape={tuple(outputs[key].shape)}")


if __name__ == "__main__":
    main()
