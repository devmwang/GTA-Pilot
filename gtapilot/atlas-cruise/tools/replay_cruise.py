from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import torch

from ..config import AtlasCruiseConfig
from ..dataset import AtlasCruiseBlackboxDataset
from ..model import AtlasCruise
from ..trajectory_legalizer import CruiseTrajectoryLegalizer
from ..visualization import trajectory_topdown_image


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay Atlas-Cruise predictions over a blackbox dataset.")
    parser.add_argument("--recordings-root", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--metadata-path", action="append", dest="metadata_paths")
    parser.add_argument("--split-file")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--summary-jsonl")
    parser.add_argument("--topdown-dir")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    cfg = AtlasCruiseConfig(pretrained_backbone=not args.no_pretrained)
    device = torch.device(args.device)
    dataset = AtlasCruiseBlackboxDataset(
        args.recordings_root,
        cfg,
        metadata_paths=args.metadata_paths,
        split_file=args.split_file,
        validate=True,
    )
    model = AtlasCruise(cfg).to(device).eval()
    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(payload.get("model", payload))
    legalizer = CruiseTrajectoryLegalizer(cfg)
    summary_handle = None
    if args.summary_jsonl:
        summary_path = Path(args.summary_jsonl)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_handle = summary_path.open("w", encoding="utf-8")
    topdown_dir = Path(args.topdown_dir) if args.topdown_dir else None
    if topdown_dir is not None:
        topdown_dir.mkdir(parents=True, exist_ok=True)
    try:
        count = min(len(dataset), max(0, int(args.max_samples)))
        with torch.no_grad():
            for idx in range(count):
                sample = dataset[idx]
                outputs = model(
                    sample.rgb_recent.unsqueeze(0).to(device),
                    sample.actions_hist.unsqueeze(0).to(device),
                    sample.dt_hist.unsqueeze(0).to(device),
                    ui_mask_recent=sample.ui_mask_recent.unsqueeze(0).to(device),
                    frame_freshness=sample.frame_freshness.unsqueeze(0).to(device),
                )
                legalized = legalizer.legalize(
                    outputs["traj"][0],
                    slow_or_brake_logit=outputs["slow_or_brake_logit"][0],
                    fallback_logit=outputs["fallback_logit"][0],
                )
                row = {
                    "sample_index": idx,
                    "clip_id": sample.metadata["clip_id"],
                    "anchor_timestamp_ns": sample.metadata["anchor_timestamp_ns"],
                    "traj_conf": float(torch.sigmoid(outputs["traj_conf_logit"])[0, 0].detach().cpu()),
                    "slow_or_brake": float(torch.sigmoid(outputs["slow_or_brake_logit"])[0, 0].detach().cpu()),
                    "fallback": float(torch.sigmoid(outputs["fallback_logit"])[0, 0].detach().cpu()),
                    "legalizer_valid": bool(legalized.valid),
                    "reject_reasons": legalized.reject_reasons,
                }
                print(json.dumps(row, sort_keys=True))
                if summary_handle is not None:
                    summary_handle.write(json.dumps(row, sort_keys=True) + "\n")
                if topdown_dir is not None:
                    image = trajectory_topdown_image(outputs["traj"][0].detach().cpu(), target_traj=sample.targets["target_traj"])
                    cv2.imwrite(str(topdown_dir / f"sample_{idx:06d}.png"), image)
    finally:
        if summary_handle is not None:
            summary_handle.close()


if __name__ == "__main__":
    main()
