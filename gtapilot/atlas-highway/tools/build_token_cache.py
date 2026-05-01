from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..config import AtlasHAConfig
from ..dataset import AtlasHAJsonlDataset, collate_atlas_ha_samples
from ..model import AtlasHA


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Atlas-HA frame-token caches for HA-B training.")
    parser.add_argument("--samples-jsonl", required=True)
    parser.add_argument("--output-pt", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-pretrained", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    cfg = AtlasHAConfig(pretrained_backbone=not args.no_pretrained)
    dataset = AtlasHAJsonlDataset(args.samples_jsonl, cfg)
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_atlas_ha_samples)
    model = AtlasHA(cfg).to(args.device).eval()
    all_tokens = []
    all_summary = []
    with torch.no_grad():
        for batch in loader:
            frame_tokens, frame_summary, _ = model.encode_scene(
                batch["scene_rgb_current"].to(args.device)
            )
            all_tokens.append(frame_tokens.cpu())
            all_summary.append(frame_summary.cpu())
    output = {
        "frame_tokens": torch.cat(all_tokens, dim=0),
        "frame_summary": torch.cat(all_summary, dim=0),
        "config": cfg,
    }
    torch.save(output, Path(args.output_pt))
    print(f"saved token cache to {args.output_pt}")


if __name__ == "__main__":
    main()
