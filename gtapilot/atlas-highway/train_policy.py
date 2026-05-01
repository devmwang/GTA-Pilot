from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from .config import AtlasHAConfig
from .dataset import AtlasHAJsonlDataset, collate_atlas_ha_samples
from .losses import compute_atlas_ha_losses
from .model import AtlasHA


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HA-B/HA-C cached-context and legality training entrypoint.")
    parser.add_argument("--samples-jsonl", required=True)
    parser.add_argument("--stage", choices=("ha-b", "ha-c"), default="ha-b")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    cfg = AtlasHAConfig(pretrained_backbone=not args.no_pretrained)
    dataset = AtlasHAJsonlDataset(args.samples_jsonl, cfg)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_atlas_ha_samples,
    )
    model = AtlasHA(cfg).to(args.device)
    if args.freeze_backbone:
        for parameter in model.backbone.parameters():
            parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
    )
    model.train()
    for step, batch in enumerate(loader):
        if step >= args.max_steps:
            break
        scene = batch["scene_rgb_current"].to(args.device)
        actions = batch["actions_hist"].to(args.device)
        dt_hist = batch["dt_hist"].to(args.device)
        nav_cmd = batch["nav_cmd"].to(args.device)
        targets = {key: value.to(args.device) for key, value in batch["targets"].items()}
        outputs, _ = model(scene, actions, dt_hist, nav_cmd=nav_cmd)
        losses = compute_atlas_ha_losses(outputs, targets, cfg)
        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        optimizer.step()
        print(f"stage={args.stage} step={step} total={float(losses['total'].detach().cpu()):.4f}")


if __name__ == "__main__":
    main()
