from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, random_split

from .config import AtlasCruiseConfig
from .dataset import (
    AtlasCruiseBlackboxDataset,
    AtlasCruiseJsonlDataset,
    collate_atlas_cruise_samples,
)
from .losses import compute_atlas_cruise_losses
from .model import AtlasCruise


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Atlas-Cruise trajectory/control model.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--recordings-root", help="Root containing blackbox capture_* clips with Cruise target packages.")
    source.add_argument("--samples-jsonl", help="Prebuilt Atlas-Cruise JSONL sample index.")
    parser.add_argument("--split-file", help="Optional list of metadata.json paths for blackbox dataset.")
    parser.add_argument("--metadata-path", action="append", dest="metadata_paths", help="Explicit metadata.json path; may be repeated.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--anchor-stride-frames", type=int, default=3)
    parser.add_argument("--max-samples-per-clip", type=int)
    parser.add_argument("--val-fraction", type=float, default=0.0)
    parser.add_argument("--checkpoint-out", default="atlas_cruise_checkpoint.pt")
    return parser


def _build_dataset(args: argparse.Namespace, cfg: AtlasCruiseConfig):
    if args.samples_jsonl:
        return AtlasCruiseJsonlDataset(args.samples_jsonl, cfg)
    return AtlasCruiseBlackboxDataset(
        args.recordings_root,
        cfg,
        metadata_paths=args.metadata_paths,
        split_file=args.split_file,
        anchor_stride_frames=args.anchor_stride_frames,
        max_samples_per_clip=args.max_samples_per_clip,
    )


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = {
        "rgb_recent": batch["rgb_recent"].to(device, non_blocking=device.type == "cuda"),
        "actions_hist": batch["actions_hist"].to(device, non_blocking=device.type == "cuda"),
        "dt_hist": batch["dt_hist"].to(device, non_blocking=device.type == "cuda"),
        "ui_mask_recent": batch["ui_mask_recent"].to(device, non_blocking=device.type == "cuda"),
        "frame_freshness": batch["frame_freshness"].to(device, non_blocking=device.type == "cuda"),
        "targets": {
            key: value.to(device, non_blocking=device.type == "cuda")
            for key, value in batch["targets"].items()
        },
        "metadata": batch["metadata"],
    }
    return moved


def _run_validation(
    model: AtlasCruise,
    loader: DataLoader,
    cfg: AtlasCruiseConfig,
    device: torch.device,
    *,
    amp: bool,
) -> float:
    model.eval()
    totals: list[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, device)
            with torch.amp.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                outputs = model(
                    batch["rgb_recent"],
                    batch["actions_hist"],
                    batch["dt_hist"],
                    ui_mask_recent=batch["ui_mask_recent"],
                    frame_freshness=batch["frame_freshness"],
                )
                losses = compute_atlas_cruise_losses(outputs, batch["targets"], cfg)
            totals.append(float(losses["total"].detach().cpu()))
    model.train()
    return sum(totals) / max(1, len(totals))


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    cfg = AtlasCruiseConfig(pretrained_backbone=not args.no_pretrained)
    cfg.validate()
    dataset = _build_dataset(args, cfg)
    if len(dataset) == 0:
        raise RuntimeError("Atlas-Cruise dataset is empty.")
    if args.val_fraction > 0.0:
        val_len = max(1, int(round(len(dataset) * float(args.val_fraction))))
        train_len = max(1, len(dataset) - val_len)
        train_dataset, val_dataset = random_split(dataset, [train_len, len(dataset) - train_len])
    else:
        train_dataset = dataset
        val_dataset = None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_atlas_cruise_samples,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = None
    if val_dataset is not None and len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_atlas_cruise_samples,
            pin_memory=torch.cuda.is_available(),
        )
    device = torch.device(args.device)
    model = AtlasCruise(cfg).to(device)
    if args.freeze_backbone:
        for parameter in model.backbone.parameters():
            parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    model.train()
    global_step = 0
    for epoch in range(args.epochs):
        for batch in train_loader:
            if args.max_steps is not None and global_step >= args.max_steps:
                break
            batch = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
                outputs = model(
                    batch["rgb_recent"],
                    batch["actions_hist"],
                    batch["dt_hist"],
                    ui_mask_recent=batch["ui_mask_recent"],
                    frame_freshness=batch["frame_freshness"],
                )
                losses = compute_atlas_cruise_losses(outputs, batch["targets"], cfg)
            scaler.scale(losses["total"]).backward()
            if args.grad_clip > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            print(
                " ".join(
                    [
                        f"epoch={epoch}",
                        f"step={global_step}",
                        f"total={float(losses['total'].detach().cpu()):.4f}",
                        f"traj_pos={float(losses['traj_pos'].detach().cpu()):.4f}",
                        f"speed={float(losses['traj_speed'].detach().cpu()):.4f}",
                        f"slow={float(losses['slow'].detach().cpu()):.4f}",
                    ]
                )
            )
            global_step += 1
        if val_loader is not None:
            val_total = _run_validation(model, val_loader, cfg, device, amp=args.amp)
            print(f"epoch={epoch} val_total={val_total:.4f}")
        if args.max_steps is not None and global_step >= args.max_steps:
            break
    checkpoint_path = Path(args.checkpoint_out)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "cfg": cfg,
            "global_step": global_step,
        },
        checkpoint_path,
    )
    print(f"wrote_checkpoint={checkpoint_path}")


if __name__ == "__main__":
    main()
