from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import torch

from ..config import AtlasHAConfig
from ..model import AtlasHA
from ..ui import UIPreprocessor
from ..visualization import HighwayDebugFrame, render_highway_debug_overlay, summarize_debug_frame


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay a video through Atlas-HA and render debug overlays.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--output-video")
    parser.add_argument("--output-jsonl")
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-pretrained", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    cfg = AtlasHAConfig(pretrained_backbone=not args.no_pretrained)
    model = AtlasHA(cfg).to(args.device).eval()
    state = model.init_state(1, device=args.device)
    ui = UIPreprocessor(cfg)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open {args.video}")
    writer = None
    if args.output_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output_video, fourcc, 20.0, (1280, 720))
    jsonl_path = Path(args.output_jsonl) if args.output_jsonl else None
    actions = torch.zeros(1, cfg.action_history_steps, cfg.action_dim, device=args.device)
    dt_hist = torch.full((1, cfg.action_history_steps, 1), 1.0 / cfg.action_sample_hz, device=args.device)
    rows = []
    with torch.no_grad():
        for frame_idx in range(args.max_frames):
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            ui_out = ui.preprocess(torch.from_numpy(frame_rgb))
            outputs, state = model(
                ui_out.scene_rgb.unsqueeze(0).to(args.device),
                actions,
                dt_hist,
                ui_mask=ui_out.ui_mask.unsqueeze(0).to(args.device),
                state=state,
            )
            debug = HighwayDebugFrame(
                raw_rgb=frame_rgb,
                sanitized_rgb=ui_out.scene_rgb.cpu(),
                ui_mask=ui_out.ui_mask.cpu(),
                ui_state=ui_out.ui_state,
                outputs=outputs,
            )
            overlay = render_highway_debug_overlay(debug, size=(1280, 720))
            if writer is not None:
                writer.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            rows.append(summarize_debug_frame(debug) | {"frame_idx": frame_idx})
    cap.release()
    if writer is not None:
        writer.release()
    if jsonl_path is not None:
        jsonl_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows), encoding="utf-8")
    print(f"replayed_frames={len(rows)}")


if __name__ == "__main__":
    main()
