from __future__ import annotations

import argparse

import torch

from ..config import AtlasHAConfig, AtlasHAStretchConfig
from ..model import AtlasHA


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Print Atlas-HA model size and rough runtime budget.")
    parser.add_argument("--stretch", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    cfg = AtlasHAStretchConfig() if args.stretch else AtlasHAConfig()
    if args.no_pretrained:
        cfg.pretrained_backbone = False
    model = AtlasHA(cfg)
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    cache_mib = cfg.full_context_frames * cfg.tokens_per_frame * cfg.hidden_dim * 2 / (1024 * 1024)
    kv_mib = cache_mib * 2 * cfg.full_context_blocks if cfg.cache_projected_kv else 0.0
    flops_ref = 0.83 if cfg.backbone_name == "regnet_y_800mf" else 1.61
    scale = (cfg.input_h * cfg.input_w) / (224 * 224)
    print(f"backbone={cfg.backbone_name} pretrained={cfg.pretrained_backbone}")
    print(f"parameters_total={total:,} trainable={trainable:,}")
    print(f"full_token_cache_fp16_mib={cache_mib:.1f} projected_kv_cache_fp16_mib={kv_mib:.1f}")
    print(f"rough_backbone_gflops_at_input={flops_ref * scale:.1f}")
    print(
        "shapes "
        f"full_cache=[B,{cfg.full_context_frames},{cfg.tokens_per_frame},{cfg.hidden_dim}] "
        f"summary=[B,{cfg.summary_context_steps},{cfg.hidden_dim}] "
        f"actions=[B,{cfg.action_history_steps},{cfg.action_dim}]"
    )


if __name__ == "__main__":
    main()
