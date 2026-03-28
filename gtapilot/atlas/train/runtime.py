from __future__ import annotations

import contextlib
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from ..data import (
    AtlasBlackboxClipDataset,
    AtlasPrivilegedClipDataset,
    ClipGroupedBatchSampler,
    collate_temporal_clips,
)
from ..heads.pretrain_heads import Stage1APretrainHeads, Stage1CPretrainHeads
from ..losses import (
    compute_stage1a_losses,
    compute_stage1b_losses,
    compute_stage1c_losses,
)
from .checkpointing import load_checkpoint, save_checkpoint
from .common import (
    batch_to_device,
    batch_to_device_except,
    build_model_from_variant,
    ensure_dir,
    resolve_device,
    set_seed,
)
from .ema import EMAModel
from .logging_utils import ScalarLogger
from .metrics import StepTimer, grad_norm, tensor_memory_mb
from .target_builders import build_stage1a_targets, build_stage1c_targets
from .train_config import DataConfig, TrainerConfig


class StageTrainModule(nn.Module):
    def __init__(
        self,
        atlas: nn.Module,
        *,
        stage1a_heads: Stage1APretrainHeads | None = None,
        stage1c_heads: Stage1CPretrainHeads | None = None,
    ):
        super().__init__()
        self.atlas = atlas
        self.stage1a_heads = stage1a_heads
        self.stage1c_heads = stage1c_heads



def _has_explicit_source(data_cfg: DataConfig) -> bool:
    return bool(data_cfg.metadata_paths or data_cfg.split_file)


def _resolve_metadata_source_paths(data_cfg: DataConfig) -> list[Path]:
    root = Path(data_cfg.recordings_root)

    def _resolve_entry(
        raw_path: str | Path,
        *,
        split_parent: Path | None = None,
    ) -> Path:
        path = Path(raw_path)
        if path.is_absolute():
            return path.resolve()
        candidates: list[Path] = []
        if split_parent is not None:
            candidates.append(split_parent / path)
        candidates.append(root / path)
        candidates.append(path)
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        return candidates[0].resolve()

    if data_cfg.metadata_paths:
        return [_resolve_entry(path) for path in data_cfg.metadata_paths]
    if data_cfg.split_file:
        split_path = Path(data_cfg.split_file).resolve()
        lines = split_path.read_text(encoding="utf-8").splitlines()
        return [
            _resolve_entry(line.strip(), split_parent=split_path.parent)
            for line in lines
            if line.strip()
        ]
    return sorted(root.glob("capture_*_metadata.json"))


def _missing_privileged_message(
    *,
    stage: str,
    data_cfg: DataConfig,
) -> str:
    root = Path(data_cfg.recordings_root)
    metadata_paths = _resolve_metadata_source_paths(data_cfg)
    expected_dirs = [
        metadata_path.with_name(
            f"{metadata_path.stem.replace('_metadata', '')}_privileged"
        )
        for metadata_path in metadata_paths
    ]
    present = [
        privileged_dir
        for privileged_dir in expected_dirs
        if (privileged_dir / "manifest.json").exists()
    ]
    if not expected_dirs:
        coverage = f"found 0 source metadata files under {root}"
    else:
        coverage = (
            f"found {len(present)} of {len(expected_dirs)} expected privileged sibling packages"
        )
    return (
        f"{stage} requires privileged clip packages, but {coverage}. "
        "Build sibling capture_<timestamp>_privileged/ manifests first, for example with "
        "`python -m gtapilot.atlas.data.build_stage1b_privileged_dataset ...`, "
        "or point the split at clips that already have privileged packages."
    )


def _build_dataset(
    stage: str,
    data_cfg: DataConfig,
    atlas_cfg,
):
    dataset_cls = AtlasBlackboxClipDataset
    if data_cfg.use_privileged_dataset or stage == "stage1b":
        dataset_cls = AtlasPrivilegedClipDataset
    return dataset_cls(
        data_cfg.recordings_root,
        atlas_cfg,
        metadata_paths=data_cfg.metadata_paths or None,
        split_file=data_cfg.split_file,
        index_cache_dir=data_cfg.index_cache_dir,
        recent_steps=data_cfg.recent_steps,
        older_steps=data_cfg.older_steps,
        mid_steps=data_cfg.mid_steps,
        action_steps=data_cfg.action_steps,
        anchor_stride_steps=data_cfg.anchor_stride_steps,
        max_samples_per_clip_per_epoch=data_cfg.max_samples_per_clip_per_epoch,
    )


def _build_loader(
    dataset,
    data_cfg: DataConfig,
    *,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    stage1_gpu_loader = (
        isinstance(dataset, (AtlasBlackboxClipDataset, AtlasPrivilegedClipDataset))
        and float(dataset.model_hz) in {24.0, 36.0}
    )
    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "num_workers": data_cfg.num_workers,
        # Stage 1 batches now keep the large RGB banks on CPU and stream frames to CUDA
        # one step at a time. Pinning the full three-tier clip tensors eagerly can exhaust
        # host/GPU staging memory before training even begins.
        "pin_memory": data_cfg.pin_memory and not stage1_gpu_loader,
        "persistent_workers": data_cfg.persistent_workers and data_cfg.num_workers > 0,
        "collate_fn": collate_temporal_clips,
    }
    if shuffle and data_cfg.clip_batch_grouping:
        loader_kwargs["batch_sampler"] = ClipGroupedBatchSampler(
            clip_ids=[sample.clip_id for sample in dataset.samples],
            batch_size=data_cfg.batch_size,
            drop_last=True,
            shuffle=True,
            seed=seed,
        )
    else:
        loader_kwargs["batch_size"] = data_cfg.batch_size
        loader_kwargs["shuffle"] = shuffle
        loader_kwargs["drop_last"] = shuffle
    if data_cfg.num_workers > 0 and data_cfg.prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = data_cfg.prefetch_factor
    return DataLoader(**loader_kwargs)


def _split_batch(value: Any, start: int, end: int) -> Any:
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value
        return value[start:end]
    if isinstance(value, list):
        return value[start:end]
    if isinstance(value, tuple):
        return tuple(value[start:end])
    if isinstance(value, dict):
        return {key: _split_batch(item, start, end) for key, item in value.items()}
    return value


def _iter_microbatches(batch: dict[str, Any], microbatch_size: int) -> Iterable[dict[str, Any]]:
    batch_size = int(batch["rgb_recent"].shape[0])
    if microbatch_size <= 0 or microbatch_size >= batch_size:
        yield batch
        return
    for start in range(0, batch_size, microbatch_size):
        end = min(batch_size, start + microbatch_size)
        yield _split_batch(batch, start, end)

def _autocast_context(device: torch.device, enabled: bool):
    if not enabled or device.type != "cuda":
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.float16)


def _stage_batch_to_device(
    stage: str,
    batch: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    if device.type != "cuda" or stage not in {"stage1a", "stage1b", "stage1c"}:
        return batch_to_device(batch, device)
    return batch_to_device_except(
        batch,
        device,
        skip_top_level_keys={"rgb_recent", "rgb_older", "rgb_mid"},
    )


def _build_horizons(values: list[float], device: torch.device) -> torch.Tensor:
    return torch.tensor(values, device=device, dtype=torch.float32)


def _forward_stage_model(
    atlas: nn.Module,
    stage: str,
    *,
    privileged: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    teacher_method = getattr(atlas, f"forward_{stage}_teacher", None)
    if callable(teacher_method) and getattr(atlas.cfg, "enable_privileged_teacher_adapters", False):
        return teacher_method(privileged=privileged, **kwargs)
    return getattr(atlas, f"forward_{stage}")(**kwargs)


def _summary_insert_sequence(
    summary_seq: torch.Tensor,
    dt_recent: torch.Tensor,
    stride: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if summary_seq.ndim != 4:
        raise ValueError("summary_seq must have shape [B, T, N, D].")
    if dt_recent.ndim != 3 or dt_recent.shape[:2] != summary_seq.shape[:2]:
        raise ValueError("dt_recent must align with summary_seq on [B, T].")

    stride = max(1, int(stride))
    insert_indices = list(range(stride - 1, summary_seq.shape[1], stride))
    if not insert_indices:
        return summary_seq[:, :0], dt_recent[:, :0]

    index_tensor = torch.tensor(insert_indices, device=summary_seq.device, dtype=torch.long)
    insert_seq = summary_seq.index_select(dim=1, index=index_tensor)

    dt_values = dt_recent.squeeze(-1)
    chunks: list[torch.Tensor] = []
    prev_insert_index = -1
    for insert_index in insert_indices:
        chunks.append(dt_values[:, prev_insert_index + 1 : insert_index + 1].sum(dim=1))
        prev_insert_index = insert_index
    dt_summary = torch.stack(chunks, dim=1).unsqueeze(-1)
    return insert_seq, dt_summary


def _build_train_module(stage: str, model_variant: str, trainer_cfg: TrainerConfig) -> StageTrainModule:
    atlas = build_model_from_variant(model_variant)
    _prune_atlas_for_stage(atlas, stage)
    atlas_cfg = atlas.cfg
    stage1a_heads = None
    stage1c_heads = None
    if stage == "stage1a" or (stage == "stage1b" and trainer_cfg.stage1b.retain_stage1a_losses):
        stage1a_heads = Stage1APretrainHeads(atlas_cfg)
    if stage == "stage1c":
        stage1c_heads = Stage1CPretrainHeads(atlas_cfg)
    return StageTrainModule(
        atlas,
        stage1a_heads=stage1a_heads,
        stage1c_heads=stage1c_heads,
    )


def _prune_atlas_for_stage(atlas: nn.Module, stage: str) -> None:
    teacher_only_modules = (
        "lidar_adapter",
        "pose_adapter",
        "actor_adapter",
        "map_adapter",
        "hidden_actor_adapter",
        "visibility_adapter",
        "flow_adapter",
        "risk_adapter",
    )
    prune_map = {
        "stage1a": (
            "action_encoder",
            "ego_filter",
            "geometry",
            "obs_pool",
            "route_adapter",
            "reasoner_bridge",
            "world",
            "risk_decoder",
            "planner",
            "aux",
            "scheduler",
            *teacher_only_modules,
        ),
        "stage1b": (
            "obs_pool",
            "route_adapter",
            "reasoner_bridge",
            "world",
            "risk_decoder",
            "planner",
            "aux",
            "scheduler",
            *teacher_only_modules,
        ),
        "stage1c": (
            "risk_decoder",
            "planner",
            "aux",
            "scheduler",
            *teacher_only_modules,
        ),
    }
    for name in prune_map.get(stage, ()):
        if hasattr(atlas, name):
            setattr(atlas, name, None)


def _stage_init_checkpoint(stage: str, trainer_cfg: TrainerConfig) -> str | None:
    if stage == "stage1b":
        return trainer_cfg.stage1b.init_checkpoint
    if stage == "stage1c":
        return trainer_cfg.stage1c.init_checkpoint
    return None


def _load_partial_module_state(module: nn.Module, path: str | Path) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = payload.get("model", payload)
    current = module.state_dict()
    filtered = {
        key: value
        for key, value in state_dict.items()
        if key in current and current[key].shape == value.shape
    }
    module.load_state_dict(filtered, strict=False)
    return len(filtered)


def _build_optimizer(module: StageTrainModule, trainer_cfg: TrainerConfig) -> torch.optim.Optimizer:
    opt_cfg = trainer_cfg.optimizer
    vision_ids = {id(param) for param in module.atlas.vision.parameters() if param.requires_grad}
    explicit_no_decay_suffixes = (
        "current_pos",
        "recent_pos",
        "pos",
        "queries",
        "base_queries",
        "proposal_queries",
        "query_tokens",
        "scale_embeddings",
        "relative_position_bias_table",
    )
    groups = {
        "vision_decay": [],
        "vision_no_decay": [],
        "base_decay": [],
        "base_no_decay": [],
    }
    seen_param_ids: set[int] = set()
    for name, param in module.named_parameters():
        if not param.requires_grad or id(param) in seen_param_ids:
            continue
        seen_param_ids.add(id(param))
        lower_name = name.lower()
        no_decay = (
            param.ndim == 1
            or name.endswith(".bias")
            or "norm" in lower_name
            or "layernorm" in lower_name
            or lower_name.endswith(".ln")
            or ".ln" in lower_name
            or "bn" in lower_name
            or any(
                lower_name == suffix or lower_name.endswith(f".{suffix}")
                for suffix in explicit_no_decay_suffixes
            )
        )
        group_name = "vision_" if id(param) in vision_ids else "base_"
        group_name += "no_decay" if no_decay else "decay"
        groups[group_name].append(param)
    if opt_cfg.name.lower() != "adamw":
        raise ValueError(f"Unsupported optimizer: {opt_cfg.name}")
    param_groups = []
    if groups["base_decay"]:
        param_groups.append(
            {"params": groups["base_decay"], "lr": opt_cfg.lr, "weight_decay": opt_cfg.weight_decay}
        )
    if groups["base_no_decay"]:
        param_groups.append({"params": groups["base_no_decay"], "lr": opt_cfg.lr, "weight_decay": 0.0})
    if groups["vision_decay"]:
        param_groups.append(
            {
                "params": groups["vision_decay"],
                "lr": opt_cfg.lr * opt_cfg.vision_lr_scale,
                "weight_decay": opt_cfg.weight_decay,
            }
        )
    if groups["vision_no_decay"]:
        param_groups.append(
            {
                "params": groups["vision_no_decay"],
                "lr": opt_cfg.lr * opt_cfg.vision_lr_scale,
                "weight_decay": 0.0,
            }
        )
    return AdamW(
        param_groups,
        lr=opt_cfg.lr,
        weight_decay=0.0,
        betas=opt_cfg.betas,
        eps=opt_cfg.eps,
    )


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    trainer_cfg: TrainerConfig,
) -> LambdaLR:
    sched_cfg = trainer_cfg.lr_schedule
    total_steps = max(int(trainer_cfg.max_steps), int(sched_cfg.total_steps))
    warmup_steps = max(0, int(sched_cfg.warmup_steps))
    min_lr_ratio = float(sched_cfg.min_lr_ratio)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        if total_steps <= warmup_steps:
            return 1.0
        progress = min(
            1.0,
            max(0.0, (step - warmup_steps) / float(total_steps - warmup_steps)),
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def _current_lr(optimizer: torch.optim.Optimizer) -> float:
    return max(group["lr"] for group in optimizer.param_groups)


def _maybe_prune_checkpoints(output_dir: Path, keep_last: int) -> None:
    if keep_last <= 0:
        return
    checkpoints = sorted(output_dir.glob("step_*.pt"))
    if len(checkpoints) <= keep_last:
        return
    for path in checkpoints[:-keep_last]:
        path.unlink(missing_ok=True)


def _canonical_stage1b_outputs(outputs: dict[str, Any]) -> dict[str, torch.Tensor]:
    seq = outputs["seq"]
    return {
        "pose_delta_seq": seq["pose_delta"],
        "kinematics_seq": seq["kinematics"],
        "logvar_pose_seq": seq["logvar_pose"],
        "depth_mean_seq": seq["depth_mean"],
        "track_offsets_seq": seq["track_offsets"],
    }


def _canonical_stage1c_outputs(outputs: dict[str, Any]) -> dict[str, torch.Tensor]:
    seq = outputs["seq"]
    return {
        "pose_delta_seq": seq["pose_delta"],
        "kinematics_seq": seq["kinematics"],
        "logvar_pose_seq": seq["logvar_pose"],
        "static_grid_seq": seq["static_grid"],
        "dynamic_slots_seq": seq["dynamic_slots"],
        "speculative_slots_seq": seq["speculative_slots"],
        "dynamic_slot_alive_seq": seq["dynamic_slot_alive"],
        "speculative_slot_alive_seq": seq["speculative_slot_alive"],
        "lane_slots_seq": seq["lane_slots"],
        "map_elem_slots_seq": seq["map_elem_slots"],
        "route_tokens_seq": seq["route_tokens"],
        "reasoner_tokens_seq": seq["reasoner_tokens"],
        "ego_tokens_seq": seq["ego_tokens"],
    }


def _stage1a_forward_loss(
    module: StageTrainModule,
    batch: dict[str, Any],
    trainer_cfg: TrainerConfig,
    data_cfg: DataConfig,
    ema_modules: dict[str, EMAModel],
    device: torch.device,
    *,
    amp_enabled: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if module.stage1a_heads is None or "atlas" not in ema_modules:
        raise RuntimeError("Stage 1A requires stage1a_heads and atlas EMA.")
    horizons_cam = _build_horizons(trainer_cfg.stage1a.cam_horizons_s, device)
    horizons_sum = _build_horizons(trainer_cfg.stage1a.summary_horizons_s, device)
    masking = {
        "rgb_erasing_prob": trainer_cfg.stage1a.rgb_erasing_prob,
        "rgb_erasing_frac": trainer_cfg.stage1a.rgb_erasing_frac,
        "token_dropout_prob": trainer_cfg.stage1a.token_dropout_prob,
        "recent_frame_drop_prob": trainer_cfg.stage1a.recent_frame_drop_prob,
        "summary_frame_drop_prob": trainer_cfg.stage1a.summary_frame_drop_prob,
    }
    with _autocast_context(device, amp_enabled):
        student = _forward_stage_model(
            module.atlas,
            "stage1a",
            rgb_recent=batch["rgb_recent"],
            dt_recent=batch["dt_recent"],
            rgb_older=batch["rgb_older"],
            dt_older=batch["dt_older"],
            rgb_mid=batch["rgb_mid"],
            dt_mid=batch["dt_mid"],
            privileged=batch.get("teacher_privileged"),
            student_masking=masking,
            seed_older_with_grad=data_cfg.seed_older_with_grad,
            seed_mid_with_grad=data_cfg.seed_mid_with_grad,
        )
        summary_src, dt_summary = _summary_insert_sequence(
            student["seq"]["frame_summary"],
            batch["dt_recent"],
            module.atlas.cfg.temporal.summary_stride_steps,
        )
        preds = module.stage1a_heads(
            cam_src=student["seq"]["cam_now"],
            summary_src=summary_src,
            cam_horizons_s=horizons_cam,
            summary_horizons_s=horizons_sum,
        )
    del student
    with torch.no_grad():
        with _autocast_context(device, amp_enabled):
            teacher = _forward_stage_model(
                ema_modules["atlas"].module,
                "stage1a",
                rgb_recent=batch["rgb_recent"],
                dt_recent=batch["dt_recent"],
                rgb_older=batch["rgb_older"],
                dt_older=batch["dt_older"],
                rgb_mid=batch["rgb_mid"],
                dt_mid=batch["dt_mid"],
                privileged=batch.get("teacher_privileged"),
                seed_older_with_grad=data_cfg.seed_older_with_grad,
                seed_mid_with_grad=data_cfg.seed_mid_with_grad,
            )
        teacher_summary_src, _ = _summary_insert_sequence(
            teacher["seq"]["frame_summary"],
            batch["dt_recent"],
            module.atlas.cfg.temporal.summary_stride_steps,
        )
        targets = build_stage1a_targets(
            teacher_cam_seq=teacher["seq"]["cam_now"].detach(),
            teacher_summary_seq=teacher_summary_src.detach(),
            dt_recent=batch["dt_recent"],
            dt_summary=dt_summary,
            cam_horizons_s=horizons_cam,
            summary_horizons_s=horizons_sum,
        )
    del teacher, teacher_summary_src
    losses = compute_stage1a_losses(preds, targets)
    return losses["total"], losses


def _stage1b_forward_loss(
    module: StageTrainModule,
    batch: dict[str, Any],
    trainer_cfg: TrainerConfig,
    data_cfg: DataConfig,
    ema_modules: dict[str, EMAModel],
    device: torch.device,
    *,
    amp_enabled: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    with _autocast_context(device, amp_enabled):
        outputs = _forward_stage_model(
            module.atlas,
            "stage1b",
            rgb_recent=batch["rgb_recent"],
            dt_recent=batch["dt_recent"],
            rgb_older=batch["rgb_older"],
            dt_older=batch["dt_older"],
            rgb_mid=batch["rgb_mid"],
            dt_mid=batch["dt_mid"],
            actions_hist=batch["actions_hist"],
            dt_hist=batch["dt_hist"],
            privileged=batch.get("teacher_privileged"),
            seed_older_with_grad=data_cfg.seed_older_with_grad,
            seed_mid_with_grad=data_cfg.seed_mid_with_grad,
        )
        canonical = _canonical_stage1b_outputs(outputs)
        losses = compute_stage1b_losses(canonical, batch)
    if trainer_cfg.stage1b.retain_stage1a_losses:
        if module.stage1a_heads is None or "atlas" not in ema_modules:
            raise RuntimeError(
                "retain_stage1a_losses requires Stage1APretrainHeads and atlas EMA."
            )
        horizons_cam = _build_horizons(trainer_cfg.stage1a.cam_horizons_s, device)
        horizons_sum = _build_horizons(trainer_cfg.stage1a.summary_horizons_s, device)
        retain_summary_src, dt_summary = _summary_insert_sequence(
            outputs["seq"]["frame_summary"],
            batch["dt_recent"],
            module.atlas.cfg.temporal.summary_stride_steps,
        )
        with _autocast_context(device, amp_enabled):
            retain_preds = module.stage1a_heads(
                cam_src=outputs["seq"]["cam_now"],
                summary_src=retain_summary_src,
                cam_horizons_s=horizons_cam,
                summary_horizons_s=horizons_sum,
            )
        del outputs, canonical
        with torch.no_grad():
            with _autocast_context(device, amp_enabled):
                teacher = _forward_stage_model(
                    ema_modules["atlas"].module,
                    "stage1a",
                    rgb_recent=batch["rgb_recent"],
                    dt_recent=batch["dt_recent"],
                    rgb_older=batch["rgb_older"],
                    dt_older=batch["dt_older"],
                    rgb_mid=batch["rgb_mid"],
                    dt_mid=batch["dt_mid"],
                    privileged=batch.get("teacher_privileged"),
                    seed_older_with_grad=data_cfg.seed_older_with_grad,
                    seed_mid_with_grad=data_cfg.seed_mid_with_grad,
                )
            teacher_summary_src, _ = _summary_insert_sequence(
                teacher["seq"]["frame_summary"],
                batch["dt_recent"],
                module.atlas.cfg.temporal.summary_stride_steps,
            )
            retain_targets = build_stage1a_targets(
                teacher_cam_seq=teacher["seq"]["cam_now"].detach(),
                teacher_summary_seq=teacher_summary_src.detach(),
                dt_recent=batch["dt_recent"],
                dt_summary=dt_summary,
                cam_horizons_s=horizons_cam,
                summary_horizons_s=horizons_sum,
            )
        del teacher, teacher_summary_src
        retain_losses = compute_stage1a_losses(retain_preds, retain_targets)
        losses["retain_stage1a"] = retain_losses["total"]
        losses["total"] = losses["total"] + (
            trainer_cfg.stage1b.retain_stage1a_weight * retain_losses["total"]
        )
    else:
        del outputs, canonical
    return losses["total"], losses


def _stage1c_burn_in_batch(
    batch: dict[str, Any],
    burn_in_recent_steps: int | None,
) -> dict[str, Any] | None:
    total_older_steps = int(batch["rgb_older"].shape[1])
    current_recent_steps = int(batch["rgb_recent"].shape[1])
    if total_older_steps <= 0:
        return None
    available_action_steps = int(batch["actions_hist"].shape[1]) - current_recent_steps
    if available_action_steps <= 0:
        return None
    burn_in_steps = total_older_steps if burn_in_recent_steps is None else int(burn_in_recent_steps)
    burn_in_steps = max(0, min(burn_in_steps, total_older_steps, available_action_steps))
    if burn_in_steps <= 0:
        return None
    burn_in_actions = batch["actions_hist"][:, :-current_recent_steps]
    burn_in_dt_hist = batch["dt_hist"][:, :-current_recent_steps]
    if burn_in_actions.shape[1] < burn_in_steps or burn_in_dt_hist.shape[1] < burn_in_steps:
        return None
    return {
        "rgb_recent": batch["rgb_older"][:, -burn_in_steps:],
        "dt_recent": batch["dt_older"][:, -burn_in_steps:],
        "rgb_older": batch["rgb_older"][:, :-burn_in_steps],
        "dt_older": batch["dt_older"][:, :-burn_in_steps],
        "rgb_mid": batch["rgb_mid"],
        "dt_mid": batch["dt_mid"],
        "actions_hist": burn_in_actions,
        "dt_hist": burn_in_dt_hist,
        "route_polyline": batch.get("route_polyline"),
        "nav_cmd": batch.get("nav_cmd"),
        # Reasoner tokens are optional and currently not part of the Stage 1 data path.
        "reasoner_tok": None,
    }


def _stage1c_burn_in_state(
    atlas: nn.Module,
    burn_in_batch: dict[str, Any] | None,
    data_cfg: DataConfig,
    device: torch.device,
    *,
    amp_enabled: bool,
):
    if burn_in_batch is None:
        return None
    with torch.no_grad():
        with _autocast_context(device, amp_enabled):
            burn_in_outputs = _forward_stage_model(
                atlas,
                "stage1c",
                rgb_recent=burn_in_batch["rgb_recent"],
                dt_recent=burn_in_batch["dt_recent"],
                rgb_older=burn_in_batch["rgb_older"],
                dt_older=burn_in_batch["dt_older"],
                rgb_mid=burn_in_batch["rgb_mid"],
                dt_mid=burn_in_batch["dt_mid"],
                actions_hist=burn_in_batch["actions_hist"],
                dt_hist=burn_in_batch["dt_hist"],
                route_polyline=burn_in_batch.get("route_polyline"),
                nav_cmd=burn_in_batch.get("nav_cmd"),
                reasoner_tok=burn_in_batch.get("reasoner_tok"),
                privileged=burn_in_batch.get("teacher_privileged"),
                seed_older_with_grad=data_cfg.seed_older_with_grad,
                seed_mid_with_grad=data_cfg.seed_mid_with_grad,
            )
    return burn_in_outputs["final_state"]


def _stage1c_forward_loss(
    module: StageTrainModule,
    batch: dict[str, Any],
    trainer_cfg: TrainerConfig,
    data_cfg: DataConfig,
    ema_modules: dict[str, EMAModel],
    device: torch.device,
    *,
    amp_enabled: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if module.stage1c_heads is None:
        raise RuntimeError("Stage 1C requires Stage1CPretrainHeads.")
    if "atlas" not in ema_modules or "stage1c_heads" not in ema_modules:
        raise RuntimeError("Stage 1C requires EMA atlas and Stage1CPretrainHeads.")
    horizons_world = _build_horizons(trainer_cfg.stage1c.world_horizons_s, device)
    burn_in_batch = _stage1c_burn_in_batch(
        batch,
        trainer_cfg.stage1c.burn_in_recent_steps,
    )
    init_state = _stage1c_burn_in_state(
        module.atlas,
        burn_in_batch,
        data_cfg,
        device,
        amp_enabled=amp_enabled,
    )
    with _autocast_context(device, amp_enabled):
        outputs = _forward_stage_model(
            module.atlas,
            "stage1c",
            rgb_recent=batch["rgb_recent"],
            dt_recent=batch["dt_recent"],
            rgb_older=batch["rgb_older"],
            dt_older=batch["dt_older"],
            rgb_mid=batch["rgb_mid"],
            dt_mid=batch["dt_mid"],
            actions_hist=batch["actions_hist"],
            dt_hist=batch["dt_hist"],
            route_polyline=batch.get("route_polyline"),
            nav_cmd=batch.get("nav_cmd"),
            reasoner_tok=batch.get("reasoner_tok"),
            privileged=batch.get("teacher_privileged"),
            init_state=init_state,
            student_corruption={
                "cam_drop_prob": trainer_cfg.stage1c.cam_drop_prob,
                "frustum_drop_prob": trainer_cfg.stage1c.frustum_drop_prob,
                "context_family_drop_prob": trainer_cfg.stage1c.context_family_drop_prob,
            },
            seed_older_with_grad=data_cfg.seed_older_with_grad,
            seed_mid_with_grad=data_cfg.seed_mid_with_grad,
        )
        canonical = _canonical_stage1c_outputs(outputs)
        student_readout = module.stage1c_heads.readout_seq(
            static_grid_seq=canonical["static_grid_seq"],
            dynamic_slots_seq=canonical["dynamic_slots_seq"],
            speculative_slots_seq=canonical["speculative_slots_seq"],
            lane_slots_seq=canonical["lane_slots_seq"],
            map_elem_slots_seq=canonical["map_elem_slots_seq"],
            route_tokens_seq=canonical["route_tokens_seq"],
            reasoner_tokens_seq=canonical["reasoner_tokens_seq"],
            ego_tokens_seq=canonical["ego_tokens_seq"],
            dynamic_alive_seq=canonical["dynamic_slot_alive_seq"],
            speculative_alive_seq=canonical["speculative_slot_alive_seq"],
        )
        world_pred = module.stage1c_heads.project(
            student_readout,
            horizons_world,
        )
    del outputs
    teacher_init_state = _stage1c_burn_in_state(
        ema_modules["atlas"].module,
        burn_in_batch,
        data_cfg,
        device,
        amp_enabled=amp_enabled,
    )
    with torch.no_grad():
        with _autocast_context(device, amp_enabled):
            teacher = _forward_stage_model(
                ema_modules["atlas"].module,
                "stage1c",
                rgb_recent=batch["rgb_recent"],
                dt_recent=batch["dt_recent"],
                rgb_older=batch["rgb_older"],
                dt_older=batch["dt_older"],
                rgb_mid=batch["rgb_mid"],
                dt_mid=batch["dt_mid"],
                actions_hist=batch["actions_hist"],
                dt_hist=batch["dt_hist"],
                route_polyline=batch.get("route_polyline"),
                nav_cmd=batch.get("nav_cmd"),
                reasoner_tok=batch.get("reasoner_tok"),
                privileged=batch.get("teacher_privileged"),
                init_state=teacher_init_state,
                seed_older_with_grad=data_cfg.seed_older_with_grad,
                seed_mid_with_grad=data_cfg.seed_mid_with_grad,
            )
            teacher_readout = ema_modules["stage1c_heads"].module.readout_seq(
                static_grid_seq=teacher["seq"]["static_grid"],
                dynamic_slots_seq=teacher["seq"]["dynamic_slots"],
                speculative_slots_seq=teacher["seq"]["speculative_slots"],
                lane_slots_seq=teacher["seq"]["lane_slots"],
                map_elem_slots_seq=teacher["seq"]["map_elem_slots"],
                route_tokens_seq=teacher["seq"]["route_tokens"],
                reasoner_tokens_seq=teacher["seq"]["reasoner_tokens"],
                ego_tokens_seq=teacher["seq"]["ego_tokens"],
                dynamic_alive_seq=teacher["seq"]["dynamic_slot_alive"],
                speculative_alive_seq=teacher["seq"]["speculative_slot_alive"],
            )
        targets = build_stage1c_targets(
            teacher_world_seq=teacher_readout.detach(),
            teacher_static_grid_seq=teacher["seq"]["static_grid"].detach(),
            dt_recent=batch["dt_recent"],
            world_horizons_s=horizons_world,
        )
    del teacher, teacher_readout
    if "pose_delta_recent" in batch:
        targets["pose_delta_recent"] = batch["pose_delta_recent"]
        targets["kinematics_recent"] = batch["kinematics_recent"]
        targets["pose_valid_recent"] = batch.get("pose_valid_recent")
    loss_inputs = {
        "world_projector_pred": world_pred,
        "static_grid_seq": canonical["static_grid_seq"],
        "pose_delta_seq": canonical["pose_delta_seq"],
        "kinematics_seq": canonical["kinematics_seq"],
        "logvar_pose_seq": canonical["logvar_pose_seq"],
    }
    losses = compute_stage1c_losses(loss_inputs, targets)
    return losses["total"], losses


def _forward_loss_for_stage(
    stage: str,
    module: StageTrainModule,
    batch: dict[str, Any],
    trainer_cfg: TrainerConfig,
    data_cfg: DataConfig,
    ema_modules: dict[str, EMAModel],
    device: torch.device,
    *,
    amp_enabled: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if stage == "stage1a":
        return _stage1a_forward_loss(
            module,
            batch,
            trainer_cfg,
            data_cfg,
            ema_modules,
            device,
            amp_enabled=amp_enabled,
        )
    if stage == "stage1b":
        return _stage1b_forward_loss(
            module,
            batch,
            trainer_cfg,
            data_cfg,
            ema_modules,
            device,
            amp_enabled=amp_enabled,
        )
    if stage == "stage1c":
        return _stage1c_forward_loss(
            module,
            batch,
            trainer_cfg,
            data_cfg,
            ema_modules,
            device,
            amp_enabled=amp_enabled,
        )
    raise ValueError(f"Unsupported stage: {stage}")


def _train_one_step(
    stage: str,
    module: StageTrainModule,
    batch: dict[str, Any],
    trainer_cfg: TrainerConfig,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    ema_modules: dict[str, EMAModel],
    device: torch.device,
) -> tuple[dict[str, float], bool]:
    module.train()
    optimizer.zero_grad(set_to_none=True)
    microbatch_size = trainer_cfg.train.microbatch_size
    microbatches = list(_iter_microbatches(batch, microbatch_size))
    aggregated: dict[str, float] = {}
    for microbatch in microbatches:
        total_loss, losses = _forward_loss_for_stage(
            stage,
            module,
            microbatch,
            trainer_cfg,
            trainer_cfg.train,
            ema_modules,
            device,
            amp_enabled=trainer_cfg.amp,
        )
        microbatch_weight = float(microbatch["rgb_recent"].shape[0]) / float(
            max(1, batch["rgb_recent"].shape[0])
        )
        scaled_loss = total_loss * microbatch_weight
        scaler.scale(scaled_loss).backward()
        weight = float(microbatch["rgb_recent"].shape[0])
        for key, value in losses.items():
            aggregated[key] = aggregated.get(key, 0.0) + float(value.detach().item()) * weight

    scaler.unscale_(optimizer)
    if trainer_cfg.optimizer.grad_clip_norm > 0.0:
        torch.nn.utils.clip_grad_norm_(
            module.parameters(),
            trainer_cfg.optimizer.grad_clip_norm,
        )
    grad = grad_norm(module.parameters())
    prev_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    optimizer_step_applied = scaler.get_scale() >= prev_scale
    if optimizer_step_applied:
        if "atlas" in ema_modules:
            ema_modules["atlas"].update(module.atlas)
        if "stage1c_heads" in ema_modules and module.stage1c_heads is not None:
            ema_modules["stage1c_heads"].update(module.stage1c_heads)
    batch_total = float(batch["rgb_recent"].shape[0])
    for key in aggregated:
        aggregated[key] /= max(1.0, batch_total)
    aggregated["grad_norm"] = grad
    aggregated["batch_size"] = batch_total
    return aggregated, optimizer_step_applied


@torch.no_grad()
def _validate(
    stage: str,
    module: StageTrainModule,
    loader: DataLoader | None,
    trainer_cfg: TrainerConfig,
    ema_modules: dict[str, EMAModel],
    device: torch.device,
) -> dict[str, float]:
    if loader is None:
        return {}
    module.eval()
    totals: dict[str, float] = {}
    seen = 0.0
    for batch in loader:
        batch = _stage_batch_to_device(stage, batch, device)
        total_loss, losses = _forward_loss_for_stage(
            stage,
            module,
            batch,
            trainer_cfg,
            trainer_cfg.val,
            ema_modules,
            device,
            amp_enabled=trainer_cfg.amp,
        )
        batch_weight = float(batch["rgb_recent"].shape[0])
        seen += batch_weight
        totals["total"] = totals.get("total", 0.0) + float(total_loss.item()) * batch_weight
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value.item()) * batch_weight
    if seen <= 0.0:
        return {}
    return {key: value / seen for key, value in totals.items()}


def run_training_stage(
    stage: str,
    trainer_cfg: TrainerConfig,
) -> dict[str, Any]:
    set_seed(trainer_cfg.seed)
    device = resolve_device(trainer_cfg.device)
    output_dir = ensure_dir(Path(trainer_cfg.checkpoint.output_dir) / stage)

    module = _build_train_module(stage, trainer_cfg.model_variant, trainer_cfg)
    atlas_cfg = module.atlas.cfg
    if stage == "stage1c":
        recent_steps = int(atlas_cfg.temporal.recent_full_frames)
        if trainer_cfg.stage1c.tbptt_steps is None:
            trainer_cfg.stage1c.tbptt_steps = recent_steps
        if int(trainer_cfg.stage1c.tbptt_steps) != recent_steps:
            raise ValueError(
                "Current Stage 1C runtime uses the recent clip length as the TBPTT window. "
                f"Set tbptt_steps={recent_steps} for this model configuration."
            )
    trainer_cfg.save_json(output_dir / "trainer_config.json")
    module.to(device)

    train_dataset = _build_dataset(stage, trainer_cfg.train, atlas_cfg)
    if len(train_dataset) == 0:
        if stage == "stage1b" or trainer_cfg.train.use_privileged_dataset:
            raise RuntimeError(
                _missing_privileged_message(stage=stage, data_cfg=trainer_cfg.train)
            )
        raise RuntimeError("Training dataset is empty.")
    train_loader = _build_loader(
        train_dataset,
        trainer_cfg.train,
        shuffle=True,
        seed=trainer_cfg.seed,
    )

    val_loader = None
    validation_requested = trainer_cfg.validate_first or trainer_cfg.logging.val_every_steps > 0
    if validation_requested:
        if not _has_explicit_source(trainer_cfg.val):
            raise RuntimeError(
                "Validation was requested, but val.metadata_paths or val.split_file is not configured."
            )
        val_dataset = _build_dataset(stage, trainer_cfg.val, atlas_cfg)
        if len(val_dataset) == 0:
            if stage == "stage1b" or trainer_cfg.val.use_privileged_dataset:
                raise RuntimeError(
                    "Validation was requested, but the privileged validation dataset is empty. "
                    + _missing_privileged_message(stage=stage, data_cfg=trainer_cfg.val)
                )
            raise RuntimeError("Validation was requested, but the validation dataset is empty.")
        val_loader = _build_loader(
            val_dataset,
            trainer_cfg.val,
            shuffle=False,
            seed=trainer_cfg.seed,
        )

    optimizer = _build_optimizer(module, trainer_cfg)
    scheduler = _build_scheduler(optimizer, trainer_cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=trainer_cfg.amp and device.type == "cuda")

    init_checkpoint = _stage_init_checkpoint(stage, trainer_cfg)
    if trainer_cfg.checkpoint.resume_from:
        ema_modules: dict[str, EMAModel] = {}
        if stage == "stage1a":
            ema_modules["atlas"] = EMAModel(module.atlas, trainer_cfg.stage1a.ema_decay)
        elif stage == "stage1c":
            ema_modules["atlas"] = EMAModel(module.atlas, trainer_cfg.stage1c.ema_decay)
            if module.stage1c_heads is None:
                raise RuntimeError("Stage 1C requires Stage1CPretrainHeads.")
            ema_modules["stage1c_heads"] = EMAModel(
                module.stage1c_heads,
                trainer_cfg.stage1c.ema_decay,
            )
        elif stage == "stage1b" and trainer_cfg.stage1b.retain_stage1a_losses:
            ema_modules["atlas"] = EMAModel(module.atlas, trainer_cfg.stage1a.ema_decay)
        resume_state = load_checkpoint(
            trainer_cfg.checkpoint.resume_from,
            model=module,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            ema_loaders={name: ema.load_state_dict for name, ema in ema_modules.items()},
            map_location=device,
        )
        start_step = int(resume_state["step"])
    else:
        if init_checkpoint:
            _load_partial_module_state(module, init_checkpoint)
        ema_modules = {}
        if stage == "stage1a":
            ema_modules["atlas"] = EMAModel(module.atlas, trainer_cfg.stage1a.ema_decay)
        elif stage == "stage1c":
            ema_modules["atlas"] = EMAModel(module.atlas, trainer_cfg.stage1c.ema_decay)
            if module.stage1c_heads is None:
                raise RuntimeError("Stage 1C requires Stage1CPretrainHeads.")
            ema_modules["stage1c_heads"] = EMAModel(
                module.stage1c_heads,
                trainer_cfg.stage1c.ema_decay,
            )
        elif stage == "stage1b" and trainer_cfg.stage1b.retain_stage1a_losses:
            ema_modules["atlas"] = EMAModel(module.atlas, trainer_cfg.stage1a.ema_decay)
        start_step = 0

    logger = ScalarLogger(
        output_dir,
        jsonl_file=trainer_cfg.logging.jsonl_file,
        tensorboard_dir=trainer_cfg.logging.tensorboard_dir,
    )
    timer = StepTimer()

    if trainer_cfg.validate_first and val_loader is not None:
        val_metrics = _validate(stage, module, val_loader, trainer_cfg, ema_modules, device)
        if val_metrics:
            logger.log_scalars(start_step, val_metrics, prefix="val/")

    step = start_step
    while step < trainer_cfg.max_steps:
        for raw_batch in train_loader:
            if step >= trainer_cfg.max_steps:
                break
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            batch = _stage_batch_to_device(stage, raw_batch, device)
            train_metrics, optimizer_step_applied = _train_one_step(
                stage,
                module,
                batch,
                trainer_cfg,
                optimizer,
                scaler,
                ema_modules,
                device,
            )
            if optimizer_step_applied:
                scheduler.step()
            step += 1
            elapsed = timer.tick()
            train_metrics["lr"] = _current_lr(optimizer)
            train_metrics["throughput_samples_per_s"] = train_metrics.pop("batch_size") / max(
                elapsed,
                1e-6,
            )
            train_metrics["gpu_mem_mb"] = tensor_memory_mb(device)
            if step % max(1, trainer_cfg.logging.log_every_steps) == 0:
                logger.log_scalars(step, train_metrics, prefix="train/")
            if (
                val_loader is not None
                and trainer_cfg.logging.val_every_steps > 0
                and step % trainer_cfg.logging.val_every_steps == 0
            ):
                val_metrics = _validate(
                    stage,
                    module,
                    val_loader,
                    trainer_cfg,
                    ema_modules,
                    device,
                )
                if val_metrics:
                    logger.log_scalars(step, val_metrics, prefix="val/")
            if step % max(1, trainer_cfg.checkpoint.save_every_steps) == 0:
                checkpoint_path = output_dir / f"step_{step:06d}.pt"
                save_checkpoint(
                    checkpoint_path,
                    model=module,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    ema_states={name: ema.state_dict() for name, ema in ema_modules.items()},
                    step=step,
                    extra_state={"stage": stage, "trainer_cfg": trainer_cfg.to_dict()},
                )
                save_checkpoint(
                    output_dir / "latest.pt",
                    model=module,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    ema_states={name: ema.state_dict() for name, ema in ema_modules.items()},
                    step=step,
                    extra_state={"stage": stage, "trainer_cfg": trainer_cfg.to_dict()},
                )
                _maybe_prune_checkpoints(output_dir, trainer_cfg.checkpoint.keep_last)

    save_checkpoint(
        output_dir / "latest.pt",
        model=module,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        ema_states={name: ema.state_dict() for name, ema in ema_modules.items()},
        step=step,
        extra_state={"stage": stage, "trainer_cfg": trainer_cfg.to_dict()},
    )
    logger.close()
    final_state = {
        "step": step,
        "checkpoint_dir": str(output_dir),
        "stage": stage,
        "model_variant": trainer_cfg.model_variant,
    }
    return final_state
