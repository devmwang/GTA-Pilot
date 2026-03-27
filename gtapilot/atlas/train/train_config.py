from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DataConfig:
    recordings_root: str = "blackbox-recordings"
    metadata_paths: list[str] = field(default_factory=list)
    split_file: str | None = None
    index_cache_dir: str | None = None
    use_privileged_dataset: bool = False
    batch_size: int = 1
    microbatch_size: int = 1
    num_workers: int = 0
    prefetch_factor: int | None = None
    pin_memory: bool = True
    persistent_workers: bool = False
    recent_steps: int | None = None
    older_steps: int | None = None
    mid_steps: int | None = None
    action_steps: int | None = None
    anchor_stride_steps: int = 1
    max_samples_per_clip_per_epoch: int | None = None
    clip_batch_grouping: bool = True
    seed_older_with_grad: bool = False
    seed_mid_with_grad: bool = False


@dataclass
class OptimizerConfig:
    name: str = "adamw"
    lr: float = 1e-4
    weight_decay: float = 1e-2
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    grad_clip_norm: float = 1.0
    vision_lr_scale: float = 0.5


@dataclass
class LRScheduleConfig:
    name: str = "cosine"
    warmup_steps: int = 100
    total_steps: int = 10_000
    min_lr_ratio: float = 0.1


@dataclass
class CheckpointConfig:
    output_dir: str = "artifacts/atlas-stage1"
    save_every_steps: int = 250
    keep_last: int = 3
    resume_from: str | None = None


@dataclass
class LoggingConfig:
    log_every_steps: int = 10
    val_every_steps: int = 0
    jsonl_file: str = "metrics.jsonl"
    tensorboard_dir: str | None = None


@dataclass
class Stage1AConfig:
    cam_horizons_s: list[float] = field(
        default_factory=lambda: [1.0 / 24.0, 2.0 / 24.0, 4.0 / 24.0]
    )
    summary_horizons_s: list[float] = field(default_factory=lambda: [1.0 / 6.0, 0.5, 1.0])
    ema_decay: float = 0.996
    rgb_erasing_prob: float = 0.2
    rgb_erasing_frac: float = 0.15
    token_dropout_prob: float = 0.1
    recent_frame_drop_prob: float = 0.05
    summary_frame_drop_prob: float = 0.05


@dataclass
class Stage1BConfig:
    init_checkpoint: str | None = None
    retain_stage1a_losses: bool = False
    retain_stage1a_weight: float = 0.1


@dataclass
class Stage1CConfig:
    world_horizons_s: list[float] = field(
        default_factory=lambda: [1.0 / 24.0, 4.0 / 24.0, 0.5]
    )
    ema_decay: float = 0.996
    cam_drop_prob: float = 0.1
    frustum_drop_prob: float = 0.1
    context_family_drop_prob: float = 0.05
    tbptt_steps: int | None = None
    init_checkpoint: str | None = None


@dataclass
class TrainerConfig:
    stage: str
    model_variant: str = "student"
    seed: int = 1337
    device: str = "cuda"
    amp: bool = True
    max_steps: int = 1_000
    validate_first: bool = False
    train: DataConfig = field(default_factory=DataConfig)
    val: DataConfig = field(default_factory=DataConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    lr_schedule: LRScheduleConfig = field(default_factory=LRScheduleConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    stage1a: Stage1AConfig = field(default_factory=Stage1AConfig)
    stage1b: Stage1BConfig = field(default_factory=Stage1BConfig)
    stage1c: Stage1CConfig = field(default_factory=Stage1CConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def _update_dataclass(instance: Any, values: dict[str, Any]) -> Any:
    for key, value in values.items():
        current = getattr(instance, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _update_dataclass(current, value)
        else:
            setattr(instance, key, value)
    return instance


def load_trainer_config(
    stage: str,
    path: str | Path | None = None,
) -> TrainerConfig:
    cfg = TrainerConfig(stage=stage)
    if path is None:
        return cfg
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return _update_dataclass(cfg, payload)
