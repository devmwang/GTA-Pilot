from __future__ import annotations

from dataclasses import dataclass, field


DEFAULT_VISUAL_OFFSETS_S: tuple[float, ...] = (
    0.00,
    -0.10,
    -0.20,
    -0.35,
    -0.50,
    -0.70,
    -0.90,
    -1.10,
    -1.35,
    -1.60,
    -1.90,
    -2.20,
    -2.50,
    -2.75,
    -3.00,
    -3.50,
)


@dataclass(slots=True)
class AtlasCruiseConfig:
    # Source/input.
    source_h: int = 1080
    source_w: int = 1920
    source_capture_hz: float = 60.0
    input_h: int = 540
    input_w: int = 960
    input_channels: int = 3

    # Runtime.
    model_hz: float = 20.0
    controller_hz: float = 60.0

    # Visual context for training.
    visual_context_s: float = 3.5
    num_visual_frames: int = 16
    visual_offsets_s: tuple[float, ...] = field(default_factory=lambda: DEFAULT_VISUAL_OFFSETS_S)

    # Action context.
    action_context_s: float = 5.0
    action_sample_hz: float = 20.0
    action_dim: int = 6

    # Backbone.
    backbone_name: str = "regnet_y_800mf"
    pretrained_backbone: bool = True
    hidden_dim: int = 128
    tokens_per_frame: int = 32
    source_tokens_per_frame: int = 64

    # Temporal model.
    action_tokens: int = 4
    temporal_layers: int = 2
    temporal_heads: int = 4
    temporal_mlp_ratio: float = 2.0
    temporal_dropout: float = 0.1

    # Trajectory output.
    traj_points: int = 30
    traj_horizon_s: float = 4.5
    target_latency_s: float = 0.10

    # Optional auxiliary control output.
    predict_control_aux: bool = True
    control_horizon_steps: int = 5
    control_dt_s: float = 0.10

    # UI/freshness.
    use_ui_mask: bool = True
    use_frame_freshness: bool = True
    neutralize_minimap: bool = True
    neutralize_phone: bool = True
    neutralize_top_left_popup: bool = True
    neutralize_center_reticle: bool = True

    # Training regularization.
    action_dropout_prob: float = 0.15
    rgb_aug_prob: float = 0.5

    # Dataset/target validation.
    target_package_name: str = "cruise_targets.npz"
    target_manifest_name: str = "cruise_targets_manifest.json"
    min_valid_traj_points: int = 20
    max_action_staleness_s: float = 0.25
    max_frame_staleness_s: float = 0.20
    max_ego_interp_gap_s: float = 0.25
    max_ego_teleport_m: float = 12.0
    max_ego_speed_mps: float = 95.0

    # Legalizer/controller.
    user_max_speed_mps: float = 38.0
    caution_speed_cap_mps: float = 13.0
    fallback_speed_cap_mps: float = 4.0
    comfort_lat_acc_mps2: float = 3.0
    hard_lat_acc_mps2: float = 5.0
    rain_lat_multiplier: float = 0.75
    wheelbase_eff_m: float = 2.6
    steer_max_rad: float = 0.55
    steer_rate_limit_per_s: float = 2.5
    throttle_rate_limit_per_s: float = 3.0
    brake_rate_limit_per_s: float = 4.0

    @property
    def action_history_steps(self) -> int:
        return int(round(self.action_context_s * self.action_sample_hz))

    @property
    def traj_dt_s(self) -> float:
        return float(self.traj_horizon_s) / float(self.traj_points)

    @property
    def expected_source_shape(self) -> tuple[int, int]:
        return (self.source_h, self.source_w)

    @property
    def expected_input_shape(self) -> tuple[int, int]:
        return (self.input_h, self.input_w)

    def resolved_visual_offsets_s(self) -> tuple[float, ...]:
        if len(self.visual_offsets_s) == self.num_visual_frames:
            return tuple(float(value) for value in self.visual_offsets_s)
        if self.num_visual_frames <= 1:
            return (0.0,)
        step = self.visual_context_s / float(self.num_visual_frames - 1)
        return tuple(-idx * step for idx in range(self.num_visual_frames))

    def validate(self) -> None:
        if self.source_h <= 0 or self.source_w <= 0:
            raise ValueError("source_h/source_w must be positive.")
        if self.input_h <= 0 or self.input_w <= 0:
            raise ValueError("input_h/input_w must be positive.")
        if self.input_channels != 3:
            raise ValueError("Atlas-Cruise expects RGB input_channels=3.")
        if self.model_hz <= 0.0 or self.controller_hz <= 0.0:
            raise ValueError("model_hz and controller_hz must be positive.")
        if self.num_visual_frames <= 0:
            raise ValueError("num_visual_frames must be positive.")
        if self.action_history_steps <= 0:
            raise ValueError("action history must contain at least one step.")
        if self.action_dim != 6:
            raise ValueError("action_dim must match [steer, throttle, brake, handbrake, reverse, pilot_active].")
        if self.hidden_dim <= 0 or self.hidden_dim % self.temporal_heads != 0:
            raise ValueError("hidden_dim must be positive and divisible by temporal_heads.")
        if self.tokens_per_frame <= 0:
            raise ValueError("tokens_per_frame must be positive.")
        if self.source_tokens_per_frame <= 0:
            raise ValueError("source_tokens_per_frame must be positive.")
        if self.action_tokens <= 0:
            raise ValueError("action_tokens must be positive.")
        if self.traj_points <= 1:
            raise ValueError("traj_points must be greater than 1.")
        if self.traj_horizon_s <= 0.0:
            raise ValueError("traj_horizon_s must be positive.")
        if self.control_horizon_steps <= 0:
            raise ValueError("control_horizon_steps must be positive.")
        if self.min_valid_traj_points <= 0 or self.min_valid_traj_points > self.traj_points:
            raise ValueError("min_valid_traj_points must be in [1, traj_points].")


def atlas_cruise_default_config(*, pretrained_backbone: bool | None = None) -> AtlasCruiseConfig:
    cfg = AtlasCruiseConfig()
    if pretrained_backbone is not None:
        cfg.pretrained_backbone = bool(pretrained_backbone)
    cfg.validate()
    return cfg
