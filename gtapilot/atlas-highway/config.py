from __future__ import annotations

from dataclasses import dataclass


TRAJECTORY_CANDIDATE_NAMES: tuple[str, ...] = (
    "KEEP_LANE_CRUISE",
    "KEEP_LANE_SLOW_OR_FOLLOW",
    "CHANGE_LEFT",
    "CHANGE_RIGHT",
    "FALLBACK_SLOW_STOP",
)

SCENE_TYPE_NAMES: tuple[str, ...] = (
    "highway_marked",
    "well_marked_non_highway",
    "ambiguous_road_like",
    "no_usable_road",
)

NAV_COMMAND_NAMES: tuple[str, ...] = (
    "KEEP_FOLLOW",
    "PREFER_LEFT",
    "PREFER_RIGHT",
    "PREPARE_EXIT_LEFT",
    "PREPARE_EXIT_RIGHT",
    "SLOW_CAUTION",
)


@dataclass(slots=True)
class AtlasHAConfig:
    # Source capture.
    source_h: int = 1080
    source_w: int = 1920
    source_capture_hz: float = 60.0

    # Model input.
    input_h: int = 864
    input_w: int = 1536
    input_channels: int = 3

    # Runtime rates.
    model_hz: float = 20.0
    controller_hz: float = 60.0
    ui_detector_hz: float = 30.0

    # Backbone.
    backbone_name: str = "regnet_y_800mf"
    pretrained_backbone: bool = True
    hidden_dim: int = 192
    fpn_dim: int = 128
    bifpn_repeats: int = 1

    # Visual tokenizer.
    tokens_per_frame: int = 128

    # Full-token temporal context.
    full_token_context_s: float = 30.0
    full_context_queries: int = 24
    full_context_blocks: int = 2
    cache_projected_kv: bool = True

    # Summary context.
    summary_context_s: float = 60.0
    summary_hz: float = 2.0
    summary_context_queries: int = 8
    summary_context_blocks: int = 1

    # Action history.
    action_context_s: float = 30.0
    action_sample_hz: float = 10.0
    action_dim: int = 6
    action_summary_tokens: int = 16
    action_dropout_prob: float = 0.15

    # Trajectory candidates.
    num_traj_candidates: int = 5
    num_traj_points: int = 30
    traj_horizon_s: float = 4.5

    # Lane geometry.
    num_lane_points: int = 24
    lane_long_min_m: float = 5.0
    lane_long_max_m: float = 120.0

    # Optional BEV auxiliary.
    enable_bev_aux: bool = False
    bev_long_min_m: float = 0.0
    bev_long_max_m: float = 120.0
    bev_lat_min_m: float = -24.0
    bev_lat_max_m: float = 24.0
    bev_resolution_m: float = 1.5
    bev_channels: int = 4

    # Runtime safety.
    enable_latency_compensation: bool = True
    nominal_control_latency_s: float = 0.10
    max_frame_staleness_caution_s: float = 0.10
    max_frame_staleness_fallback_s: float = 0.30
    enable_trajectory_sanitizer: bool = True
    enable_trajectory_legalizer: bool = True
    enable_minimum_risk_maneuver: bool = True

    # Vehicle/controller profile.
    default_vehicle_profile: str = "generic_gta_car_conservative"

    # UI handling.
    enable_ui_preprocessor: bool = True
    enable_phone_detector: bool = True
    enable_minimap_crop: bool = True

    # Command conditioning.
    use_command_conditioning: bool = True
    nav_cmd_dim: int = 6

    @property
    def full_context_frames(self) -> int:
        return int(round(self.full_token_context_s * self.model_hz))

    @property
    def summary_context_steps(self) -> int:
        return int(round(self.summary_context_s * self.summary_hz))

    @property
    def summary_stride_steps(self) -> int:
        return max(1, int(round(self.model_hz / self.summary_hz)))

    @property
    def action_history_steps(self) -> int:
        return int(round(self.action_context_s * self.action_sample_hz))

    @property
    def traj_dt_s(self) -> float:
        return float(self.traj_horizon_s) / float(self.num_traj_points)

    @property
    def bev_h(self) -> int:
        span = self.bev_long_max_m - self.bev_long_min_m
        return max(1, int(round(span / self.bev_resolution_m)))

    @property
    def bev_w(self) -> int:
        span = self.bev_lat_max_m - self.bev_lat_min_m
        return max(1, int(round(span / self.bev_resolution_m)))

    @property
    def expected_source_shape(self) -> tuple[int, int]:
        return (self.source_h, self.source_w)

    @property
    def expected_input_shape(self) -> tuple[int, int]:
        return (self.input_h, self.input_w)

    def validate(self) -> None:
        if self.source_h <= 0 or self.source_w <= 0:
            raise ValueError("source_h/source_w must be positive.")
        if self.input_h <= 0 or self.input_w <= 0:
            raise ValueError("input_h/input_w must be positive.")
        if self.input_channels != 3:
            raise ValueError("Atlas-HA v1 expects RGB input_channels=3.")
        if self.model_hz <= 0.0 or self.controller_hz <= 0.0:
            raise ValueError("model_hz and controller_hz must be positive.")
        if self.hidden_dim % 8 != 0:
            raise ValueError("hidden_dim must be divisible by 8 attention heads.")
        if self.tokens_per_frame <= 0:
            raise ValueError("tokens_per_frame must be positive.")
        if self.num_traj_candidates != len(TRAJECTORY_CANDIDATE_NAMES):
            raise ValueError("Atlas-HA v1 requires exactly 5 semantic candidates.")
        if self.num_traj_points <= 1:
            raise ValueError("num_traj_points must be greater than 1.")
        if self.num_lane_points <= 1:
            raise ValueError("num_lane_points must be greater than 1.")
        if self.action_dim != 6:
            raise ValueError("Atlas-HA action_dim must match [steer, throttle, brake, handbrake, reverse, pilot_active].")
        if self.nav_cmd_dim != len(NAV_COMMAND_NAMES):
            raise ValueError("nav_cmd_dim must match the v1 command classes.")


@dataclass(slots=True)
class AtlasHAStretchConfig(AtlasHAConfig):
    backbone_name: str = "regnet_y_1_6gf"
    fpn_dim: int = 160
    bifpn_repeats: int = 2
    model_hz: float = 20.0


def atlas_ha_default_config(*, pretrained_backbone: bool | None = None) -> AtlasHAConfig:
    cfg = AtlasHAConfig()
    if pretrained_backbone is not None:
        cfg.pretrained_backbone = bool(pretrained_backbone)
    cfg.validate()
    return cfg


def atlas_ha_stretch_config(*, pretrained_backbone: bool | None = None) -> AtlasHAStretchConfig:
    cfg = AtlasHAStretchConfig()
    if pretrained_backbone is not None:
        cfg.pretrained_backbone = bool(pretrained_backbone)
    cfg.validate()
    return cfg
