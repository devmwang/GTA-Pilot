from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Tuple


@dataclass
class ImageConfig:
    raw_height: int = 1080
    raw_width: int = 1920
    padded_height: int = 1088
    padded_width: int = 1920
    channels: int = 3
    hfov_deg: float = 90.0
    depth_min_m: float = 0.5
    depth_max_m: float = 80.0
    depth_bins: int = 64

    @property
    def cx(self) -> float:
        return self.raw_width / 2.0

    @property
    def cy(self) -> float:
        return self.raw_height / 2.0

    @property
    def fx(self) -> float:
        return self.raw_width / (2.0 * math.tan(math.radians(self.hfov_deg) / 2.0))

    @property
    def fy(self) -> float:
        vfov = 2.0 * math.atan(
            math.tan(math.radians(self.hfov_deg) / 2.0)
            * (self.raw_height / self.raw_width)
        )
        return self.raw_height / (2.0 * math.tan(vfov / 2.0))


@dataclass
class VisionTokenizerConfig:
    encoder_type: str = "dualpath_hvt_bifpn"
    token_dim: int = 448
    stem_hidden_dim: int = 48
    detail_dim: int = 96
    detail_blocks: int = 4
    detail_8x_dim: int = 192
    ctx_dims: Tuple[int, int, int, int] = (128, 256, 384, 512)
    ctx_blocks: Tuple[int, int, int, int] = (2, 4, 8, 4)
    ctx_heads: Tuple[int, int, int, int] = (4, 8, 12, 16)
    ctx_window_sizes: Tuple[int, int, int, int] = (8, 8, 8, 8)
    cam_tokens_per_frame: int = 192
    fusion_dim: int = 256
    bifpn_repeats: int = 2
    query_pool_heads: int = 8
    dropout: float = 0.1
    freeze_stages: int = 2
    use_bifpn: bool = True
    include_detail_summary: bool = True


@dataclass
class TemporalMixerConfig:
    num_frames: int = 8
    mixer_blocks: int = 4
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.1


@dataclass
class ActionEncoderConfig:
    action_dim: int = 6
    history_len: int = 20
    act_tokens: int = 4
    hidden_dim: int = 192
    num_heads: int = 8
    dropout: float = 0.1


@dataclass
class EgoFilterConfig:
    hidden_size: int = 192
    num_layers: int = 2
    ego_tokens: int = 6


@dataclass
class GeometryLifterConfig:
    frustum_tokens: int = 384
    depth_bins: int = 64
    track_history: int = 7
    grid_h_8x: int = 136
    grid_w_8x: int = 240
    min_depth_m: float = 0.5
    max_depth_m: float = 80.0


@dataclass
class ObservationPoolConfig:
    obs_tokens: int = 96
    num_heads: int = 8


@dataclass
class WorldMemoryConfig:
    static_grid_h: int = 20
    static_grid_w: int = 14
    dynamic_slots: int = 64
    lane_slots: int = 32
    map_elem_slots: int = 16
    ego_tokens: int = 6
    route_tokens: int = 8
    reasoner_tokens: int = 8
    world_blocks: int = 8
    num_heads: int = 8
    x_range_m: Tuple[float, float] = (-16.0, 64.0)
    y_range_m: Tuple[float, float] = (-21.0, 21.0)
    static_write_dropout: float = 0.1

    @property
    def cell_x_m(self) -> float:
        return (self.x_range_m[1] - self.x_range_m[0]) / self.static_grid_h

    @property
    def cell_y_m(self) -> float:
        return (self.y_range_m[1] - self.y_range_m[0]) / self.static_grid_w


@dataclass
class PlannerConfig:
    proposals: int = 10
    decoder_blocks: int = 4
    evaluator_rollout_steps: int = 5
    control_steps: int = 20
    control_dt: float = 0.2
    waypoint_dt: float = 0.1
    reward_terms: Tuple[str, ...] = (
        "collision",
        "offroad",
        "rule",
        "progress",
        "comfort",
        "uncertainty",
        "route",
    )
    num_heads: int = 8


@dataclass
class LaneHeadConfig:
    lane_queries: int = 32
    lane_pts: int = 20
    lane_semantic_classes: Tuple[str, ...] = (
        "driving",
        "merge",
        "split",
        "turn_left",
        "turn_right",
        "shoulder_parking",
        "intersection_imputed",
        "other",
    )
    lane_direction_classes: Tuple[str, ...] = (
        "same",
        "opposite",
        "bidirectional",
        "unknown",
    )
    boundary_kind_classes: Tuple[str, ...] = (
        "paint",
        "curb",
        "barrier",
        "inferred",
        "unknown",
    )
    boundary_color_classes: Tuple[str, ...] = (
        "white",
        "yellow",
        "red",
        "other",
        "unknown",
    )
    boundary_pattern_classes: Tuple[str, ...] = (
        "solid",
        "dashed",
        "double",
        "botts",
        "unknown",
    )
    boundary_continuity_classes: Tuple[str, ...] = (
        "continuous",
        "broken",
        "mixed",
        "unknown",
    )


@dataclass
class MapElementHeadConfig:
    queries: int = 16
    pts: int = 20
    classes: Tuple[str, ...] = (
        "crosswalk_boundary",
        "road_boundary",
        "stop_line",
        "keepout_other",
    )


@dataclass
class OccupancyHeadConfig:
    x_m: float = 64.0
    y_m: float = 36.0
    z_m: float = 6.0
    voxel_m: float = 0.5
    state_classes: int = 3
    semantic_classes: int = 16
    decoder_bev_channels: int = 128
    out_z: int = 12
    out_y: int = 72
    out_x: int = 128


@dataclass
class BEVLiteHeadConfig:
    channels: int = 8
    provenance_classes: int = 3
    out_h: int = 96
    out_w: int = 160


@dataclass
class ActorHeadConfig:
    queries: int = 40
    classes: Tuple[str, ...] = (
        "car",
        "large_vehicle",
        "two_wheeler",
        "pedestrian",
        "dynamic_static",
        "other",
    )
    future_steps: int = 10


@dataclass
class HeadSchedulerConfig:
    drive_mode_cadence: Dict[str, int] = field(
        default_factory=lambda: {
            "planner": 1,
            "ego": 1,
            "lane": 1,
            "map": 1,
            "bev": 1,
            "actors": 2,
            "occupancy": 3,
        }
    )
    inspect_mode_cadence: Dict[str, int] = field(
        default_factory=lambda: {
            "planner": 1,
            "ego": 1,
            "lane": 1,
            "map": 1,
            "bev": 1,
            "actors": 1,
            "occupancy": 1,
        }
    )


@dataclass
class RouteAdapterConfig:
    route_points: int = 32
    route_tokens: int = 8
    nav_cmd_dim: int = 6


@dataclass
class ReasonerAdapterConfig:
    input_dim: int = 448
    output_tokens: int = 8
    ttl_steps: int = 20


@dataclass
class AtlasConfig:
    variant: str
    hidden_dim: int
    image: ImageConfig = field(default_factory=ImageConfig)
    vision: VisionTokenizerConfig = field(default_factory=VisionTokenizerConfig)
    temporal: TemporalMixerConfig = field(default_factory=TemporalMixerConfig)
    action: ActionEncoderConfig = field(default_factory=ActionEncoderConfig)
    ego: EgoFilterConfig = field(default_factory=EgoFilterConfig)
    geometry: GeometryLifterConfig = field(default_factory=GeometryLifterConfig)
    obs_pool: ObservationPoolConfig = field(default_factory=ObservationPoolConfig)
    world: WorldMemoryConfig = field(default_factory=WorldMemoryConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    lane: LaneHeadConfig = field(default_factory=LaneHeadConfig)
    map_elem: MapElementHeadConfig = field(default_factory=MapElementHeadConfig)
    occupancy: OccupancyHeadConfig = field(default_factory=OccupancyHeadConfig)
    bev_lite: BEVLiteHeadConfig = field(default_factory=BEVLiteHeadConfig)
    actor: ActorHeadConfig = field(default_factory=ActorHeadConfig)
    scheduler: HeadSchedulerConfig = field(default_factory=HeadSchedulerConfig)
    route_adapter: RouteAdapterConfig = field(default_factory=RouteAdapterConfig)
    reasoner_adapter: ReasonerAdapterConfig = field(
        default_factory=ReasonerAdapterConfig
    )
    enable_route_tokens: bool = True
    enable_reasoner_tokens: bool = True
    enable_privileged_teacher_adapters: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def atlas_s1080_config() -> AtlasConfig:
    cfg = AtlasConfig(variant="Atlas-S1080", hidden_dim=448)
    cfg.vision.encoder_type = "dualpath_hvt_bifpn"
    cfg.vision.token_dim = 448
    cfg.vision.stem_hidden_dim = 48
    cfg.vision.detail_dim = 96
    cfg.vision.detail_blocks = 4
    cfg.vision.detail_8x_dim = 192
    cfg.vision.cam_tokens_per_frame = 192
    cfg.vision.ctx_dims = (128, 256, 384, 512)
    cfg.vision.ctx_blocks = (2, 4, 8, 4)
    cfg.vision.ctx_heads = (4, 8, 12, 16)
    cfg.vision.ctx_window_sizes = (8, 8, 8, 8)
    cfg.vision.fusion_dim = 256
    cfg.vision.bifpn_repeats = 2
    cfg.vision.freeze_stages = 2
    cfg.temporal.mixer_blocks = 4
    cfg.action.hidden_dim = 192
    cfg.ego.hidden_size = 192
    cfg.geometry.frustum_tokens = 384
    cfg.obs_pool.obs_tokens = 96
    cfg.world.static_grid_h = 20
    cfg.world.static_grid_w = 14
    cfg.world.dynamic_slots = 64
    cfg.world.lane_slots = 32
    cfg.world.map_elem_slots = 16
    cfg.world.ego_tokens = 6
    cfg.world.route_tokens = 8
    cfg.world.reasoner_tokens = 8
    cfg.world.world_blocks = 8
    cfg.planner.proposals = 10
    cfg.planner.decoder_blocks = 4
    cfg.planner.evaluator_rollout_steps = 5
    cfg.map_elem.queries = 16
    cfg.actor.queries = 40
    cfg.reasoner_adapter.input_dim = 448
    return cfg


def atlas_t1080_priv_config() -> AtlasConfig:
    cfg = AtlasConfig(variant="Atlas-T1080-Priv", hidden_dim=640)
    cfg.vision.encoder_type = "dualpath_hvt_bifpn"
    cfg.vision.token_dim = 640
    cfg.vision.stem_hidden_dim = 64
    cfg.vision.detail_dim = 128
    cfg.vision.detail_blocks = 4
    cfg.vision.detail_8x_dim = 256
    cfg.vision.ctx_dims = (160, 320, 512, 640)
    cfg.vision.ctx_blocks = (3, 6, 10, 6)
    cfg.vision.ctx_heads = (5, 10, 16, 20)
    cfg.vision.ctx_window_sizes = (8, 8, 8, 8)
    cfg.vision.cam_tokens_per_frame = 256
    cfg.vision.fusion_dim = 320
    cfg.vision.bifpn_repeats = 3
    cfg.vision.query_pool_heads = 10
    cfg.vision.freeze_stages = 1
    cfg.temporal.mixer_blocks = 6
    cfg.temporal.num_heads = 10
    cfg.action.hidden_dim = 256
    cfg.action.num_heads = 10
    cfg.ego.hidden_size = 256
    cfg.ego.ego_tokens = 8
    cfg.geometry.frustum_tokens = 512
    cfg.obs_pool.obs_tokens = 128
    cfg.obs_pool.num_heads = 10
    cfg.world.static_grid_h = 24
    cfg.world.static_grid_w = 16
    cfg.world.dynamic_slots = 96
    cfg.world.lane_slots = 48
    cfg.world.map_elem_slots = 24
    cfg.world.ego_tokens = 8
    cfg.world.route_tokens = 8
    cfg.world.reasoner_tokens = 8
    cfg.world.world_blocks = 10
    cfg.world.num_heads = 10
    cfg.planner.proposals = 14
    cfg.planner.decoder_blocks = 5
    cfg.planner.evaluator_rollout_steps = 6
    cfg.planner.num_heads = 10
    cfg.lane.lane_queries = 48
    cfg.map_elem.queries = 24
    cfg.actor.queries = 64
    cfg.reasoner_adapter.input_dim = 640
    cfg.enable_privileged_teacher_adapters = True
    return cfg


def atlas_smoke_config() -> AtlasConfig:
    cfg = AtlasConfig(variant="Atlas-SMOKE", hidden_dim=64)
    cfg.image.raw_height = 64
    cfg.image.raw_width = 96
    cfg.image.padded_height = 64
    cfg.image.padded_width = 96
    cfg.vision.encoder_type = "dualpath_hvt_bifpn"
    cfg.vision.token_dim = 64
    cfg.vision.stem_hidden_dim = 8
    cfg.vision.detail_dim = 16
    cfg.vision.detail_blocks = 2
    cfg.vision.detail_8x_dim = 24
    cfg.vision.ctx_dims = (24, 32, 48, 64)
    cfg.vision.ctx_blocks = (1, 1, 2, 1)
    cfg.vision.ctx_heads = (2, 2, 4, 4)
    cfg.vision.ctx_window_sizes = (4, 4, 4, 4)
    cfg.vision.cam_tokens_per_frame = 8
    cfg.vision.fusion_dim = 32
    cfg.vision.bifpn_repeats = 1
    cfg.vision.query_pool_heads = 2
    cfg.temporal.num_frames = 4
    cfg.temporal.mixer_blocks = 2
    cfg.temporal.num_heads = 2
    cfg.action.history_len = 8
    cfg.action.act_tokens = 2
    cfg.action.hidden_dim = 32
    cfg.action.num_heads = 2
    cfg.ego.hidden_size = 32
    cfg.ego.ego_tokens = 3
    cfg.geometry.frustum_tokens = 8
    cfg.geometry.grid_h_8x = cfg.image.padded_height // 8
    cfg.geometry.grid_w_8x = cfg.image.padded_width // 8
    cfg.geometry.track_history = cfg.temporal.num_frames - 1
    cfg.obs_pool.obs_tokens = 4
    cfg.obs_pool.num_heads = 4
    cfg.world.static_grid_h = 4
    cfg.world.static_grid_w = 3
    cfg.world.dynamic_slots = 4
    cfg.world.lane_slots = 4
    cfg.world.map_elem_slots = 2
    cfg.world.ego_tokens = 3
    cfg.world.route_tokens = 2
    cfg.world.reasoner_tokens = 2
    cfg.world.world_blocks = 1
    cfg.world.num_heads = 2
    cfg.planner.proposals = 3
    cfg.planner.decoder_blocks = 1
    cfg.planner.evaluator_rollout_steps = 2
    cfg.planner.control_steps = 4
    cfg.planner.num_heads = 2
    cfg.lane.lane_queries = 2
    cfg.lane.lane_pts = 4
    cfg.map_elem.queries = 1
    cfg.map_elem.pts = 4
    cfg.occupancy.out_z = 2
    cfg.occupancy.out_y = 4
    cfg.occupancy.out_x = 6
    cfg.bev_lite.out_h = 6
    cfg.bev_lite.out_w = 8
    cfg.actor.queries = 2
    cfg.actor.future_steps = 2
    cfg.reasoner_adapter.input_dim = 64
    return cfg


def build_atlas_config(variant: str) -> AtlasConfig:
    variant_key = variant.lower()
    if variant_key in {"atlas-s1080", "s1080", "student"}:
        return atlas_s1080_config()
    if variant_key in {"atlas-t1080-priv", "t1080-priv", "teacher"}:
        return atlas_t1080_priv_config()
    if variant_key in {"atlas-smoke", "smoke"}:
        return atlas_smoke_config()
    raise ValueError(f"Unknown Atlas config variant: {variant}")
