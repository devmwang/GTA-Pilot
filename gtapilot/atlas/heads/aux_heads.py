from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.nn as nn

from ..config import AtlasConfig
from ..state import AtlasWorldMemoryView
from ..utils import LearnedQueryPool


def _masked_slot_mean(slots: torch.Tensor, alive: torch.Tensor) -> torch.Tensor:
    weights = alive.to(dtype=slots.dtype, device=slots.device).clamp(0.0, 1.0).unsqueeze(-1)
    denom = weights.sum(dim=1).clamp(min=1.0)
    return (slots * weights).sum(dim=1) / denom


def _upsample_stack(in_ch: int, mid_ch: int, out_h: int, out_w: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1),
        nn.GELU(),
        nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1),
        nn.GELU(),
        nn.Upsample(size=(out_h, out_w), mode="bilinear", align_corners=False),
        nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1),
        nn.GELU(),
    )


class LaneSegmentHead(nn.Module):
    def __init__(self, cfg: AtlasConfig):
        super().__init__()
        self.cfg = cfg
        d_model = cfg.hidden_dim
        self.query_pool = LearnedQueryPool(
            cfg.lane.lane_queries,
            d_model,
            heads=min(8, max(1, d_model // 64)),
        )
        points = cfg.lane.lane_pts
        self.center_head = nn.Linear(d_model, points * 3)
        self.left_head = nn.Linear(d_model, points * 3)
        self.right_head = nn.Linear(d_model, points * 3)
        self.lane_cls = nn.Linear(d_model, len(cfg.lane.lane_semantic_classes))
        self.dir_cls = nn.Linear(d_model, len(cfg.lane.lane_direction_classes))
        n_kind = len(cfg.lane.boundary_kind_classes)
        n_color = len(cfg.lane.boundary_color_classes)
        n_pattern = len(cfg.lane.boundary_pattern_classes)
        n_cont = len(cfg.lane.boundary_continuity_classes)
        self.left_kind = nn.Linear(d_model, n_kind)
        self.left_color = nn.Linear(d_model, n_color)
        self.left_pattern = nn.Linear(d_model, n_pattern)
        self.left_cont = nn.Linear(d_model, n_cont)
        self.right_kind = nn.Linear(d_model, n_kind)
        self.right_color = nn.Linear(d_model, n_color)
        self.right_pattern = nn.Linear(d_model, n_pattern)
        self.right_cont = nn.Linear(d_model, n_cont)
        self.conf = nn.Linear(d_model, 1)

    def forward(self, world: AtlasWorldMemoryView) -> dict[str, torch.Tensor]:
        src = torch.cat(
            [world.lane_slots, world.static_grid.flatten(2).transpose(1, 2)], dim=1
        )
        query = self.query_pool(src)
        batch, queries, _ = query.shape
        points = self.cfg.lane.lane_pts
        return {
            "centerline": self.center_head(query).reshape(batch, queries, points, 3),
            "left_boundary": self.left_head(query).reshape(batch, queries, points, 3),
            "right_boundary": self.right_head(query).reshape(batch, queries, points, 3),
            "lane_sem_cls": self.lane_cls(query),
            "dir_cls": self.dir_cls(query),
            "left_kind": self.left_kind(query),
            "left_color": self.left_color(query),
            "left_pattern": self.left_pattern(query),
            "left_continuity": self.left_cont(query),
            "right_kind": self.right_kind(query),
            "right_color": self.right_color(query),
            "right_pattern": self.right_pattern(query),
            "right_continuity": self.right_cont(query),
            "pred_edge_succ": torch.matmul(query, query.transpose(1, 2)),
            "pred_edge_prev": torch.matmul(query, query.transpose(1, 2)),
            "pred_edge_left": torch.matmul(query, query.transpose(1, 2)),
            "pred_edge_right": torch.matmul(query, query.transpose(1, 2)),
            "lane_conf": self.conf(query),
            "lane_queries": query,
        }


class MapElementHead(nn.Module):
    def __init__(self, cfg: AtlasConfig):
        super().__init__()
        self.cfg = cfg
        d_model = cfg.hidden_dim
        self.query_pool = LearnedQueryPool(
            cfg.map_elem.queries,
            d_model,
            heads=min(8, max(1, d_model // 64)),
        )
        self.poly_head = nn.Linear(d_model, cfg.map_elem.pts * 3)
        self.cls_head = nn.Linear(d_model, len(cfg.map_elem.classes))
        self.conf_head = nn.Linear(d_model, 1)

    def forward(
        self,
        world: AtlasWorldMemoryView,
        lane_queries: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        src = torch.cat(
            [world.map_elem_slots, world.static_grid.flatten(2).transpose(1, 2)], dim=1
        )
        query = self.query_pool(src)
        batch, queries, _ = query.shape
        points = self.cfg.map_elem.pts
        bind = torch.matmul(lane_queries, query.transpose(1, 2))
        return {
            "map_poly": self.poly_head(query).reshape(batch, queries, points, 3),
            "map_elem_cls": self.cls_head(query),
            "map_elem_conf": self.conf_head(query),
            "lane_elem_bind": bind,
        }


class OccupancyHead(nn.Module):
    def __init__(self, cfg: AtlasConfig):
        super().__init__()
        self.cfg = cfg
        d_model = cfg.hidden_dim
        channels = cfg.occupancy.decoder_bev_channels
        self.project = nn.Conv2d(d_model, channels, kernel_size=1)
        self.decoder = _upsample_stack(
            channels,
            channels,
            cfg.occupancy.out_y,
            cfg.occupancy.out_x,
        )
        self.state_head = nn.Conv2d(
            channels,
            cfg.occupancy.state_classes * cfg.occupancy.out_z,
            kernel_size=1,
        )
        self.sem_head = nn.Conv2d(
            channels,
            cfg.occupancy.semantic_classes * cfg.occupancy.out_z,
            kernel_size=1,
        )
        self.dynamic_to_bev = nn.Linear(d_model, channels)
        self.speculative_to_bev = nn.Linear(d_model, channels)

    def forward(self, world: AtlasWorldMemoryView) -> dict[str, torch.Tensor]:
        bev = self.project(world.static_grid)
        dyn_bias = self.dynamic_to_bev(
            _masked_slot_mean(world.dynamic_slots, world.dynamic_slot_alive)
        )[:, :, None, None]
        spec_bias = self.speculative_to_bev(
            _masked_slot_mean(world.speculative_slots, world.speculative_slot_alive)
        )[
            :, :, None, None
        ]
        bev = self.decoder(bev + dyn_bias + spec_bias)
        state = self.state_head(bev).reshape(
            bev.shape[0],
            self.cfg.occupancy.state_classes,
            self.cfg.occupancy.out_z,
            self.cfg.occupancy.out_y,
            self.cfg.occupancy.out_x,
        )
        sem = self.sem_head(bev).reshape(
            bev.shape[0],
            self.cfg.occupancy.semantic_classes,
            self.cfg.occupancy.out_z,
            self.cfg.occupancy.out_y,
            self.cfg.occupancy.out_x,
        )
        return {"occ_state": state, "occ_sem": sem}


class BEVLiteHead(nn.Module):
    def __init__(self, cfg: AtlasConfig):
        super().__init__()
        self.cfg = cfg
        d_model = cfg.hidden_dim
        channels = 96
        self.project = nn.Conv2d(d_model, channels, kernel_size=1)
        self.decoder = _upsample_stack(
            channels,
            channels,
            cfg.bev_lite.out_h,
            cfg.bev_lite.out_w,
        )
        self.bev_head = nn.Conv2d(channels, cfg.bev_lite.channels, kernel_size=1)

    def forward(self, world: AtlasWorldMemoryView) -> dict[str, torch.Tensor]:
        bev = self.decoder(self.project(world.static_grid))
        return {"bev_lite": self.bev_head(bev)}


class ActorHead(nn.Module):
    def __init__(self, cfg: AtlasConfig):
        super().__init__()
        self.cfg = cfg
        d_model = cfg.hidden_dim
        self.track_pool = LearnedQueryPool(
            cfg.actor.queries,
            d_model,
            heads=min(8, max(1, d_model // 64)),
        )
        self.spec_pool = LearnedQueryPool(
            cfg.actor.speculative_queries,
            d_model,
            heads=min(8, max(1, d_model // 64)),
        )
        self.cls_head = nn.Linear(d_model, len(cfg.actor.classes))
        self.box_head = nn.Linear(d_model, 7)
        self.vel_head = nn.Linear(d_model, 2)
        self.future_head = nn.Linear(d_model, cfg.actor.future_steps * 2)
        self.spec_cls_head = nn.Linear(d_model, len(cfg.actor.classes))
        self.spec_risk_head = nn.Linear(d_model, 1)
        self.spec_future_head = nn.Linear(d_model, cfg.actor.future_steps * 2)

    def forward(self, world: AtlasWorldMemoryView) -> dict[str, torch.Tensor]:
        dynamic_mask = (
            ~world.dynamic_slot_alive
            if world.dynamic_slot_alive.dtype == torch.bool
            else world.dynamic_slot_alive <= 1e-4
        )
        speculative_mask = (
            ~world.speculative_slot_alive
            if world.speculative_slot_alive.dtype == torch.bool
            else world.speculative_slot_alive <= 1e-4
        )
        track_query = self.track_pool(
            world.dynamic_slots,
            key_padding_mask=dynamic_mask,
        )
        spec_query = self.spec_pool(
            world.speculative_slots,
            key_padding_mask=speculative_mask,
        )
        track_query = track_query * world.dynamic_slot_alive.to(track_query.dtype).amax(
            dim=1, keepdim=True
        ).unsqueeze(-1)
        spec_query = spec_query * world.speculative_slot_alive.to(spec_query.dtype).amax(
            dim=1, keepdim=True
        ).unsqueeze(-1)
        batch, queries, _ = track_query.shape
        spec_queries = spec_query.shape[1]
        return {
            "actor_cls": self.cls_head(track_query),
            "actor_box": self.box_head(track_query),
            "actor_vel": self.vel_head(track_query),
            "actor_future": self.future_head(track_query).reshape(
                batch, queries, self.cfg.actor.future_steps, 2
            ),
            "spec_actor_cls": self.spec_cls_head(spec_query),
            "spec_actor_risk": self.spec_risk_head(spec_query),
            "spec_actor_future": self.spec_future_head(spec_query).reshape(
                batch, spec_queries, self.cfg.actor.future_steps, 2
            ),
        }


class EgoHead(nn.Module):
    def __init__(self, cfg: AtlasConfig):
        super().__init__()
        d_model = cfg.hidden_dim
        self.out = nn.Linear(d_model, 7)
        self.logvar = nn.Linear(d_model, 7)

    def forward(self, world: AtlasWorldMemoryView) -> dict[str, torch.Tensor]:
        x = world.ego_tokens.mean(dim=1)
        return {
            "ego_out": self.out(x),
            "ego_logvar": self.logvar(x).clamp(min=-6.0, max=4.0),
        }


class AuxHeads(nn.Module):
    def __init__(self, cfg: AtlasConfig):
        super().__init__()
        self.cfg = cfg
        self.lane = LaneSegmentHead(cfg)
        self.map_elem = MapElementHead(cfg)
        self.occupancy = OccupancyHead(cfg)
        self.bev = BEVLiteHead(cfg)
        self.actors = ActorHead(cfg)
        self.ego = EgoHead(cfg)

    def forward(
        self,
        world: AtlasWorldMemoryView,
        active_heads: Optional[Iterable[str]] = None,
        risk_context: Optional[dict[str, torch.Tensor]] = None,
    ) -> dict[str, torch.Tensor]:
        active = set(
            active_heads or {"lane", "map", "occupancy", "bev", "actors", "ego"}
        )
        out: dict[str, torch.Tensor] = {}
        lane_out: dict[str, torch.Tensor] = {}
        if "lane" in active or "map" in active:
            lane_out = self.lane(world)
            if "lane" in active:
                out.update(lane_out)
        if "map" in active:
            lane_queries = lane_out.get("lane_queries")
            if lane_queries is None:
                lane_queries = self.lane(world)["lane_queries"]
            out.update(self.map_elem(world, lane_queries))
        if "occupancy" in active:
            out.update(self.occupancy(world))
            if risk_context is not None:
                out["dyn_flow_bev"] = risk_context["dyn_flow_bev"]
                out["occl_risk_bev"] = risk_context["occl_risk_bev"]
                out["provenance"] = risk_context["provenance"]
        if "bev" in active:
            out.update(self.bev(world))
            if risk_context is not None and "provenance" not in out:
                out["provenance"] = risk_context["provenance"]
        if "actors" in active:
            out.update(self.actors(world))
        if "ego" in active:
            out.update(self.ego(world))
        return out
