from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.nn as nn

from ..config import AtlasConfig
from ..state import AtlasWorldMemoryView
from ..utils import LearnedQueryPool


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
        D = cfg.hidden_dim
        self.query_pool = LearnedQueryPool(cfg.lane.lane_queries, D, heads=min(8, max(1, D // 64)))
        pts = cfg.lane.lane_pts
        self.center_head = nn.Linear(D, pts * 3)
        self.left_head = nn.Linear(D, pts * 3)
        self.right_head = nn.Linear(D, pts * 3)
        self.lane_cls = nn.Linear(D, len(cfg.lane.lane_semantic_classes))
        self.dir_cls = nn.Linear(D, len(cfg.lane.lane_direction_classes))
        n_kind = len(cfg.lane.boundary_kind_classes)
        n_color = len(cfg.lane.boundary_color_classes)
        n_pattern = len(cfg.lane.boundary_pattern_classes)
        n_cont = len(cfg.lane.boundary_continuity_classes)
        self.left_kind = nn.Linear(D, n_kind)
        self.left_color = nn.Linear(D, n_color)
        self.left_pattern = nn.Linear(D, n_pattern)
        self.left_cont = nn.Linear(D, n_cont)
        self.right_kind = nn.Linear(D, n_kind)
        self.right_color = nn.Linear(D, n_color)
        self.right_pattern = nn.Linear(D, n_pattern)
        self.right_cont = nn.Linear(D, n_cont)
        self.conf = nn.Linear(D, 1)

    def forward(self, world: AtlasWorldMemoryView) -> dict[str, torch.Tensor]:
        src = torch.cat([world.lane_slots, world.static_grid.flatten(2).transpose(1, 2)], dim=1)
        q = self.query_pool(src)
        B, Q, D = q.shape
        P = self.cfg.lane.lane_pts
        return {
            "centerline": self.center_head(q).reshape(B, Q, P, 3),
            "left_boundary": self.left_head(q).reshape(B, Q, P, 3),
            "right_boundary": self.right_head(q).reshape(B, Q, P, 3),
            "lane_sem_cls": self.lane_cls(q),
            "dir_cls": self.dir_cls(q),
            "left_kind": self.left_kind(q),
            "left_color": self.left_color(q),
            "left_pattern": self.left_pattern(q),
            "left_continuity": self.left_cont(q),
            "right_kind": self.right_kind(q),
            "right_color": self.right_color(q),
            "right_pattern": self.right_pattern(q),
            "right_continuity": self.right_cont(q),
            "pred_edge_succ": torch.matmul(q, q.transpose(1, 2)),
            "pred_edge_prev": torch.matmul(q, q.transpose(1, 2)),
            "pred_edge_left": torch.matmul(q, q.transpose(1, 2)),
            "pred_edge_right": torch.matmul(q, q.transpose(1, 2)),
            "lane_conf": self.conf(q),
            "lane_queries": q,
        }


class MapElementHead(nn.Module):
    def __init__(self, cfg: AtlasConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.hidden_dim
        self.query_pool = LearnedQueryPool(cfg.map_elem.queries, D, heads=min(8, max(1, D // 64)))
        self.poly_head = nn.Linear(D, cfg.map_elem.pts * 3)
        self.cls_head = nn.Linear(D, len(cfg.map_elem.classes))
        self.conf_head = nn.Linear(D, 1)

    def forward(
        self, world: AtlasWorldMemoryView, lane_queries: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        src = torch.cat([world.map_elem_slots, world.static_grid.flatten(2).transpose(1, 2)], dim=1)
        q = self.query_pool(src)
        B, Q, D = q.shape
        P = self.cfg.map_elem.pts
        bind = torch.matmul(lane_queries, q.transpose(1, 2))
        return {
            "map_poly": self.poly_head(q).reshape(B, Q, P, 3),
            "map_elem_cls": self.cls_head(q),
            "map_elem_conf": self.conf_head(q),
            "lane_elem_bind": bind,
        }


class OccupancyHead(nn.Module):
    def __init__(self, cfg: AtlasConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.hidden_dim
        C = cfg.occupancy.decoder_bev_channels
        self.project = nn.Conv2d(D, C, kernel_size=1)
        self.decoder = _upsample_stack(C, C, cfg.occupancy.out_y, cfg.occupancy.out_x)
        self.state_head = nn.Conv2d(C, cfg.occupancy.state_classes * cfg.occupancy.out_z, kernel_size=1)
        self.sem_head = nn.Conv2d(C, cfg.occupancy.semantic_classes * cfg.occupancy.out_z, kernel_size=1)
        self.dynamic_to_bev = nn.Linear(D, C)

    def forward(self, world: AtlasWorldMemoryView) -> dict[str, torch.Tensor]:
        x = self.project(world.static_grid)
        dyn_bias = self.dynamic_to_bev(world.dynamic_slots.mean(dim=1))[:, :, None, None]
        x = self.decoder(x + dyn_bias)
        state = self.state_head(x).reshape(
            x.shape[0], self.cfg.occupancy.state_classes, self.cfg.occupancy.out_z, self.cfg.occupancy.out_y, self.cfg.occupancy.out_x
        )
        sem = self.sem_head(x).reshape(
            x.shape[0], self.cfg.occupancy.semantic_classes, self.cfg.occupancy.out_z, self.cfg.occupancy.out_y, self.cfg.occupancy.out_x
        )
        return {"occ_state": state, "occ_sem": sem}


class BEVLiteHead(nn.Module):
    def __init__(self, cfg: AtlasConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.hidden_dim
        C = 96
        self.project = nn.Conv2d(D, C, kernel_size=1)
        self.decoder = _upsample_stack(C, C, cfg.bev_lite.out_h, cfg.bev_lite.out_w)
        self.bev_head = nn.Conv2d(C, cfg.bev_lite.channels, kernel_size=1)
        self.prov_head = nn.Conv2d(C, cfg.bev_lite.provenance_classes, kernel_size=1)

    def forward(self, world: AtlasWorldMemoryView) -> dict[str, torch.Tensor]:
        x = self.decoder(self.project(world.static_grid))
        return {
            "bev_lite": self.bev_head(x),
            "provenance": self.prov_head(x),
        }


class ActorHead(nn.Module):
    def __init__(self, cfg: AtlasConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.hidden_dim
        self.query_pool = LearnedQueryPool(cfg.actor.queries, D, heads=min(8, max(1, D // 64)))
        self.cls_head = nn.Linear(D, len(cfg.actor.classes))
        self.box_head = nn.Linear(D, 7)
        self.vel_head = nn.Linear(D, 2)
        self.future_head = nn.Linear(D, cfg.actor.future_steps * 2)

    def forward(self, world: AtlasWorldMemoryView) -> dict[str, torch.Tensor]:
        q = self.query_pool(world.dynamic_slots)
        B, Q, D = q.shape
        return {
            "actor_cls": self.cls_head(q),
            "actor_box": self.box_head(q),
            "actor_vel": self.vel_head(q),
            "actor_future": self.future_head(q).reshape(B, Q, self.cfg.actor.future_steps, 2),
        }


class EgoHead(nn.Module):
    def __init__(self, cfg: AtlasConfig):
        super().__init__()
        D = cfg.hidden_dim
        self.out = nn.Linear(D, 7)
        self.logvar = nn.Linear(D, 7)

    def forward(self, world: AtlasWorldMemoryView) -> dict[str, torch.Tensor]:
        x = world.ego_tokens.mean(dim=1)
        return {"ego_out": self.out(x), "ego_logvar": self.logvar(x).clamp(min=-6.0, max=4.0)}


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
    ) -> dict[str, torch.Tensor]:
        active = set(active_heads or {"lane", "map", "occupancy", "bev", "actors", "ego"})
        out: dict[str, torch.Tensor] = {}
        lane_out = {}
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
        if "bev" in active:
            out.update(self.bev(world))
        if "actors" in active:
            out.update(self.actors(world))
        if "ego" in active:
            out.update(self.ego(world))
        return out


AtlasAuxHeads = AuxHeads
