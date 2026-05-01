from __future__ import annotations

import torch
import torch.nn as nn

from .config import AtlasHAConfig


def _mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(in_dim),
        nn.Linear(in_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, out_dim),
    )


class TrajectoryHead(nn.Module):
    def __init__(self, cfg: AtlasHAConfig):
        super().__init__()
        self.cfg = cfg
        self.delta = _mlp(
            cfg.hidden_dim,
            cfg.hidden_dim,
            cfg.num_traj_candidates * cfg.num_traj_points * 4,
        )
        self.logits = _mlp(cfg.hidden_dim, cfg.hidden_dim, cfg.num_traj_candidates)
        long_samples = torch.linspace(2.0, 90.0, cfg.num_traj_points)
        lane_width = 3.6
        smooth = torch.linspace(0.0, 1.0, cfg.num_traj_points)
        smooth = smooth * smooth * (3.0 - 2.0 * smooth)
        base = torch.zeros(cfg.num_traj_candidates, cfg.num_traj_points, 4)
        base[:, :, 0] = long_samples
        base[0, :, 3] = 24.0
        base[1, :, 3] = 14.0
        base[2, :, 1] = lane_width * smooth
        base[2, :, 3] = 20.0
        base[3, :, 1] = -lane_width * smooth
        base[3, :, 3] = 20.0
        base[4, :, 3] = torch.linspace(12.0, 0.0, cfg.num_traj_points)
        self.register_buffer("base_traj", base, persistent=False)

    def forward(self, head_context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = head_context.shape[0]
        delta = self.delta(head_context).view(
            batch_size,
            self.cfg.num_traj_candidates,
            self.cfg.num_traj_points,
            4,
        )
        scaled = torch.empty_like(delta)
        scaled[..., 0] = torch.tanh(delta[..., 0]) * 6.0
        scaled[..., 1] = torch.tanh(delta[..., 1]) * 2.0
        scaled[..., 2] = torch.tanh(delta[..., 2]) * 0.35
        scaled[..., 3] = torch.tanh(delta[..., 3]) * 6.0
        traj = self.base_traj.to(device=head_context.device, dtype=head_context.dtype)
        traj = traj.unsqueeze(0) + scaled
        traj[..., 3] = traj[..., 3].clamp_min(0.0)
        return traj, self.logits(head_context)


class LaneRoadHead(nn.Module):
    def __init__(self, cfg: AtlasHAConfig):
        super().__init__()
        self.cfg = cfg
        self.lane_lat = _mlp(cfg.hidden_dim, cfg.hidden_dim, 3 * cfg.num_lane_points)
        self.lane_valid = _mlp(cfg.hidden_dim, cfg.hidden_dim, 3 * cfg.num_lane_points)
        self.lane_conf = _mlp(cfg.hidden_dim, cfg.hidden_dim, 3)
        self.road_edge_lat = _mlp(cfg.hidden_dim, cfg.hidden_dim, 2 * cfg.num_lane_points)
        self.road_edge_conf = _mlp(cfg.hidden_dim, cfg.hidden_dim, 2)
        lane_long_samples = torch.linspace(
            cfg.lane_long_min_m,
            cfg.lane_long_max_m,
            cfg.num_lane_points,
        )
        self.register_buffer("lane_long_samples_m", lane_long_samples, persistent=False)

    def forward(self, head_context: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = head_context.shape[0]
        lane_lat = self.lane_lat(head_context).view(batch_size, 3, self.cfg.num_lane_points)
        road_edge_lat = self.road_edge_lat(head_context).view(
            batch_size,
            2,
            self.cfg.num_lane_points,
        )
        return {
            "lane_lat_pred": lane_lat,
            "lane_valid_logit": self.lane_valid(head_context).view(
                batch_size,
                3,
                self.cfg.num_lane_points,
            ),
            "lane_conf": torch.sigmoid(self.lane_conf(head_context)),
            "road_edge_lat_pred": road_edge_lat,
            "road_edge_conf": torch.sigmoid(self.road_edge_conf(head_context)),
            "lane_long_samples_m": self.lane_long_samples_m.to(
                device=head_context.device,
                dtype=head_context.dtype,
            ),
        }


class LeadAdjacentHead(nn.Module):
    def __init__(self, cfg: AtlasHAConfig):
        super().__init__()
        self.lead_present = _mlp(cfg.hidden_dim, cfg.hidden_dim, 1)
        self.lead_state = _mlp(cfg.hidden_dim, cfg.hidden_dim, 5)
        self.adjacent_left = _mlp(cfg.hidden_dim, cfg.hidden_dim, 4)
        self.adjacent_right = _mlp(cfg.hidden_dim, cfg.hidden_dim, 4)

    def forward(self, head_context: torch.Tensor) -> dict[str, torch.Tensor]:
        lead_state = self.lead_state(head_context)
        lead_state = torch.stack(
            [
                lead_state[:, 0].clamp_min(0.0),
                lead_state[:, 1],
                lead_state[:, 2].clamp_min(0.0),
                lead_state[:, 3],
                lead_state[:, 4].clamp(0.0, 20.0),
            ],
            dim=-1,
        )
        adjacent_left = self.adjacent_left(head_context)
        adjacent_right = self.adjacent_right(head_context)
        adjacent_left = torch.cat(
            [torch.sigmoid(adjacent_left[:, :1]), adjacent_left[:, 1:3], torch.sigmoid(adjacent_left[:, 3:4])],
            dim=-1,
        )
        adjacent_right = torch.cat(
            [torch.sigmoid(adjacent_right[:, :1]), adjacent_right[:, 1:3], torch.sigmoid(adjacent_right[:, 3:4])],
            dim=-1,
        )
        return {
            "lead_present_logit": self.lead_present(head_context),
            "lead_state": lead_state,
            "adjacent_left": adjacent_left,
            "adjacent_right": adjacent_right,
        }


class EgoHead(nn.Module):
    def __init__(self, cfg: AtlasHAConfig):
        super().__init__()
        self.net = _mlp(cfg.hidden_dim, cfg.hidden_dim, 4)

    def forward(self, head_context: torch.Tensor) -> torch.Tensor:
        ego = self.net(head_context)
        ego = torch.stack(
            [
                ego[:, 0].clamp_min(0.0),
                ego[:, 1],
                ego[:, 2],
                ego[:, 3],
            ],
            dim=-1,
        )
        return ego


class ConfidenceHead(nn.Module):
    def __init__(self, cfg: AtlasHAConfig):
        super().__init__()
        self.road = _mlp(cfg.hidden_dim, cfg.hidden_dim, 1)
        self.lane = _mlp(cfg.hidden_dim, cfg.hidden_dim, 1)
        self.takeover = _mlp(cfg.hidden_dim, cfg.hidden_dim, 1)
        self.scene_type = _mlp(cfg.hidden_dim, cfg.hidden_dim, 4)

    def forward(self, head_context: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "road_followable_conf": torch.sigmoid(self.road(head_context)),
            "lane_tracking_conf": torch.sigmoid(self.lane(head_context)),
            "takeover_required_logit": self.takeover(head_context),
            "scene_type_logits": self.scene_type(head_context),
        }


class BEVHighwayHead(nn.Module):
    def __init__(self, cfg: AtlasHAConfig):
        super().__init__()
        self.cfg = cfg
        self.net = _mlp(
            cfg.hidden_dim,
            cfg.hidden_dim,
            cfg.bev_channels * cfg.bev_h * cfg.bev_w,
        )

    def forward(self, head_context: torch.Tensor) -> torch.Tensor:
        batch_size = head_context.shape[0]
        return self.net(head_context).view(
            batch_size,
            self.cfg.bev_channels,
            self.cfg.bev_h,
            self.cfg.bev_w,
        )


class AtlasHAHeads(nn.Module):
    def __init__(self, cfg: AtlasHAConfig):
        super().__init__()
        self.cfg = cfg
        self.trajectory = TrajectoryHead(cfg)
        self.lane_road = LaneRoadHead(cfg)
        self.lead_adjacent = LeadAdjacentHead(cfg)
        self.ego = EgoHead(cfg)
        self.confidence = ConfidenceHead(cfg)
        self.bev = BEVHighwayHead(cfg) if cfg.enable_bev_aux else None

    def forward(self, head_context: torch.Tensor) -> dict[str, torch.Tensor | None]:
        traj_candidates, candidate_logits = self.trajectory(head_context)
        outputs: dict[str, torch.Tensor | None] = {
            "traj_candidates": traj_candidates,
            "candidate_logits": candidate_logits,
            **self.lane_road(head_context),
            **self.lead_adjacent(head_context),
            "ego_kinematics": self.ego(head_context),
            **self.confidence(head_context),
            "bev_highway": None,
        }
        if self.bev is not None:
            outputs["bev_highway"] = self.bev(head_context)
        return outputs
