from __future__ import annotations

import torch
import torch.nn as nn

from ..config import AtlasConfig
from ..state import AtlasWorldMemoryView
from ..utils import (
    CrossAttentionBlock,
    KinematicIntegrator,
    MLP,
    SelfAttentionBlock,
    assert_rank,
)


class ProposalDecoderBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.self_block = SelfAttentionBlock(dim, heads, dropout)
        self.cross_block = CrossAttentionBlock(dim, heads, dropout)

    def forward(self, q: torch.Tensor, world: torch.Tensor) -> torch.Tensor:
        q = self.self_block(q)
        q = self.cross_block(q, world)
        return q


class PlannerHead(nn.Module):
    def __init__(self, cfg: AtlasConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.hidden_dim
        K = cfg.planner.proposals
        self.proposal_queries = nn.Parameter(torch.randn(1, K, D) * 0.02)
        self.decoder = nn.ModuleList(
            [
            ProposalDecoderBlock(D, cfg.planner.num_heads, cfg.vision.dropout)
                for _ in range(cfg.planner.decoder_blocks)
            ]
        )
        S = cfg.planner.control_steps
        self.curv_head = nn.Linear(D, S)
        self.speed_head = nn.Linear(D, S)
        self.stop_head = nn.Linear(D, 1)
        self.reward_head = MLP(D * 2, D * 2, len(cfg.planner.reward_terms), dropout=cfg.vision.dropout)
        self.score_head = nn.Linear(len(cfg.planner.reward_terms), 1)
        self.future_dyn_head = nn.Linear(D, cfg.world.dynamic_slots * D)
        self.future_ego_head = nn.Linear(D, cfg.world.ego_tokens * D)
        self.integrator = KinematicIntegrator(
            control_dt=cfg.planner.control_dt,
            waypoint_dt=cfg.planner.waypoint_dt,
            control_steps=cfg.planner.control_steps,
        )

    def forward(
        self,
        world_view: AtlasWorldMemoryView,
        kinematics: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        kinematics: [B, 4] -> speed, a_long, yaw_rate, slip_proxy
        """
        assert_rank(kinematics, 2, "kinematics")
        world = world_view.persistent_tokens
        b, _, d = world.shape
        q = self.proposal_queries.expand(b, -1, -1)
        for block in self.decoder:
            q = block(q, world)

        curvature = 0.15 * torch.tanh(self.curv_head(q))
        raw_speed = self.speed_head(q)
        base_speed = kinematics[:, 0:1, None]
        speed = torch.relu(raw_speed + base_speed)
        stop_logit = self.stop_head(q)

        traj = self.integrator(curvature, speed, init_speed=kinematics[:, 0])
        world_summary = world.mean(dim=1, keepdim=True).expand_as(q)
        reward_in = torch.cat([q, world_summary], dim=-1)
        reward_terms = self.reward_head(reward_in)
        score = self.score_head(reward_terms).squeeze(-1)

        H = self.cfg.planner.evaluator_rollout_steps
        dyn_base = world_view.dynamic_slots[:, None, None].expand(-1, q.shape[1], H, -1, -1)
        ego_base = world_view.ego_tokens[:, None, None].expand(-1, q.shape[1], H, -1, -1)
        dyn_delta = self.future_dyn_head(q).reshape(b, q.shape[1], self.cfg.world.dynamic_slots, d)
        ego_delta = self.future_ego_head(q).reshape(b, q.shape[1], self.cfg.world.ego_tokens, d)
        step_scale = torch.linspace(0.2, 1.0, H, device=q.device, dtype=q.dtype)[None, None, :, None, None]
        future_dyn = dyn_base + step_scale * dyn_delta[:, :, None]
        future_ego = ego_base + step_scale * ego_delta[:, :, None]

        best_idx = score.argmax(dim=1)
        best_traj = traj[torch.arange(b, device=traj.device), best_idx]
        return {
            "proposal_embed": q,
            "curvature": curvature,
            "speed": speed,
            "stop_logit": stop_logit,
            "traj": traj,
            "reward_terms": reward_terms,
            "score": score,
            "future_dyn": future_dyn,
            "future_ego": future_ego,
            "best_idx": best_idx,
            "best_traj": best_traj,
        }


AtlasPlannerHead = PlannerHead
