from __future__ import annotations

import torch
import torch.nn as nn

from ..config import AtlasConfig
from ..state import AtlasWorldMemoryView
from ..utils import (
    CrossAttentionBlock,
    KinematicIntegrator,
    LearnedQueryPool,
    MLP,
    SelfAttentionBlock,
    assert_rank,
)


def _masked_slot_mean(slots: torch.Tensor, alive: torch.Tensor) -> torch.Tensor:
    weights = alive.to(dtype=slots.dtype, device=slots.device).clamp(0.0, 1.0).unsqueeze(-1)
    denom = weights.sum(dim=1).clamp(min=1.0)
    return (slots * weights).sum(dim=1) / denom


class ProposalDecoderBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.self_block = SelfAttentionBlock(dim, heads, dropout)
        self.cross_block = CrossAttentionBlock(dim, heads, dropout)

    def forward(self, q: torch.Tensor, world: torch.Tensor) -> torch.Tensor:
        q = self.self_block(q)
        q = self.cross_block(q, world)
        return q


class RiskContextDecoder(nn.Module):
    def __init__(self, cfg: AtlasConfig):
        super().__init__()
        self.cfg = cfg
        d_model = cfg.hidden_dim
        mid_ch = 96
        out_h = cfg.bev_lite.out_h
        out_w = cfg.bev_lite.out_w
        self.static_proj = nn.Sequential(
            nn.Conv2d(d_model, mid_ch, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Upsample(size=(out_h, out_w), mode="bilinear", align_corners=False),
            nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.dynamic_bias = nn.Linear(d_model, mid_ch)
        self.speculative_bias = nn.Linear(d_model, mid_ch)
        self.flow_head = nn.Conv2d(mid_ch, 2, kernel_size=1)
        self.occl_risk_head = nn.Conv2d(mid_ch, 2, kernel_size=1)
        self.provenance_head = nn.Conv2d(
            mid_ch, cfg.bev_lite.provenance_classes, kernel_size=1
        )

    def forward(self, world_view: AtlasWorldMemoryView) -> dict[str, torch.Tensor]:
        bev = self.static_proj(world_view.static_grid)
        dynamic_bias = self.dynamic_bias(
            _masked_slot_mean(
                world_view.dynamic_slots,
                world_view.dynamic_slot_alive,
            )
        )[
            :, :, None, None
        ]
        speculative_bias = self.speculative_bias(
            _masked_slot_mean(
                world_view.speculative_slots,
                world_view.speculative_slot_alive,
            )
        )[:, :, None, None]
        fused = bev + dynamic_bias + speculative_bias
        return {
            "dyn_flow_bev": self.flow_head(fused),
            "occl_risk_bev": self.occl_risk_head(fused),
            "provenance": self.provenance_head(fused),
        }


class PlannerHead(nn.Module):
    def __init__(self, cfg: AtlasConfig):
        super().__init__()
        self.cfg = cfg
        d_model = cfg.hidden_dim
        proposals = cfg.planner.proposals
        self.proposal_queries = nn.Parameter(torch.randn(1, proposals, d_model) * 0.02)
        self.decoder = nn.ModuleList(
            [
                ProposalDecoderBlock(d_model, cfg.planner.num_heads, cfg.vision.dropout)
                for _ in range(cfg.planner.decoder_blocks)
            ]
        )
        self.risk_pool = LearnedQueryPool(
            num_queries=3,
            dim=d_model,
            heads=max(1, cfg.planner.num_heads),
            dropout=cfg.vision.dropout,
        )
        self.flow_token_proj = nn.Linear(2, d_model)
        self.occl_token_proj = nn.Linear(2, d_model)
        self.provenance_token_proj = nn.Linear(
            cfg.bev_lite.provenance_classes,
            d_model,
        )
        control_steps = cfg.planner.control_steps
        self.curv_head = nn.Linear(d_model, control_steps)
        self.speed_head = nn.Linear(d_model, control_steps)
        self.stop_head = nn.Linear(d_model, 1)
        self.reward_head = MLP(
            d_model * 3,
            d_model * 2,
            len(cfg.planner.reward_terms),
            dropout=cfg.vision.dropout,
        )
        self.score_head = nn.Linear(len(cfg.planner.reward_terms), 1)
        self.future_dyn_head = nn.Linear(d_model, cfg.world.dynamic_slots * d_model)
        self.future_spec_head = nn.Linear(
            d_model, cfg.world.speculative_slots * d_model
        )
        self.future_ego_head = nn.Linear(d_model, cfg.world.ego_tokens * d_model)
        self.integrator = KinematicIntegrator(
            control_dt=cfg.planner.control_dt,
            waypoint_dt=cfg.planner.waypoint_dt,
            control_steps=cfg.planner.control_steps,
        )

    def _risk_tokens(
        self,
        risk_context: dict[str, torch.Tensor],
        world_view: AtlasWorldMemoryView,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flow = self.flow_token_proj(risk_context["dyn_flow_bev"].flatten(2).transpose(1, 2))
        occl = self.occl_token_proj(
            risk_context["occl_risk_bev"].flatten(2).transpose(1, 2)
        )
        provenance = self.provenance_token_proj(
            risk_context["provenance"].flatten(2).transpose(1, 2)
        )
        pooled = self.risk_pool(
            torch.cat(
                [
                    flow,
                    occl,
                    provenance,
                    world_view.dynamic_slots,
                    world_view.speculative_slots,
                ],
                dim=1,
            )
        )
        occl_scalar = risk_context["occl_risk_bev"].sigmoid().mean(dim=(1, 2, 3))
        speculative_scalar = (
            world_view.speculative_slots.norm(dim=-1)
            * world_view.speculative_slot_alive.to(world_view.speculative_slots.dtype)
        ).sum(dim=1) / world_view.speculative_slot_alive.sum(dim=1).clamp(min=1)
        hidden_penalty = occl_scalar + 0.1 * speculative_scalar
        return pooled, hidden_penalty

    def forward(
        self,
        world_view: AtlasWorldMemoryView,
        kinematics: torch.Tensor,
        risk_context: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        kinematics: [B, 4] -> speed, a_long, yaw_rate, slip_proxy
        """
        assert_rank(kinematics, 2, "kinematics")
        world = world_view.persistent_tokens
        risk_tokens, hidden_penalty = self._risk_tokens(risk_context, world_view)
        world_with_risk = torch.cat([world, risk_tokens], dim=1)

        batch_size, _, d_model = world_with_risk.shape
        query = self.proposal_queries.expand(batch_size, -1, -1)
        for block in self.decoder:
            query = block(query, world_with_risk)

        curvature = 0.15 * torch.tanh(self.curv_head(query))
        raw_speed = self.speed_head(query)
        base_speed = kinematics[:, 0:1, None]
        speed = torch.relu(raw_speed + base_speed)
        stop_logit = self.stop_head(query)

        traj = self.integrator(curvature, speed, init_speed=kinematics[:, 0])
        world_summary = world_with_risk.mean(dim=1, keepdim=True).expand_as(query)
        risk_summary = risk_tokens.mean(dim=1, keepdim=True).expand_as(query)
        reward_in = torch.cat([query, world_summary, risk_summary], dim=-1)
        reward_terms = self.reward_head(reward_in)
        hidden_risk_penalty = hidden_penalty[:, None]
        reward_terms[..., -1] = reward_terms[..., -1] - hidden_risk_penalty
        score = self.score_head(reward_terms).squeeze(-1) - hidden_risk_penalty

        rollout = self.cfg.planner.evaluator_rollout_steps
        dyn_base = world_view.dynamic_slots[:, None, None].expand(
            -1, query.shape[1], rollout, -1, -1
        )
        spec_base = world_view.speculative_slots[:, None, None].expand(
            -1, query.shape[1], rollout, -1, -1
        )
        ego_base = world_view.ego_tokens[:, None, None].expand(
            -1, query.shape[1], rollout, -1, -1
        )
        dyn_delta = self.future_dyn_head(query).reshape(
            batch_size, query.shape[1], self.cfg.world.dynamic_slots, d_model
        )
        spec_delta = self.future_spec_head(query).reshape(
            batch_size, query.shape[1], self.cfg.world.speculative_slots, d_model
        )
        ego_delta = self.future_ego_head(query).reshape(
            batch_size, query.shape[1], self.cfg.world.ego_tokens, d_model
        )
        step_scale = torch.linspace(
            0.2,
            1.0,
            rollout,
            device=query.device,
            dtype=query.dtype,
        )[None, None, :, None, None]
        future_dyn = dyn_base + step_scale * dyn_delta[:, :, None]
        future_spec = spec_base + step_scale * spec_delta[:, :, None]
        future_ego = ego_base + step_scale * ego_delta[:, :, None]

        best_idx = score.argmax(dim=1)
        best_traj = traj[torch.arange(batch_size, device=traj.device), best_idx]
        return {
            "proposal_embed": query,
            "risk_tokens": risk_tokens,
            "curvature": curvature,
            "speed": speed,
            "stop_logit": stop_logit,
            "traj": traj,
            "reward_terms": reward_terms,
            "score": score,
            "hidden_risk_penalty": hidden_penalty,
            "future_dyn": future_dyn,
            "future_spec": future_spec,
            "future_ego": future_ego,
            "best_idx": best_idx,
            "best_traj": best_traj,
        }
