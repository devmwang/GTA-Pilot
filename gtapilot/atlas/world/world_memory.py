from __future__ import annotations

import torch
import torch.nn as nn

from ..config import AtlasConfig
from ..state import AtlasState, AtlasWorldMemoryView
from ..utils import WorldUpdateBlock, assert_rank
from .warp import warp_static_grid
from .write_gate import StaticWriteGate


class WorldMemory(nn.Module):
    def __init__(self, cfg: AtlasConfig):
        super().__init__()
        self.cfg = cfg
        d_model = cfg.hidden_dim
        heads = cfg.world.num_heads
        self.blocks = nn.ModuleList(
            [
                WorldUpdateBlock(d_model, heads, cfg.vision.dropout)
                for _ in range(cfg.world.world_blocks)
            ]
        )
        self.static_gate = StaticWriteGate(d_model=d_model)
        self.token_norm = nn.LayerNorm(d_model)
        self.dynamic_decay = nn.Sequential(
            nn.Linear(d_model + 1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )
        self.speculative_decay = nn.Sequential(
            nn.Linear(d_model + 1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )
        self.speculative_seed = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.dynamic_refresh = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.speculative_refresh = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.alive_gate_temperature = 0.25

    @staticmethod
    def _expand_alive(slots: torch.Tensor, alive: torch.Tensor) -> torch.Tensor:
        return alive.to(dtype=slots.dtype, device=slots.device).unsqueeze(-1)

    @staticmethod
    def _straight_through_gate(
        prob: torch.Tensor,
        *,
        threshold: float,
    ) -> torch.Tensor:
        hard = (prob >= threshold).to(dtype=prob.dtype)
        return hard + prob - prob.detach()

    def forward(
        self,
        state: AtlasState,
        obs_tokens: torch.Tensor,
        temporal_tokens: torch.Tensor,
        pose_delta: torch.Tensor,
        pose_uncertainty: torch.Tensor,
        ego_tokens_new: torch.Tensor,
        route_tokens: torch.Tensor,
        reasoner_tokens: torch.Tensor,
        step_dt_s: torch.Tensor,
    ) -> tuple[AtlasWorldMemoryView, dict[str, torch.Tensor]]:
        assert_rank(obs_tokens, 3, "obs_tokens")
        assert_rank(temporal_tokens, 3, "temporal_tokens")
        assert_rank(pose_delta, 2, "pose_delta")
        assert_rank(pose_uncertainty, 2, "pose_uncertainty")
        assert_rank(step_dt_s, 2, "step_dt_s")
        warped = warp_static_grid(
            state.static_grid,
            pose_delta,
            cell_x_m=self.cfg.world.cell_x_m,
            cell_y_m=self.cfg.world.cell_y_m,
        )
        static_tokens = warped.flatten(2).transpose(1, 2)
        temporal_summary = temporal_tokens.mean(dim=1, keepdim=True)
        route_summary = route_tokens.mean(dim=1, keepdim=True)
        obs_summary = obs_tokens.mean(dim=1, keepdim=True)
        uncertainty_bias = pose_uncertainty.mean(dim=-1, keepdim=True)
        dynamic_slots = state.dynamic_slots * self._expand_alive(
            state.dynamic_slots,
            state.dynamic_slot_alive,
        )
        speculative_slots = state.speculative_slots * self._expand_alive(
            state.speculative_slots,
            state.speculative_slot_alive,
        )
        dyn_gate = self.dynamic_decay(
            torch.cat(
                [
                    dynamic_slots,
                    uncertainty_bias[:, None].expand(
                        -1, dynamic_slots.shape[1], -1
                    ),
                ],
                dim=-1,
            )
        )
        spec_gate = self.speculative_decay(
            torch.cat(
                [
                    speculative_slots,
                    uncertainty_bias[:, None].expand(
                        -1, speculative_slots.shape[1], -1
                    ),
                ],
                dim=-1,
            )
        )
        propagated_dynamic = (1.0 - dyn_gate) * dynamic_slots + dyn_gate * (
            dynamic_slots + temporal_summary
        )
        propagated_speculative = (1.0 - spec_gate) * speculative_slots + spec_gate * (
            speculative_slots
            + self.speculative_seed(
                torch.cat(
                    [
                        temporal_summary.expand(
                            -1, speculative_slots.shape[1], -1
                        ),
                        route_summary.expand(
                            -1, speculative_slots.shape[1], -1
                        ),
                    ],
                    dim=-1,
                )
            )
        )
        world_tokens = torch.cat(
            [
                static_tokens,
                propagated_dynamic,
                propagated_speculative,
                state.lane_slots,
                state.map_elem_slots,
                ego_tokens_new,
                route_tokens,
                reasoner_tokens,
            ],
            dim=1,
        )
        obs_context = torch.cat([obs_tokens, temporal_tokens], dim=1)

        for block in self.blocks:
            world_tokens = block(world_tokens, obs_context)
        world_tokens = self.token_norm(world_tokens)

        n_static = self.cfg.world.static_grid_h * self.cfg.world.static_grid_w
        n_dyn = self.cfg.world.dynamic_slots
        n_spec = self.cfg.world.speculative_slots
        n_lane = self.cfg.world.lane_slots
        n_map = self.cfg.world.map_elem_slots
        n_ego = self.cfg.world.ego_tokens
        n_route = self.cfg.world.route_tokens
        n_reason = self.cfg.world.reasoner_tokens

        s0 = 0
        s1 = s0 + n_static
        s2 = s1 + n_dyn
        s3 = s2 + n_spec
        s4 = s3 + n_lane
        s5 = s4 + n_map
        s6 = s5 + n_ego
        s7 = s6 + n_route
        s8 = s7 + n_reason

        updated_static = world_tokens[:, s0:s1]
        updated_dynamic = world_tokens[:, s1:s2]
        updated_speculative = world_tokens[:, s2:s3]
        updated_lane = world_tokens[:, s3:s4]
        updated_map = world_tokens[:, s4:s5]
        updated_ego = world_tokens[:, s5:s6]
        updated_route = world_tokens[:, s6:s7]
        updated_reason = world_tokens[:, s7:s8]

        static_obs_summary = obs_summary.expand(-1, n_static, -1)
        gate = self.static_gate(
            static_tokens=static_tokens,
            updated_static=updated_static,
            obs_summary=static_obs_summary,
            pose_uncertainty=pose_uncertainty,
        )
        if self.training and self.cfg.world.static_write_dropout > 0.0:
            keep = (
                torch.rand_like(gate)
                >= float(self.cfg.world.static_write_dropout)
            ).to(gate.dtype)
            gate = gate * keep
        fused_static = (1.0 - gate) * static_tokens + gate * updated_static
        fused_static_grid = fused_static.transpose(1, 2).reshape(
            state.static_grid.shape[0],
            self.cfg.hidden_dim,
            self.cfg.world.static_grid_h,
            self.cfg.world.static_grid_w,
        )

        dynamic_refresh = torch.sigmoid(
            self.dynamic_refresh(
                torch.cat(
                    [
                        updated_dynamic,
                        obs_summary.expand(-1, n_dyn, -1),
                    ],
                    dim=-1,
                )
            )
        ).squeeze(-1)
        speculative_refresh = torch.sigmoid(
            self.speculative_refresh(
                torch.cat(
                    [
                        updated_speculative,
                        (temporal_summary + route_summary).expand(-1, n_spec, -1),
                    ],
                    dim=-1,
                )
            )
        ).squeeze(-1)

        dt_dynamic = step_dt_s.expand(-1, n_dyn)
        dt_speculative = step_dt_s.expand(-1, n_spec)
        actor_memory_survival_s = float(self.cfg.world.actor_memory_survival_s)
        stale_dynamic_age = state.dynamic_slot_age_s + dt_dynamic
        stale_speculative_age = state.speculative_slot_age_s + dt_speculative
        prev_dynamic_alive = state.dynamic_slot_alive.to(
            dtype=updated_dynamic.dtype,
            device=updated_dynamic.device,
        ).clamp_(0.0, 1.0)
        prev_speculative_alive = state.speculative_slot_alive.to(
            dtype=updated_speculative.dtype,
            device=updated_speculative.device,
        ).clamp_(0.0, 1.0)
        if self.training:
            dynamic_refresh_gate = self._straight_through_gate(
                dynamic_refresh,
                threshold=0.5,
            )
            speculative_refresh_gate = self._straight_through_gate(
                speculative_refresh,
                threshold=0.55,
            )
            dynamic_age_s = stale_dynamic_age * (1.0 - dynamic_refresh_gate)
            speculative_age_s = stale_speculative_age * (1.0 - speculative_refresh_gate)
            dynamic_survival_prob = torch.sigmoid(
                (actor_memory_survival_s - dynamic_age_s)
                / self.alive_gate_temperature
            )
            speculative_survival_prob = torch.sigmoid(
                (actor_memory_survival_s - speculative_age_s)
                / self.alive_gate_temperature
            )
            dynamic_survival_gate = self._straight_through_gate(
                dynamic_survival_prob,
                threshold=0.5,
            )
            speculative_survival_gate = self._straight_through_gate(
                speculative_survival_prob,
                threshold=0.5,
            )
            dynamic_alive = torch.maximum(
                prev_dynamic_alive,
                dynamic_refresh_gate,
            ) * dynamic_survival_gate
            speculative_alive = torch.maximum(
                prev_speculative_alive,
                speculative_refresh_gate,
            ) * speculative_survival_gate
        else:
            dynamic_refreshed = dynamic_refresh >= 0.5
            speculative_refreshed = speculative_refresh >= 0.55
            dynamic_age_s = torch.where(
                dynamic_refreshed,
                torch.zeros_like(state.dynamic_slot_age_s),
                stale_dynamic_age,
            )
            speculative_age_s = torch.where(
                speculative_refreshed,
                torch.zeros_like(state.speculative_slot_age_s),
                stale_speculative_age,
            )
            dynamic_alive = (
                (
                    prev_dynamic_alive >= 0.5
                )
                | dynamic_refreshed
            ).to(updated_dynamic.dtype) * (
                dynamic_age_s <= actor_memory_survival_s
            ).to(updated_dynamic.dtype)
            speculative_alive = (
                (
                    prev_speculative_alive >= 0.5
                )
                | speculative_refreshed
            ).to(updated_speculative.dtype) * (
                speculative_age_s <= actor_memory_survival_s
            ).to(updated_speculative.dtype)

        dynamic_alive = dynamic_alive.clamp_(0.0, 1.0)
        speculative_alive = speculative_alive.clamp_(0.0, 1.0)
        dynamic_age_s = torch.where(
            dynamic_alive > 1e-4,
            dynamic_age_s,
            torch.zeros_like(dynamic_age_s),
        )
        speculative_age_s = torch.where(
            speculative_alive > 1e-4,
            speculative_age_s,
            torch.zeros_like(speculative_age_s),
        )
        updated_dynamic = updated_dynamic * self._expand_alive(
            updated_dynamic,
            dynamic_alive,
        )
        updated_speculative = updated_speculative * self._expand_alive(
            updated_speculative,
            speculative_alive,
        )

        world_view = AtlasWorldMemoryView(
            static_grid=fused_static_grid,
            dynamic_slots=updated_dynamic,
            speculative_slots=updated_speculative,
            dynamic_slot_alive=dynamic_alive,
            speculative_slot_alive=speculative_alive,
            lane_slots=updated_lane,
            map_elem_slots=updated_map,
            ego_tokens=updated_ego,
            route_tokens=updated_route,
            reasoner_tokens=updated_reason,
            persistent_tokens=torch.cat(
                [
                    fused_static,
                    updated_dynamic,
                    updated_speculative,
                    updated_lane,
                    updated_map,
                    updated_ego,
                    updated_route,
                    updated_reason,
                ],
                dim=1,
            ),
        )
        return world_view, {
            "dynamic_slot_age_s": dynamic_age_s,
            "speculative_slot_age_s": speculative_age_s,
            "dynamic_slot_alive": dynamic_alive,
            "speculative_slot_alive": speculative_alive,
        }
