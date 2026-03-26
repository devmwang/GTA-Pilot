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

    def forward(
        self,
        state: AtlasState,
        obs_tokens: torch.Tensor,
        pose_delta: torch.Tensor,
        pose_uncertainty: torch.Tensor,
        ego_tokens_new: torch.Tensor,
        route_tokens: torch.Tensor,
        reasoner_tokens: torch.Tensor,
    ) -> AtlasWorldMemoryView:
        assert_rank(obs_tokens, 3, "obs_tokens")
        assert_rank(pose_delta, 2, "pose_delta")
        assert_rank(pose_uncertainty, 2, "pose_uncertainty")
        warped = warp_static_grid(
            state.static_grid,
            pose_delta,
            cell_x_m=self.cfg.world.cell_x_m,
            cell_y_m=self.cfg.world.cell_y_m,
        )
        static_tokens = warped.flatten(2).transpose(1, 2)
        world_tokens = torch.cat(
            [
                static_tokens,
                state.dynamic_slots,
                state.lane_slots,
                state.map_elem_slots,
                ego_tokens_new,
                route_tokens,
                reasoner_tokens,
            ],
            dim=1,
        )

        for block in self.blocks:
            world_tokens = block(world_tokens, obs_tokens)
        world_tokens = self.token_norm(world_tokens)

        n_static = self.cfg.world.static_grid_h * self.cfg.world.static_grid_w
        n_dyn = self.cfg.world.dynamic_slots
        n_lane = self.cfg.world.lane_slots
        n_map = self.cfg.world.map_elem_slots
        n_ego = self.cfg.world.ego_tokens
        n_route = self.cfg.world.route_tokens
        n_reason = self.cfg.world.reasoner_tokens

        s0 = 0
        s1 = s0 + n_static
        s2 = s1 + n_dyn
        s3 = s2 + n_lane
        s4 = s3 + n_map
        s5 = s4 + n_ego
        s6 = s5 + n_route
        s7 = s6 + n_reason

        updated_static = world_tokens[:, s0:s1]
        updated_dynamic = world_tokens[:, s1:s2]
        updated_lane = world_tokens[:, s2:s3]
        updated_map = world_tokens[:, s3:s4]
        updated_ego = world_tokens[:, s4:s5]
        updated_route = world_tokens[:, s5:s6]
        updated_reason = world_tokens[:, s6:s7]

        obs_summary = obs_tokens.mean(dim=1, keepdim=True).expand(-1, n_static, -1)
        gate = self.static_gate(
            static_tokens=static_tokens,
            updated_static=updated_static,
            obs_summary=obs_summary,
            pose_uncertainty=pose_uncertainty,
        )
        fused_static = (1.0 - gate) * static_tokens + gate * updated_static
        fused_static_grid = fused_static.transpose(1, 2).reshape(
            state.static_grid.shape[0],
            self.cfg.hidden_dim,
            self.cfg.world.static_grid_h,
            self.cfg.world.static_grid_w,
        )

        return AtlasWorldMemoryView(
            static_grid=fused_static_grid,
            dynamic_slots=updated_dynamic,
            lane_slots=updated_lane,
            map_elem_slots=updated_map,
            ego_tokens=updated_ego,
            route_tokens=updated_route,
            reasoner_tokens=updated_reason,
            persistent_tokens=torch.cat(
                [
                    fused_static,
                    updated_dynamic,
                    updated_lane,
                    updated_map,
                    updated_ego,
                    updated_route,
                    updated_reason,
                ],
                dim=1,
            ),
        )
