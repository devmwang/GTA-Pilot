from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import AtlasConfig


@dataclass
class AtlasWorldMemoryView:
    static_grid: torch.Tensor
    dynamic_slots: torch.Tensor
    speculative_slots: torch.Tensor
    dynamic_slot_alive: torch.Tensor
    speculative_slot_alive: torch.Tensor
    lane_slots: torch.Tensor
    map_elem_slots: torch.Tensor
    ego_tokens: torch.Tensor
    route_tokens: torch.Tensor
    reasoner_tokens: torch.Tensor
    persistent_tokens: torch.Tensor


@dataclass
class AtlasState:
    recent_cam_cache: torch.Tensor
    older_cam_cache: torch.Tensor
    mid_summary_cache: torch.Tensor
    recent_dt_cache: torch.Tensor
    older_dt_cache: torch.Tensor
    mid_dt_cache: torch.Tensor
    recent_valid: torch.Tensor
    older_valid: torch.Tensor
    mid_valid: torch.Tensor
    action_buffer: torch.Tensor
    dt_buffer: torch.Tensor

    static_grid: torch.Tensor
    dynamic_slots: torch.Tensor
    speculative_slots: torch.Tensor
    dynamic_slot_age_s: torch.Tensor
    speculative_slot_age_s: torch.Tensor
    dynamic_slot_alive: torch.Tensor
    speculative_slot_alive: torch.Tensor
    lane_slots: torch.Tensor
    map_elem_slots: torch.Tensor
    ego_tokens: torch.Tensor
    route_tokens: torch.Tensor
    reasoner_tokens: torch.Tensor

    ego_filter_hidden: torch.Tensor
    pose_belief: torch.Tensor
    reasoner_ttl: torch.Tensor
    step_index: int = 0

    @classmethod
    def init_empty(
        cls,
        cfg: AtlasConfig,
        batch_size: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "AtlasState":
        device = torch.device(device)
        d_model = cfg.hidden_dim
        return cls(
            recent_cam_cache=torch.zeros(
                batch_size,
                cfg.temporal.recent_cache_frames,
                cfg.temporal.recent_cam_tokens,
                d_model,
                device=device,
                dtype=dtype,
            ),
            older_cam_cache=torch.zeros(
                batch_size,
                cfg.temporal.older_compressed_frames,
                cfg.temporal.older_compressed_tokens,
                d_model,
                device=device,
                dtype=dtype,
            ),
            mid_summary_cache=torch.zeros(
                batch_size,
                cfg.temporal.mid_summary_frames,
                cfg.temporal.mid_summary_tokens,
                d_model,
                device=device,
                dtype=dtype,
            ),
            recent_dt_cache=torch.zeros(
                batch_size,
                cfg.temporal.recent_cache_frames,
                1,
                device=device,
                dtype=dtype,
            ),
            older_dt_cache=torch.zeros(
                batch_size,
                cfg.temporal.older_compressed_frames,
                1,
                device=device,
                dtype=dtype,
            ),
            mid_dt_cache=torch.zeros(
                batch_size,
                cfg.temporal.mid_summary_frames,
                1,
                device=device,
                dtype=dtype,
            ),
            recent_valid=torch.zeros(
                batch_size,
                cfg.temporal.recent_cache_frames,
                device=device,
                dtype=torch.bool,
            ),
            older_valid=torch.zeros(
                batch_size,
                cfg.temporal.older_compressed_frames,
                device=device,
                dtype=torch.bool,
            ),
            mid_valid=torch.zeros(
                batch_size,
                cfg.temporal.mid_summary_frames,
                device=device,
                dtype=torch.bool,
            ),
            action_buffer=torch.zeros(
                batch_size,
                cfg.action.history_len,
                cfg.action.action_dim,
                device=device,
                dtype=dtype,
            ),
            dt_buffer=torch.zeros(
                batch_size,
                cfg.action.history_len,
                1,
                device=device,
                dtype=dtype,
            ),
            static_grid=torch.zeros(
                batch_size,
                d_model,
                cfg.world.static_grid_h,
                cfg.world.static_grid_w,
                device=device,
                dtype=dtype,
            ),
            dynamic_slots=torch.zeros(
                batch_size,
                cfg.world.dynamic_slots,
                d_model,
                device=device,
                dtype=dtype,
            ),
            speculative_slots=torch.zeros(
                batch_size,
                cfg.world.speculative_slots,
                d_model,
                device=device,
                dtype=dtype,
            ),
            dynamic_slot_age_s=torch.zeros(
                batch_size,
                cfg.world.dynamic_slots,
                device=device,
                dtype=dtype,
            ),
            speculative_slot_age_s=torch.zeros(
                batch_size,
                cfg.world.speculative_slots,
                device=device,
                dtype=dtype,
            ),
            dynamic_slot_alive=torch.zeros(
                batch_size,
                cfg.world.dynamic_slots,
                device=device,
                dtype=torch.bool,
            ),
            speculative_slot_alive=torch.zeros(
                batch_size,
                cfg.world.speculative_slots,
                device=device,
                dtype=torch.bool,
            ),
            lane_slots=torch.zeros(
                batch_size,
                cfg.world.lane_slots,
                d_model,
                device=device,
                dtype=dtype,
            ),
            map_elem_slots=torch.zeros(
                batch_size,
                cfg.world.map_elem_slots,
                d_model,
                device=device,
                dtype=dtype,
            ),
            ego_tokens=torch.zeros(
                batch_size,
                cfg.world.ego_tokens,
                d_model,
                device=device,
                dtype=dtype,
            ),
            route_tokens=torch.zeros(
                batch_size,
                cfg.world.route_tokens,
                d_model,
                device=device,
                dtype=dtype,
            ),
            reasoner_tokens=torch.zeros(
                batch_size,
                cfg.world.reasoner_tokens,
                d_model,
                device=device,
                dtype=dtype,
            ),
            ego_filter_hidden=torch.zeros(
                cfg.ego.num_layers,
                batch_size,
                cfg.ego.hidden_size,
                device=device,
                dtype=dtype,
            ),
            pose_belief=torch.zeros(batch_size, 10, device=device, dtype=dtype),
            reasoner_ttl=torch.zeros(batch_size, 1, device=device, dtype=dtype),
            step_index=0,
        )

    def world_view(self) -> AtlasWorldMemoryView:
        dynamic_slots = self.dynamic_slots * self.dynamic_slot_alive.unsqueeze(-1).to(
            self.dynamic_slots.dtype
        )
        speculative_slots = self.speculative_slots * self.speculative_slot_alive.unsqueeze(
            -1
        ).to(self.speculative_slots.dtype)
        static_tokens = self.static_grid.flatten(2).transpose(1, 2)
        persistent = torch.cat(
            [
                static_tokens,
                dynamic_slots,
                speculative_slots,
                self.lane_slots,
                self.map_elem_slots,
                self.ego_tokens,
                self.route_tokens,
                self.reasoner_tokens,
            ],
            dim=1,
        )
        return AtlasWorldMemoryView(
            static_grid=self.static_grid,
            dynamic_slots=dynamic_slots,
            speculative_slots=speculative_slots,
            dynamic_slot_alive=self.dynamic_slot_alive,
            speculative_slot_alive=self.speculative_slot_alive,
            lane_slots=self.lane_slots,
            map_elem_slots=self.map_elem_slots,
            ego_tokens=self.ego_tokens,
            route_tokens=self.route_tokens,
            reasoner_tokens=self.reasoner_tokens,
            persistent_tokens=persistent,
        )
