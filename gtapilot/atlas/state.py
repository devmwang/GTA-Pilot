from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import AtlasConfig


@dataclass
class AtlasWorldMemoryView:
    static_grid: torch.Tensor
    dynamic_slots: torch.Tensor
    lane_slots: torch.Tensor
    map_elem_slots: torch.Tensor
    ego_tokens: torch.Tensor
    route_tokens: torch.Tensor
    reasoner_tokens: torch.Tensor
    persistent_tokens: torch.Tensor


@dataclass
class AtlasState:
    cam_token_cache: torch.Tensor
    frame_sum_cache: torch.Tensor
    action_buffer: torch.Tensor
    dt_buffer: torch.Tensor

    static_grid: torch.Tensor
    dynamic_slots: torch.Tensor
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
        t_minus_1 = cfg.temporal.num_frames - 1
        return cls(
            cam_token_cache=torch.zeros(
                batch_size,
                t_minus_1,
                cfg.vision.cam_tokens_per_frame,
                d_model,
                device=device,
                dtype=dtype,
            ),
            frame_sum_cache=torch.zeros(
                batch_size, t_minus_1, d_model, device=device, dtype=dtype
            ),
            action_buffer=torch.zeros(
                batch_size,
                cfg.action.history_len,
                cfg.action.action_dim,
                device=device,
                dtype=dtype,
            ),
            dt_buffer=torch.zeros(
                batch_size, cfg.action.history_len, 1, device=device, dtype=dtype
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
                batch_size, cfg.world.ego_tokens, d_model, device=device, dtype=dtype
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
        static_tokens = self.static_grid.flatten(2).transpose(1, 2)
        persistent = torch.cat(
            [
                static_tokens,
                self.dynamic_slots,
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
            dynamic_slots=self.dynamic_slots,
            lane_slots=self.lane_slots,
            map_elem_slots=self.map_elem_slots,
            ego_tokens=self.ego_tokens,
            route_tokens=self.route_tokens,
            reasoner_tokens=self.reasoner_tokens,
            persistent_tokens=persistent,
        )

WorldMemoryView = AtlasWorldMemoryView
