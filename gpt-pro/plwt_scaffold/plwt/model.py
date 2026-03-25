from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .config import PLWTConfig
from .state import PLWTState
from .scheduler import HeadScheduler
from .modules.common import pad_image_to_size, assert_rank
from .modules.visual_dualpath import VisualDualPathTokenizer
from .modules.temporal_mixer import TemporalMixer
from .modules.action_encoder import ActionEncoder
from .modules.ego_filter import EgoFilter
from .modules.geometry_lifter import GeometryLifter
from .modules.observation_pool import ObservationPool
from .modules.adapters import RouteAdapter, ReasonerBridge
from .modules.world_memory import WorldMemory
from .modules.planner import PlannerHead
from .modules.aux_heads import AuxHeads


class PLWT(nn.Module):
    """
    Photon -> Latent World -> Trajectory scaffold.

    This class is intentionally explicit about tensors and state. It is a runnable scaffold,
    not a claim that every internal block is final-production quality.
    """

    def __init__(self, cfg: PLWTConfig):
        super().__init__()
        self.cfg = cfg
        self.visual = VisualDualPathTokenizer(cfg)
        self.temporal = TemporalMixer(cfg)
        self.action_encoder = ActionEncoder(cfg)
        self.ego_filter = EgoFilter(cfg)
        self.geometry = GeometryLifter(cfg)
        self.obs_pool = ObservationPool(cfg)
        self.route_adapter = RouteAdapter(cfg)
        self.reasoner_bridge = ReasonerBridge(cfg)
        self.world = WorldMemory(cfg)
        self.planner = PlannerHead(cfg)
        self.aux = AuxHeads(cfg)
        self.scheduler = HeadScheduler(cfg)

    def init_state(
        self,
        batch_size: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> PLWTState:
        return PLWTState.init_empty(self.cfg, batch_size=batch_size, device=device, dtype=dtype)

    @staticmethod
    def _append_keep_last(x: torch.Tensor, new_x: torch.Tensor, keep: int) -> torch.Tensor:
        cat = torch.cat([x, new_x], dim=1)
        if cat.shape[1] <= keep:
            return cat
        return cat[:, -keep:]

    def step(
        self,
        rgb_t: torch.Tensor,
        action_prev: torch.Tensor,
        dt_t: torch.Tensor,
        state: PLWTState,
        route_polyline: Optional[torch.Tensor] = None,
        nav_cmd: Optional[torch.Tensor] = None,
        reasoner_tok: Optional[torch.Tensor] = None,
        mode: str = "drive",
    ) -> tuple[Dict[str, torch.Tensor], PLWTState]:
        """
        rgb_t:       [B, 3, H, W] where H,W are raw input shape, typically 1080x1920
        action_prev: [B, 6] -> [steer, throttle, brake, handbrake, reverse, pilot_active]
        dt_t:        [B, 1]
        """
        assert_rank(rgb_t, 4, "rgb_t")
        assert_rank(action_prev, 2, "action_prev")
        assert_rank(dt_t, 2, "dt_t")
        B = rgb_t.shape[0]

        rgb_pad = pad_image_to_size(rgb_t, self.cfg.image.padded_height, self.cfg.image.padded_width)
        pyramid, cam_tokens_cur = self.visual(rgb_pad[:, None])  # [B,1,N,D]
        cam_tokens_cur = cam_tokens_cur[:, 0]
        ctx8_now = pyramid["ctx_8x"][:, 0]

        cam_seq = torch.cat([state.cam_token_cache, cam_tokens_cur[:, None]], dim=1)
        dt_seq = torch.cat([state.dt_buffer[:, -(self.cfg.temporal.num_frames - 1):], dt_t[:, None]], dim=1)
        cam_now, frame_sum = self.temporal(cam_seq, dt_seq)

        action_buffer = self._append_keep_last(state.action_buffer, action_prev[:, None], self.cfg.action.history_len)
        dt_buffer = self._append_keep_last(state.dt_buffer, dt_t[:, None], self.cfg.action.history_len)
        act_tokens = self.action_encoder(action_buffer, dt_buffer)

        ego_out = self.ego_filter(cam_now, frame_sum, act_tokens, state.ego_filter_hidden)
        ego_tokens_new = ego_out["ego_tokens"]
        route_tokens = self.route_adapter(route_polyline, nav_cmd, B, rgb_t.device, rgb_t.dtype) if self.cfg.enable_route_tokens else state.route_tokens
        reasoner_tokens, reasoner_ttl = self.reasoner_bridge(reasoner_tok, state.reasoner_tokens, state.reasoner_ttl) if self.cfg.enable_reasoner_tokens else (state.reasoner_tokens, state.reasoner_ttl)

        geom_out = self.geometry(ctx8_now, cam_now, ego_tokens_new)
        obs_tokens = self.obs_pool(cam_now, geom_out["frustum_tokens"], ego_tokens_new, act_tokens)
        world_view = self.world(
            state=state,
            obs_tokens=obs_tokens,
            pose_delta=ego_out["pose_delta"],
            pose_uncertainty=ego_out["logvar_pose"].exp(),
            ego_tokens_new=ego_tokens_new,
            route_tokens=route_tokens,
            reasoner_tokens=reasoner_tokens,
        )
        planner_out = self.planner(world_view, ego_out["kinematics"])

        active = self.scheduler.active_heads(state.step_index, mode=mode)
        aux_out = self.aux(world_view, active_heads=active - {"planner"})

        new_state = PLWTState(
            cam_token_cache=self._append_keep_last(state.cam_token_cache, cam_tokens_cur[:, None], self.cfg.temporal.num_frames - 1),
            frame_sum_cache=self._append_keep_last(state.frame_sum_cache, frame_sum[:, -1:, :], self.cfg.temporal.num_frames - 1),
            action_buffer=action_buffer,
            dt_buffer=dt_buffer,
            static_grid=world_view.static_grid,
            dynamic_slots=world_view.dynamic_slots,
            lane_slots=world_view.lane_slots,
            map_elem_slots=world_view.map_elem_slots,
            ego_tokens=world_view.ego_tokens,
            route_tokens=world_view.route_tokens,
            reasoner_tokens=world_view.reasoner_tokens,
            ego_filter_hidden=ego_out["hidden_next"],
            pose_belief=torch.cat(
                [
                    ego_out["pose_delta"],
                    ego_out["kinematics"],
                    ego_out["logvar_pose"],
                ],
                dim=-1,
            ),
            reasoner_ttl=reasoner_ttl,
            step_index=state.step_index + 1,
        )

        outputs: Dict[str, torch.Tensor] = {
            "rgb_padded": rgb_pad,
            "cam_tokens_cur": cam_tokens_cur,
            "cam_now": cam_now,
            "frame_sum": frame_sum,
            "act_tokens": act_tokens,
            "obs_tokens": obs_tokens,
            "static_grid": world_view.static_grid,
            "persistent_tokens": world_view.persistent_tokens,
            **ego_out,
            **geom_out,
            **planner_out,
            **aux_out,
        }
        return outputs, new_state

    def forward_train(
        self,
        batch: Dict[str, Any],
        init_state: Optional[PLWTState] = None,
        mode: str = "inspect",
    ) -> Dict[str, Any]:
        rgb_seq = batch["rgb_seq"]
        actions_seq = batch["actions_seq"]
        dt_seq = batch["dt_seq"]
        route_polyline = batch.get("route_polyline")
        nav_cmd = batch.get("nav_cmd")
        reasoner_tok = batch.get("reasoner_tok")

        assert_rank(rgb_seq, 5, "rgb_seq")
        assert_rank(actions_seq, 3, "actions_seq")
        assert_rank(dt_seq, 3, "dt_seq")

        B, T, _, _, _ = rgb_seq.shape
        state = init_state or self.init_state(B, device=rgb_seq.device, dtype=rgb_seq.dtype)
        per_step = []
        last = None
        for t in range(T):
            reason_tok_t = None
            if reasoner_tok is not None:
                if reasoner_tok.ndim == 4:
                    reason_tok_t = reasoner_tok[:, t]
                else:
                    reason_tok_t = reasoner_tok
            out_t, state = self.step(
                rgb_t=rgb_seq[:, t],
                action_prev=actions_seq[:, t],
                dt_t=dt_seq[:, t],
                state=state,
                route_polyline=route_polyline,
                nav_cmd=nav_cmd,
                reasoner_tok=reason_tok_t,
                mode=mode,
            )
            per_step.append(out_t)
            last = out_t

        return {
            "last": last,
            "per_step": per_step,
            "final_state": state,
        }
