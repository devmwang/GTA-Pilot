from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .config import AtlasConfig
from .scheduler import AtlasHeadScheduler
from .state import AtlasState
from .utils import assert_rank, pad_image_to_size
from .vision.encoder_factory import build_camera_encoder
from .temporal.temporal_mixer import TemporalMixer
from .motion.action_encoder import ActionEncoder
from .motion.ego_filter import EgoFilter
from .geometry.geometry_lifter import GeometryLifter
from .world.observation_pool import ObservationPool
from .world.route_adapter import RouteAdapter
from .world.reasoner_adapter import ReasonerBridge
from .world.world_memory import WorldMemory
from .planner.planner import PlannerHead
from .heads.aux_heads import AuxHeads


class Atlas(nn.Module):
    """
    Atlas foundation model scaffold.

    Public API:
      - step(...) -> best_traj, aux_outputs, next_state
      - forward_train(...) -> debug dictionary for stage scripts and losses
    """

    def __init__(self, cfg: AtlasConfig):
        super().__init__()
        self.cfg = cfg
        self.vision = build_camera_encoder(cfg)
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
        self.scheduler = AtlasHeadScheduler(cfg)

    def init_state(
        self,
        batch_size: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> AtlasState:
        return AtlasState.init_empty(
            self.cfg, batch_size=batch_size, device=device, dtype=dtype
        )

    @staticmethod
    def _append_keep_last(
        history: torch.Tensor, new_value: torch.Tensor, keep: int
    ) -> torch.Tensor:
        cat = torch.cat([history, new_value], dim=1)
        if cat.shape[1] <= keep:
            return cat
        return cat[:, -keep:]

    def _step_impl(
        self,
        rgb_t: torch.Tensor,
        action_prev: torch.Tensor,
        dt_t: torch.Tensor,
        state: AtlasState,
        route_polyline: Optional[torch.Tensor] = None,
        nav_cmd: Optional[torch.Tensor] = None,
        reasoner_tok: Optional[torch.Tensor] = None,
        mode: str = "drive",
    ) -> tuple[Dict[str, torch.Tensor], AtlasState]:
        assert_rank(rgb_t, 4, "rgb_t")
        assert_rank(action_prev, 2, "action_prev")
        assert_rank(dt_t, 2, "dt_t")
        batch_size = rgb_t.shape[0]

        rgb_pad = pad_image_to_size(
            rgb_t, self.cfg.image.padded_height, self.cfg.image.padded_width
        )
        vision_features, cam_tokens_cur = self.vision(rgb_t[:, None], pad_to_native=True)
        cam_tokens_cur = cam_tokens_cur[:, 0]
        current_vision_features = {
            name: tensor[:, 0] for name, tensor in vision_features.items()
        }
        ctx8_now = current_vision_features["ctx_8x"]

        cam_seq = torch.cat([state.cam_token_cache, cam_tokens_cur[:, None]], dim=1)
        dt_seq = torch.cat(
            [state.dt_buffer[:, -(self.cfg.temporal.num_frames - 1) :], dt_t[:, None]],
            dim=1,
        )
        cam_now, frame_sum = self.temporal(cam_seq, dt_seq)

        action_buffer = self._append_keep_last(
            state.action_buffer, action_prev[:, None], self.cfg.action.history_len
        )
        dt_buffer = self._append_keep_last(
            state.dt_buffer, dt_t[:, None], self.cfg.action.history_len
        )
        act_tokens = self.action_encoder(action_buffer, dt_buffer)

        ego_out = self.ego_filter(cam_now, frame_sum, act_tokens, state.ego_filter_hidden)
        ego_tokens_new = ego_out["ego_tokens"]
        if self.cfg.enable_route_tokens:
            route_tokens = self.route_adapter(
                route_polyline, nav_cmd, batch_size, rgb_t.device, rgb_t.dtype
            )
        else:
            route_tokens = state.route_tokens

        if self.cfg.enable_reasoner_tokens:
            reasoner_tokens, reasoner_ttl = self.reasoner_bridge(
                reasoner_tok, state.reasoner_tokens, state.reasoner_ttl
            )
        else:
            reasoner_tokens, reasoner_ttl = state.reasoner_tokens, state.reasoner_ttl

        geom_out = self.geometry(ctx8_now, cam_now, ego_tokens_new)
        obs_tokens = self.obs_pool(
            cam_now, geom_out["frustum_tokens"], ego_tokens_new, act_tokens
        )
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

        active_heads = self.scheduler.active_heads(state.step_index, mode=mode)
        aux_out = self.aux(world_view, active_heads=active_heads - {"planner"})

        next_state = AtlasState(
            cam_token_cache=self._append_keep_last(
                state.cam_token_cache,
                cam_tokens_cur[:, None],
                self.cfg.temporal.num_frames - 1,
            ),
            frame_sum_cache=self._append_keep_last(
                state.frame_sum_cache,
                frame_sum[:, -1:, :],
                self.cfg.temporal.num_frames - 1,
            ),
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
                [ego_out["pose_delta"], ego_out["kinematics"], ego_out["logvar_pose"]],
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
            **current_vision_features,
        }
        return outputs, next_state

    def step(
        self,
        rgb_t: torch.Tensor,
        action_prev: torch.Tensor,
        dt_t: torch.Tensor,
        state: AtlasState,
        route_polyline: Optional[torch.Tensor] = None,
        nav_cmd: Optional[torch.Tensor] = None,
        reasoner_tok: Optional[torch.Tensor] = None,
        mode: str = "drive",
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor], AtlasState]:
        outputs, next_state = self._step_impl(
            rgb_t=rgb_t,
            action_prev=action_prev,
            dt_t=dt_t,
            state=state,
            route_polyline=route_polyline,
            nav_cmd=nav_cmd,
            reasoner_tok=reasoner_tok,
            mode=mode,
        )
        aux = {key: value for key, value in outputs.items() if key != "best_traj"}
        return outputs["best_traj"], aux, next_state

    def forward_train(
        self,
        rgb_past: torch.Tensor,
        actions_hist: torch.Tensor,
        dt_hist: torch.Tensor,
        route_polyline: Optional[torch.Tensor] = None,
        nav_cmd: Optional[torch.Tensor] = None,
        reasoner_tok: Optional[torch.Tensor] = None,
        privileged: Optional[Dict[str, Any]] = None,
        init_state: Optional[AtlasState] = None,
        stage: str = "stage2",
        mode: str = "inspect",
    ) -> Dict[str, Any]:
        assert_rank(rgb_past, 5, "rgb_past")
        assert_rank(actions_hist, 3, "actions_hist")
        assert_rank(dt_hist, 3, "dt_hist")

        batch_size, steps, _, _, _ = rgb_past.shape
        state = init_state or self.init_state(
            batch_size, device=rgb_past.device, dtype=rgb_past.dtype
        )
        if actions_hist.shape[1] >= self.cfg.action.history_len:
            state.action_buffer = actions_hist[:, -self.cfg.action.history_len :].to(
                state.action_buffer.dtype
            )
            state.dt_buffer = dt_hist[:, -self.cfg.action.history_len :].to(
                state.dt_buffer.dtype
            )

        if actions_hist.shape[1] == steps:
            action_seq = actions_hist
            dt_seq = dt_hist
        elif actions_hist.shape[1] >= steps:
            action_seq = actions_hist[:, -steps:]
            dt_seq = dt_hist[:, -steps:]
        else:
            raise ValueError(
                "actions_hist / dt_hist must provide either one item per RGB step or"
                " at least enough history to seed the recurrent buffers."
            )

        per_step = []
        last = None
        for step_idx in range(steps):
            step_reasoner = None
            if reasoner_tok is not None:
                if reasoner_tok.ndim == 4:
                    step_reasoner = reasoner_tok[:, step_idx]
                else:
                    step_reasoner = reasoner_tok

            last, state = self._step_impl(
                rgb_t=rgb_past[:, step_idx],
                action_prev=action_seq[:, step_idx],
                dt_t=dt_seq[:, step_idx],
                state=state,
                route_polyline=route_polyline,
                nav_cmd=nav_cmd,
                reasoner_tok=step_reasoner,
                mode=mode,
            )
            per_step.append(last)

        return {
            "stage": stage,
            "privileged": privileged or {},
            "last": last,
            "per_step": per_step,
            "final_state": state,
        }
