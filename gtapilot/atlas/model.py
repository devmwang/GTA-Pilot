from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .config import AtlasConfig
from .geometry.geometry_lifter import GeometryLifter
from .heads.aux_heads import AuxHeads
from .motion.action_encoder import ActionEncoder
from .motion.ego_filter import EgoFilter
from .planner.planner import PlannerHead, RiskContextDecoder
from .scheduler import AtlasHeadScheduler
from .state import AtlasState
from .temporal.temporal_mixer import TemporalMixer
from .utils import assert_rank, pad_image_to_size
from .vision.encoder_factory import build_camera_encoder
from .world.observation_pool import ObservationPool
from .world.reasoner_adapter import ReasonerBridge
from .world.route_adapter import RouteAdapter
from .world.world_memory import WorldMemory


class Atlas(nn.Module):
    """
    Atlas foundation model scaffold.

    Public API:
      - step(...) -> best_traj, aux_outputs, next_state
      - forward_train(...) -> debug dictionary for stage scripts and losses
    """

    def __init__(self, cfg: AtlasConfig):
        super().__init__()
        if cfg.vision.cam_tokens_per_frame != cfg.temporal.recent_cam_tokens:
            raise ValueError(
                "Vision tokenizer token count must match temporal recent_cam_tokens."
            )

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
        self.risk_decoder = RiskContextDecoder(cfg)
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
        history: torch.Tensor,
        new_value: torch.Tensor,
        keep: int,
    ) -> torch.Tensor:
        if keep <= 0:
            shape = list(history.shape)
            shape[1] = 0
            return history.new_zeros(shape)
        cat = torch.cat([history, new_value], dim=1)
        if cat.shape[1] <= keep:
            return cat
        return cat[:, -keep:]

    @staticmethod
    def _right_align_buffer(buffer: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        if buffer.shape[0] != values.shape[0]:
            raise ValueError("Batch size mismatch while seeding Atlas history buffers.")
        out = torch.zeros_like(buffer)
        keep = min(buffer.shape[1], values.shape[1])
        if keep > 0:
            out[:, -keep:] = values[:, -keep:].to(out.dtype)
        return out

    def _encode_camera_token_sequence(
        self,
        rgb_seq: torch.Tensor,
        *,
        chunk_size: int = 4,
    ) -> torch.Tensor:
        assert_rank(rgb_seq, 5, "rgb_seq")
        steps = rgb_seq.shape[1]
        tokens = []
        for start in range(0, steps, chunk_size):
            end = min(steps, start + chunk_size)
            _, chunk_tokens = self.vision(rgb_seq[:, start:end], pad_to_native=True)
            tokens.append(chunk_tokens)
        return torch.cat(tokens, dim=1)

    def _advance_temporal_caches(
        self,
        *,
        state: AtlasState,
        cam_tokens_cur: torch.Tensor,
        dt_t: torch.Tensor,
        frame_summary: torch.Tensor,
        dt_buffer: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        recent_keep = self.cfg.temporal.recent_cache_frames
        recent_cat = torch.cat([state.recent_cam_cache, cam_tokens_cur[:, None]], dim=1)
        recent_dt_cat = torch.cat([state.recent_dt_cache, dt_t[:, None]], dim=1)
        recent_valid_cat = torch.cat(
            [
                state.recent_valid,
                torch.ones(
                    cam_tokens_cur.shape[0],
                    1,
                    device=cam_tokens_cur.device,
                    dtype=torch.bool,
                ),
            ],
            dim=1,
        )

        if recent_cat.shape[1] > recent_keep:
            overflow = recent_cat.shape[1] - recent_keep
            aged_recent = recent_cat[:, :overflow]
            aged_recent_dt = recent_dt_cat[:, :overflow]
            aged_recent_valid = recent_valid_cat[:, :overflow]
            recent_next = recent_cat[:, -recent_keep:]
            recent_dt_next = recent_dt_cat[:, -recent_keep:]
            recent_valid_next = recent_valid_cat[:, -recent_keep:]
            compressed_aged = self.temporal.compress_history_frames(
                aged_recent,
                aged_recent_valid,
            )
            older_next = self._append_keep_last(
                state.older_cam_cache,
                compressed_aged,
                self.cfg.temporal.older_compressed_frames,
            )
            older_dt_next = self._append_keep_last(
                state.older_dt_cache,
                aged_recent_dt
                * aged_recent_valid[:, :, None].to(aged_recent_dt.dtype),
                self.cfg.temporal.older_compressed_frames,
            )
            older_valid_next = self._append_keep_last(
                state.older_valid,
                aged_recent_valid,
                self.cfg.temporal.older_compressed_frames,
            )
        else:
            recent_next = recent_cat
            recent_dt_next = recent_dt_cat
            recent_valid_next = recent_valid_cat
            older_next = state.older_cam_cache
            older_dt_next = state.older_dt_cache
            older_valid_next = state.older_valid

        mid_next = state.mid_summary_cache
        mid_dt_next = state.mid_dt_cache
        mid_valid_next = state.mid_valid
        stride = max(1, int(self.cfg.temporal.summary_stride_steps))
        if (state.step_index + 1) % stride == 0:
            history_span = min(stride, dt_buffer.shape[1])
            summary_dt = dt_buffer[:, -history_span:].sum(dim=1, keepdim=True)
            mid_next = self._append_keep_last(
                state.mid_summary_cache,
                frame_summary[:, None],
                self.cfg.temporal.mid_summary_frames,
            )
            mid_dt_next = self._append_keep_last(
                state.mid_dt_cache,
                summary_dt,
                self.cfg.temporal.mid_summary_frames,
            )
            mid_valid_next = self._append_keep_last(
                state.mid_valid,
                torch.ones(
                    cam_tokens_cur.shape[0],
                    1,
                    device=cam_tokens_cur.device,
                    dtype=torch.bool,
                ),
                self.cfg.temporal.mid_summary_frames,
            )

        return {
            "recent_cam_cache": recent_next,
            "recent_dt_cache": recent_dt_next,
            "recent_valid": recent_valid_next,
            "older_cam_cache": older_next,
            "older_dt_cache": older_dt_next,
            "older_valid": older_valid_next,
            "mid_summary_cache": mid_next,
            "mid_dt_cache": mid_dt_next,
            "mid_valid": mid_valid_next,
        }

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

        action_buffer = self._append_keep_last(
            state.action_buffer,
            action_prev[:, None],
            self.cfg.action.history_len,
        )
        dt_buffer = self._append_keep_last(
            state.dt_buffer,
            dt_t[:, None],
            self.cfg.action.history_len,
        )
        act_tokens = self.action_encoder(action_buffer, dt_buffer)

        temporal_out = self.temporal(
            current_cam_tokens=cam_tokens_cur,
            recent_cam_cache=state.recent_cam_cache,
            recent_dt_cache=state.recent_dt_cache,
            recent_valid=state.recent_valid,
            older_cam_cache=state.older_cam_cache,
            older_dt_cache=state.older_dt_cache,
            older_valid=state.older_valid,
            mid_summary_cache=state.mid_summary_cache,
            mid_dt_cache=state.mid_dt_cache,
            mid_valid=state.mid_valid,
        )
        cam_now = temporal_out["cam_now"]
        short_ctx = temporal_out["short_ctx"]
        older_ctx = temporal_out["older_ctx"]
        long_ctx = temporal_out["long_ctx"]
        frame_summary = temporal_out["frame_summary"]

        ego_out = self.ego_filter(
            cam_now,
            short_ctx,
            older_ctx,
            long_ctx,
            act_tokens,
            state.ego_filter_hidden,
        )
        ego_tokens_new = ego_out["ego_tokens"]
        if self.cfg.enable_route_tokens:
            route_tokens = self.route_adapter(
                route_polyline,
                nav_cmd,
                batch_size,
                rgb_t.device,
                rgb_t.dtype,
            )
        else:
            route_tokens = state.route_tokens

        if self.cfg.enable_reasoner_tokens:
            reasoner_tokens, reasoner_ttl = self.reasoner_bridge(
                reasoner_tok,
                state.reasoner_tokens,
                state.reasoner_ttl,
            )
        else:
            reasoner_tokens, reasoner_ttl = state.reasoner_tokens, state.reasoner_ttl

        geom_out = self.geometry(ctx8_now, cam_now, ego_tokens_new)
        obs_tokens = self.obs_pool(
            cam_now,
            short_ctx,
            older_ctx,
            long_ctx,
            geom_out["frustum_tokens"],
            ego_tokens_new,
            act_tokens,
        )
        temporal_tokens = torch.cat([short_ctx, older_ctx, long_ctx], dim=1)
        world_view, world_lifecycle = self.world(
            state=state,
            obs_tokens=obs_tokens,
            temporal_tokens=temporal_tokens,
            pose_delta=ego_out["pose_delta"],
            pose_uncertainty=ego_out["logvar_pose"].exp(),
            ego_tokens_new=ego_tokens_new,
            route_tokens=route_tokens,
            reasoner_tokens=reasoner_tokens,
            step_dt_s=dt_t,
        )
        risk_context = self.risk_decoder(world_view)
        planner_out = self.planner(world_view, ego_out["kinematics"], risk_context)

        active_heads = self.scheduler.active_heads(state.step_index, mode=mode)
        aux_out = self.aux(
            world_view,
            active_heads=active_heads - {"planner"},
            risk_context=risk_context,
        )

        cache_next = self._advance_temporal_caches(
            state=state,
            cam_tokens_cur=cam_tokens_cur,
            dt_t=dt_t,
            frame_summary=frame_summary,
            dt_buffer=dt_buffer,
        )
        next_state = AtlasState(
            recent_cam_cache=cache_next["recent_cam_cache"],
            older_cam_cache=cache_next["older_cam_cache"],
            mid_summary_cache=cache_next["mid_summary_cache"],
            recent_dt_cache=cache_next["recent_dt_cache"],
            older_dt_cache=cache_next["older_dt_cache"],
            mid_dt_cache=cache_next["mid_dt_cache"],
            recent_valid=cache_next["recent_valid"],
            older_valid=cache_next["older_valid"],
            mid_valid=cache_next["mid_valid"],
            action_buffer=action_buffer,
            dt_buffer=dt_buffer,
            static_grid=world_view.static_grid,
            dynamic_slots=world_view.dynamic_slots,
            speculative_slots=world_view.speculative_slots,
            dynamic_slot_age_s=world_lifecycle["dynamic_slot_age_s"],
            speculative_slot_age_s=world_lifecycle["speculative_slot_age_s"],
            dynamic_slot_alive=world_lifecycle["dynamic_slot_alive"],
            speculative_slot_alive=world_lifecycle["speculative_slot_alive"],
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
            "short_ctx": short_ctx,
            "older_ctx": older_ctx,
            "long_ctx": long_ctx,
            "frame_summary": frame_summary,
            "act_tokens": act_tokens,
            "obs_tokens": obs_tokens,
            "static_grid": world_view.static_grid,
            "persistent_tokens": world_view.persistent_tokens,
            "dynamic_slots": world_view.dynamic_slots,
            "speculative_slots": world_view.speculative_slots,
            "dynamic_slot_alive": world_view.dynamic_slot_alive,
            "speculative_slot_alive": world_view.speculative_slot_alive,
            **ego_out,
            **geom_out,
            **risk_context,
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
        rgb_recent: torch.Tensor,
        dt_recent: torch.Tensor,
        rgb_older: torch.Tensor,
        dt_older: torch.Tensor,
        rgb_mid: torch.Tensor,
        dt_mid: torch.Tensor,
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
        assert_rank(rgb_recent, 5, "rgb_recent")
        assert_rank(dt_recent, 3, "dt_recent")
        assert_rank(rgb_older, 5, "rgb_older")
        assert_rank(dt_older, 3, "dt_older")
        assert_rank(rgb_mid, 5, "rgb_mid")
        assert_rank(dt_mid, 3, "dt_mid")
        assert_rank(actions_hist, 3, "actions_hist")
        assert_rank(dt_hist, 3, "dt_hist")

        batch_size, recent_steps, _, _, _ = rgb_recent.shape
        if dt_recent.shape[:2] != rgb_recent.shape[:2]:
            raise ValueError("dt_recent must align with rgb_recent exactly.")
        if dt_older.shape[:2] != rgb_older.shape[:2]:
            raise ValueError("dt_older must align with rgb_older exactly.")
        if dt_mid.shape[:2] != rgb_mid.shape[:2]:
            raise ValueError("dt_mid must align with rgb_mid exactly.")
        if dt_hist.shape[:2] != actions_hist.shape[:2]:
            raise ValueError("dt_hist must align with actions_hist exactly.")
        state = init_state or self.init_state(
            batch_size,
            device=rgb_recent.device,
            dtype=rgb_recent.dtype,
        )

        if rgb_older.shape[1] > 0:
            older_tokens = self._encode_camera_token_sequence(rgb_older)
            older_tokens = older_tokens[:, -self.cfg.temporal.older_compressed_frames :]
            older_steps = older_tokens.shape[1]
            state.older_cam_cache = self._right_align_buffer(
                state.older_cam_cache,
                self.temporal.compress_history_frames(older_tokens).to(
                    state.older_cam_cache.dtype
                ),
            )
            state.older_dt_cache = self._right_align_buffer(
                state.older_dt_cache,
                dt_older[:, -self.cfg.temporal.older_compressed_frames :].to(
                    state.older_dt_cache.dtype
                ),
            )
            state.older_valid = self._right_align_buffer(
                state.older_valid,
                torch.ones(
                    batch_size,
                    older_steps,
                    device=rgb_recent.device,
                    dtype=torch.bool,
                ),
            )

        if rgb_mid.shape[1] > 0:
            mid_tokens = self._encode_camera_token_sequence(rgb_mid)
            mid_tokens = mid_tokens[:, -self.cfg.temporal.mid_summary_frames :]
            mid_steps = mid_tokens.shape[1]
            state.mid_summary_cache = self._right_align_buffer(
                state.mid_summary_cache,
                self.temporal.summarize_mid_frames(mid_tokens).to(
                    state.mid_summary_cache.dtype
                ),
            )
            state.mid_dt_cache = self._right_align_buffer(
                state.mid_dt_cache,
                dt_mid[:, -self.cfg.temporal.mid_summary_frames :].to(
                    state.mid_dt_cache.dtype
                ),
            )
            state.mid_valid = self._right_align_buffer(
                state.mid_valid,
                torch.ones(
                    batch_size,
                    mid_steps,
                    device=rgb_recent.device,
                    dtype=torch.bool,
                ),
            )

        if actions_hist.shape[1] < recent_steps or dt_hist.shape[1] < recent_steps:
            raise ValueError(
                "actions_hist and dt_hist must include at least one item per recent RGB step."
            )

        seed_actions = actions_hist[:, :-recent_steps]
        seed_dt = dt_hist[:, :-recent_steps]
        if seed_actions.shape[1] > 0:
            state.action_buffer = self._right_align_buffer(
                state.action_buffer,
                seed_actions[:, -self.cfg.action.history_len :].to(
                    state.action_buffer.dtype
                ),
            )
            state.dt_buffer = self._right_align_buffer(
                state.dt_buffer,
                seed_dt[:, -self.cfg.action.history_len :].to(
                    state.dt_buffer.dtype
                ),
            )

        action_seq = actions_hist[:, -recent_steps:]
        dt_seq = dt_recent[:, -recent_steps:]

        per_step = []
        last = None
        for step_idx in range(recent_steps):
            step_reasoner = None
            if reasoner_tok is not None:
                if reasoner_tok.ndim == 4:
                    step_reasoner = reasoner_tok[:, step_idx]
                else:
                    step_reasoner = reasoner_tok

            last, state = self._step_impl(
                rgb_t=rgb_recent[:, step_idx],
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
