from __future__ import annotations

import contextlib
from dataclasses import replace
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
        new_value = new_value.to(device=history.device, dtype=history.dtype)
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
            out[:, -keep:] = values[:, -keep:].to(device=out.device, dtype=out.dtype)
        return out

    def _model_device_dtype(self) -> tuple[torch.device, torch.dtype]:
        param = next(self.parameters())
        return param.device, param.dtype

    def _state_device_dtype(self, reference: torch.Tensor) -> tuple[torch.device, torch.dtype]:
        device, dtype = self._model_device_dtype()
        if device.type == "cuda" and torch.is_autocast_enabled():
            dtype = torch.get_autocast_gpu_dtype()
        elif not torch.is_floating_point(reference):
            dtype = torch.float32
        return device, dtype

    def _move_rgb_to_model_device(self, rgb: torch.Tensor) -> torch.Tensor:
        device, _ = self._model_device_dtype()
        rgb = rgb.to(device=device, non_blocking=device.type == "cuda")
        if rgb.dtype == torch.uint8:
            rgb = rgb.float().mul_(1.0 / 255.0)
        elif not torch.is_floating_point(rgb):
            rgb = rgb.float()
        return rgb

    @staticmethod
    def _grad_context(enabled: bool):
        if enabled:
            return contextlib.nullcontext()
        return torch.no_grad()

    def _vision_encode_cam_tokens(self, rgb_step: torch.Tensor) -> torch.Tensor:
        _, cam_tokens = self.vision(rgb_step[:, None], pad_to_native=True)
        return cam_tokens[:, 0]

    def _vision_encode_ctx8_and_cam_tokens(
        self,
        rgb_step: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        vision_features, cam_tokens = self.vision(rgb_step[:, None], pad_to_native=True)
        return vision_features["ctx_8x"][:, 0], cam_tokens[:, 0]

    def _encode_current_step(
        self,
        rgb_step: torch.Tensor,
        *,
        need_ctx_8x: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        rgb_step = self._move_rgb_to_model_device(rgb_step)
        if need_ctx_8x:
            ctx_8x, cam_tokens_cur = self._vision_encode_ctx8_and_cam_tokens(rgb_step)
            return cam_tokens_cur, ctx_8x
        cam_tokens_cur = self._vision_encode_cam_tokens(rgb_step)
        return cam_tokens_cur, None

    def _temporal_forward_tuple(
        self,
        current_cam_tokens: torch.Tensor,
        recent_cam_cache: torch.Tensor,
        recent_dt_cache: torch.Tensor,
        recent_valid: torch.Tensor,
        older_cam_cache: torch.Tensor,
        older_dt_cache: torch.Tensor,
        older_valid: torch.Tensor,
        mid_summary_cache: torch.Tensor,
        mid_dt_cache: torch.Tensor,
        mid_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        temporal_out = self.temporal(
            current_cam_tokens=current_cam_tokens,
            recent_cam_cache=recent_cam_cache,
            recent_dt_cache=recent_dt_cache,
            recent_valid=recent_valid,
            older_cam_cache=older_cam_cache,
            older_dt_cache=older_dt_cache,
            older_valid=older_valid,
            mid_summary_cache=mid_summary_cache,
            mid_dt_cache=mid_dt_cache,
            mid_valid=mid_valid,
        )
        return (
            temporal_out["cam_now"],
            temporal_out["short_ctx"],
            temporal_out["older_ctx"],
            temporal_out["long_ctx"],
            temporal_out["frame_summary"],
        )

    def _temporal_forward(
        self,
        *,
        cam_tokens_cur: torch.Tensor,
        state: AtlasState,
        recent_valid: torch.Tensor,
        mid_valid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        cam_now, short_ctx, older_ctx, long_ctx, frame_summary = self._temporal_forward_tuple(
            cam_tokens_cur,
            state.recent_cam_cache,
            state.recent_dt_cache,
            recent_valid,
            state.older_cam_cache,
            state.older_dt_cache,
            state.older_valid,
            state.mid_summary_cache,
            state.mid_dt_cache,
            mid_valid,
        )
        return {
            "cam_now": cam_now,
            "short_ctx": short_ctx,
            "older_ctx": older_ctx,
            "long_ctx": long_ctx,
            "frame_summary": frame_summary,
        }

    def _geometry_stage1b(
        self,
        *,
        ctx_8x: torch.Tensor,
        cam_now: torch.Tensor,
        ego_tokens: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        geom_out = self.geometry(ctx_8x, cam_now, ego_tokens)
        return {
            "depth_mean": geom_out["depth_mean"],
            "track_offsets": geom_out["track_offsets"],
        }

    def _geometry_stage1c_frustum(
        self,
        *,
        ctx_8x: torch.Tensor,
        cam_now: torch.Tensor,
        ego_tokens: torch.Tensor,
    ) -> torch.Tensor:
        return self.geometry(ctx_8x, cam_now, ego_tokens)["frustum_tokens"]

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
            rgb_chunk = self._move_rgb_to_model_device(rgb_seq[:, start:end])
            _, chunk_tokens = self.vision(rgb_chunk, pad_to_native=True)
            tokens.append(chunk_tokens)
        return torch.cat(tokens, dim=1)

    @staticmethod
    def _stack_step_dicts(per_step: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        if not per_step:
            return {}
        keys = set.intersection(*(set(step.keys()) for step in per_step))
        stacked: dict[str, torch.Tensor] = {}
        for key in keys:
            values = [step[key] for step in per_step]
            if all(isinstance(value, torch.Tensor) for value in values):
                stacked[key] = torch.stack(values, dim=1)
        return stacked

    def _seed_stage_state(
        self,
        *,
        rgb_recent: torch.Tensor,
        rgb_older: torch.Tensor,
        dt_older: torch.Tensor,
        rgb_mid: torch.Tensor,
        dt_mid: torch.Tensor,
        actions_hist: torch.Tensor | None = None,
        dt_hist: torch.Tensor | None = None,
        init_state: AtlasState | None = None,
        seed_action_history: bool = False,
        seed_older_with_grad: bool = False,
        seed_mid_with_grad: bool = False,
    ) -> AtlasState:
        batch_size = rgb_recent.shape[0]
        state_device, state_dtype = self._state_device_dtype(rgb_recent)
        state = init_state or self.init_state(
            batch_size,
            device=state_device,
            dtype=state_dtype,
        )

        if rgb_older.shape[1] > 0:
            with self._grad_context(seed_older_with_grad):
                older_tokens = self._encode_camera_token_sequence(rgb_older)
                older_tokens = older_tokens[:, -self.cfg.temporal.older_compressed_frames :]
                compressed_older = self.temporal.compress_history_frames(older_tokens)
            older_steps = older_tokens.shape[1]
            state = replace(
                state,
                older_cam_cache=self._right_align_buffer(
                    state.older_cam_cache,
                    compressed_older.to(state.older_cam_cache.dtype),
                ),
                older_dt_cache=self._right_align_buffer(
                    state.older_dt_cache,
                    dt_older[:, -older_steps:].to(state.older_dt_cache.dtype),
                ),
                older_valid=self._right_align_buffer(
                    state.older_valid,
                    torch.ones(
                        batch_size,
                        older_steps,
                        device=state.older_valid.device,
                        dtype=torch.bool,
                    ),
                ),
            )

        if rgb_mid.shape[1] > 0:
            with self._grad_context(seed_mid_with_grad):
                mid_summary_tokens = self._emit_sparse_summary_seed_tokens(
                    rgb_mid,
                    dt_mid,
                )
            mid_summary_tokens = mid_summary_tokens[
                :, -self.cfg.temporal.mid_summary_frames :
            ]
            mid_steps = mid_summary_tokens.shape[1]
            state = replace(
                state,
                mid_summary_cache=self._right_align_buffer(
                    state.mid_summary_cache,
                    mid_summary_tokens.to(state.mid_summary_cache.dtype),
                ),
                mid_dt_cache=self._right_align_buffer(
                    state.mid_dt_cache,
                    dt_mid[:, -mid_steps:].to(state.mid_dt_cache.dtype),
                ),
                mid_valid=self._right_align_buffer(
                    state.mid_valid,
                    torch.ones(
                        batch_size,
                        mid_steps,
                        device=state.mid_valid.device,
                        dtype=torch.bool,
                    ),
                ),
            )

        if seed_action_history:
            if actions_hist is None or dt_hist is None:
                raise ValueError("actions_hist and dt_hist are required for action-seeded stages.")
            recent_steps = rgb_recent.shape[1]
            if actions_hist.shape[1] < recent_steps or dt_hist.shape[1] < recent_steps:
                raise ValueError(
                    "actions_hist and dt_hist must include at least one item per recent RGB step."
                )
            seed_actions = actions_hist[:, :-recent_steps]
            seed_dt = dt_hist[:, :-recent_steps]
            if seed_actions.shape[1] > 0:
                state = replace(
                    state,
                    action_buffer=self._right_align_buffer(
                        state.action_buffer,
                        seed_actions[:, -self.cfg.action.history_len :].to(
                            state.action_buffer.dtype
                        ),
                    ),
                    dt_buffer=self._right_align_buffer(
                        state.dt_buffer,
                        seed_dt[:, -self.cfg.action.history_len :].to(
                            state.dt_buffer.dtype
                        ),
                    ),
                )
        return state

    def _emit_sparse_summary_seed_tokens(
        self,
        rgb_mid: torch.Tensor,
        dt_mid: torch.Tensor,
    ) -> torch.Tensor:
        if rgb_mid.shape[1] == 0:
            return rgb_mid.new_zeros(
                rgb_mid.shape[0],
                0,
                self.cfg.temporal.mid_summary_tokens,
                self.cfg.hidden_dim,
            )

        state_device, state_dtype = self._state_device_dtype(rgb_mid)
        state = self.init_state(
            rgb_mid.shape[0],
            device=state_device,
            dtype=state_dtype,
        )
        summaries: list[torch.Tensor] = []
        for step_idx in range(rgb_mid.shape[1]):
            dt_buffer = self._append_keep_last(
                state.dt_buffer,
                dt_mid[:, step_idx : step_idx + 1],
                self.cfg.action.history_len,
            )
            cam_tokens_cur, _ = self._encode_current_step(
                rgb_mid[:, step_idx],
                need_ctx_8x=False,
            )
            temporal_out = self._temporal_forward(
                cam_tokens_cur=cam_tokens_cur,
                state=state,
                recent_valid=state.recent_valid,
                mid_valid=state.mid_valid,
            )
            summaries.append(temporal_out["frame_summary"])
            cache_next = self._advance_temporal_caches(
                state=state,
                cam_tokens_cur=cam_tokens_cur,
                dt_t=dt_mid[:, step_idx],
                frame_summary=temporal_out["frame_summary"],
                dt_buffer=dt_buffer,
                insert_mid_summary=False,
            )
            state = replace(
                state,
                recent_cam_cache=cache_next["recent_cam_cache"],
                recent_dt_cache=cache_next["recent_dt_cache"],
                recent_valid=cache_next["recent_valid"],
                older_cam_cache=cache_next["older_cam_cache"],
                older_dt_cache=cache_next["older_dt_cache"],
                older_valid=cache_next["older_valid"],
                dt_buffer=dt_buffer,
                step_index=state.step_index + 1,
            )
        return torch.stack(summaries, dim=1)

    def _apply_student_rgb_mask(
        self,
        rgb_t: torch.Tensor,
        *,
        erase_prob: float,
        erase_frac: float,
    ) -> torch.Tensor:
        if erase_prob <= 0.0 or erase_frac <= 0.0 or not self.training:
            return rgb_t
        out = rgb_t.clone()
        batch, _, height, width = out.shape
        erase_h = max(1, int(height * erase_frac))
        erase_w = max(1, int(width * erase_frac))
        for batch_idx in range(batch):
            if torch.rand((), device=out.device).item() >= erase_prob:
                continue
            y0 = int(torch.randint(0, max(1, height - erase_h + 1), (1,), device=out.device).item())
            x0 = int(torch.randint(0, max(1, width - erase_w + 1), (1,), device=out.device).item())
            out[batch_idx, :, y0 : y0 + erase_h, x0 : x0 + erase_w] = 0.0
        return out

    def _apply_token_dropout(
        self,
        tokens: torch.Tensor,
        drop_prob: float,
    ) -> torch.Tensor:
        if drop_prob <= 0.0 or not self.training:
            return tokens
        keep = (torch.rand(tokens.shape[:2], device=tokens.device) >= drop_prob).to(tokens.dtype)
        return tokens * keep.unsqueeze(-1)

    def _apply_valid_dropout(
        self,
        valid: torch.Tensor,
        drop_prob: float,
    ) -> torch.Tensor:
        if drop_prob <= 0.0 or not self.training or valid.numel() == 0:
            return valid
        keep = torch.rand(valid.shape, device=valid.device) >= drop_prob
        return valid & keep

    @staticmethod
    def _select_reasoner_step(
        reasoner_tok: Optional[torch.Tensor],
        step_idx: int,
    ) -> Optional[torch.Tensor]:
        if reasoner_tok is None:
            return None
        if reasoner_tok.ndim == 4:
            return reasoner_tok[:, step_idx]
        return reasoner_tok

    def _temporal_only_step(
        self,
        *,
        rgb_t: torch.Tensor,
        dt_t: torch.Tensor,
        state: AtlasState,
        dt_buffer_for_summary: torch.Tensor,
        student_masking: dict[str, float] | None = None,
        need_ctx_8x: bool = False,
    ) -> tuple[dict[str, torch.Tensor], AtlasState]:
        rgb_in = rgb_t
        if student_masking is not None:
            rgb_in = self._apply_student_rgb_mask(
                rgb_t,
                erase_prob=float(student_masking.get("rgb_erasing_prob", 0.0)),
                erase_frac=float(student_masking.get("rgb_erasing_frac", 0.0)),
            )
        cam_tokens_cur, ctx_8x = self._encode_current_step(
            rgb_in,
            need_ctx_8x=need_ctx_8x,
        )
        if student_masking is not None:
            cam_tokens_cur = self._apply_token_dropout(
                cam_tokens_cur,
                float(student_masking.get("token_dropout_prob", 0.0)),
            )
            recent_valid = self._apply_valid_dropout(
                state.recent_valid,
                float(student_masking.get("recent_frame_drop_prob", 0.0)),
            )
            mid_valid = self._apply_valid_dropout(
                state.mid_valid,
                float(student_masking.get("summary_frame_drop_prob", 0.0)),
            )
        else:
            recent_valid = state.recent_valid
            mid_valid = state.mid_valid
        temporal_out = self._temporal_forward(
            cam_tokens_cur=cam_tokens_cur,
            state=state,
            recent_valid=recent_valid,
            mid_valid=mid_valid,
        )
        cache_next = self._advance_temporal_caches(
            state=state,
            cam_tokens_cur=cam_tokens_cur,
            dt_t=dt_t,
            frame_summary=temporal_out["frame_summary"],
            dt_buffer=dt_buffer_for_summary,
        )
        next_state = replace(
            state,
            recent_cam_cache=cache_next["recent_cam_cache"],
            recent_dt_cache=cache_next["recent_dt_cache"],
            recent_valid=cache_next["recent_valid"],
            older_cam_cache=cache_next["older_cam_cache"],
            older_dt_cache=cache_next["older_dt_cache"],
            older_valid=cache_next["older_valid"],
            mid_summary_cache=cache_next["mid_summary_cache"],
            mid_dt_cache=cache_next["mid_dt_cache"],
            mid_valid=cache_next["mid_valid"],
            dt_buffer=dt_buffer_for_summary,
            step_index=state.step_index + 1,
        )
        outputs = {
            "cam_now": temporal_out["cam_now"],
            "short_ctx": temporal_out["short_ctx"],
            "older_ctx": temporal_out["older_ctx"],
            "long_ctx": temporal_out["long_ctx"],
            "frame_summary": temporal_out["frame_summary"],
        }
        if ctx_8x is not None:
            outputs["ctx_8x"] = ctx_8x
        return outputs, next_state

    def _apply_stage1c_temporal_corruption(
        self,
        *,
        temporal_out: dict[str, torch.Tensor],
        corruption: dict[str, float] | None,
    ) -> dict[str, torch.Tensor]:
        if corruption is None or not self.training:
            return temporal_out

        cam_drop_prob = float(corruption.get("cam_drop_prob", 0.0))
        context_family_drop_prob = float(corruption.get("context_family_drop_prob", 0.0))

        if cam_drop_prob > 0.0:
            temporal_out = {
                **temporal_out,
                "cam_now": self._apply_token_dropout(temporal_out["cam_now"], cam_drop_prob),
            }

        if context_family_drop_prob > 0.0:
            family_keys = ("short_ctx", "older_ctx", "long_ctx")
            batch_size = temporal_out["cam_now"].shape[0]
            apply_dropout = torch.rand(batch_size, device=temporal_out["cam_now"].device) < context_family_drop_prob
            family_index = torch.randint(
                0,
                len(family_keys),
                (batch_size,),
                device=temporal_out["cam_now"].device,
            )
            for current_family_index, family_key in enumerate(family_keys):
                family_tokens = temporal_out[family_key]
                drop_mask = (
                    apply_dropout & (family_index == current_family_index)
                ).to(family_tokens.dtype)[:, None, None]
                temporal_out[family_key] = family_tokens * (1.0 - drop_mask)

        return temporal_out

    def _apply_stage1c_frustum_dropout(
        self,
        frustum_tokens: torch.Tensor,
        corruption: dict[str, float] | None,
    ) -> torch.Tensor:
        if corruption is None or not self.training:
            return frustum_tokens
        frustum_drop_prob = float(corruption.get("frustum_drop_prob", 0.0))
        if frustum_drop_prob <= 0.0:
            return frustum_tokens
        return self._apply_token_dropout(frustum_tokens, frustum_drop_prob)

    def forward_stage1a(
        self,
        rgb_recent: torch.Tensor,
        dt_recent: torch.Tensor,
        rgb_older: torch.Tensor,
        dt_older: torch.Tensor,
        rgb_mid: torch.Tensor,
        dt_mid: torch.Tensor,
        *,
        init_state: AtlasState | None = None,
        student_masking: dict[str, float] | None = None,
        seed_older_with_grad: bool = False,
        seed_mid_with_grad: bool = False,
    ) -> Dict[str, Any]:
        assert_rank(rgb_recent, 5, "rgb_recent")
        assert_rank(dt_recent, 3, "dt_recent")
        assert_rank(rgb_older, 5, "rgb_older")
        assert_rank(dt_older, 3, "dt_older")
        assert_rank(rgb_mid, 5, "rgb_mid")
        assert_rank(dt_mid, 3, "dt_mid")
        if dt_recent.shape[:2] != rgb_recent.shape[:2]:
            raise ValueError("dt_recent must align with rgb_recent exactly.")
        if dt_older.shape[:2] != rgb_older.shape[:2]:
            raise ValueError("dt_older must align with rgb_older exactly.")
        if dt_mid.shape[:2] != rgb_mid.shape[:2]:
            raise ValueError("dt_mid must align with rgb_mid exactly.")

        state = self._seed_stage_state(
            rgb_recent=rgb_recent,
            rgb_older=rgb_older,
            dt_older=dt_older,
            rgb_mid=rgb_mid,
            dt_mid=dt_mid,
            init_state=init_state,
            seed_action_history=False,
            seed_older_with_grad=seed_older_with_grad,
            seed_mid_with_grad=seed_mid_with_grad,
        )
        per_step: list[dict[str, torch.Tensor]] = []
        for step_idx in range(rgb_recent.shape[1]):
            dt_buffer = self._append_keep_last(
                state.dt_buffer,
                dt_recent[:, step_idx : step_idx + 1],
                self.cfg.action.history_len,
            )
            step_out, state = self._temporal_only_step(
                rgb_t=rgb_recent[:, step_idx],
                dt_t=dt_recent[:, step_idx],
                state=state,
                dt_buffer_for_summary=dt_buffer,
                student_masking=student_masking,
                need_ctx_8x=False,
            )
            per_step.append(
                {
                    "cam_now": step_out["cam_now"],
                    "frame_summary": step_out["frame_summary"],
                }
            )
        return {
            "seq": self._stack_step_dicts(per_step),
            "last": per_step[-1] if per_step else {},
            "final_state": state,
        }

    def forward_stage1b(
        self,
        rgb_recent: torch.Tensor,
        dt_recent: torch.Tensor,
        rgb_older: torch.Tensor,
        dt_older: torch.Tensor,
        rgb_mid: torch.Tensor,
        dt_mid: torch.Tensor,
        actions_hist: torch.Tensor,
        dt_hist: torch.Tensor,
        *,
        init_state: AtlasState | None = None,
        seed_older_with_grad: bool = False,
        seed_mid_with_grad: bool = False,
    ) -> Dict[str, Any]:
        assert_rank(rgb_recent, 5, "rgb_recent")
        assert_rank(dt_recent, 3, "dt_recent")
        assert_rank(rgb_older, 5, "rgb_older")
        assert_rank(dt_older, 3, "dt_older")
        assert_rank(rgb_mid, 5, "rgb_mid")
        assert_rank(dt_mid, 3, "dt_mid")
        assert_rank(actions_hist, 3, "actions_hist")
        assert_rank(dt_hist, 3, "dt_hist")
        if dt_recent.shape[:2] != rgb_recent.shape[:2]:
            raise ValueError("dt_recent must align with rgb_recent exactly.")
        if dt_older.shape[:2] != rgb_older.shape[:2]:
            raise ValueError("dt_older must align with rgb_older exactly.")
        if dt_mid.shape[:2] != rgb_mid.shape[:2]:
            raise ValueError("dt_mid must align with rgb_mid exactly.")
        if dt_hist.shape[:2] != actions_hist.shape[:2]:
            raise ValueError("dt_hist must align with actions_hist exactly.")
        state = self._seed_stage_state(
            rgb_recent=rgb_recent,
            rgb_older=rgb_older,
            dt_older=dt_older,
            rgb_mid=rgb_mid,
            dt_mid=dt_mid,
            actions_hist=actions_hist,
            dt_hist=dt_hist,
            init_state=init_state,
            seed_action_history=True,
            seed_older_with_grad=seed_older_with_grad,
            seed_mid_with_grad=seed_mid_with_grad,
        )
        recent_steps = rgb_recent.shape[1]
        action_seq = actions_hist[:, -recent_steps:]
        dt_seq = dt_recent[:, -recent_steps:]
        per_step: list[dict[str, torch.Tensor]] = []
        for step_idx in range(recent_steps):
            action_buffer = self._append_keep_last(
                state.action_buffer,
                action_seq[:, step_idx : step_idx + 1],
                self.cfg.action.history_len,
            )
            dt_buffer = self._append_keep_last(
                state.dt_buffer,
                dt_seq[:, step_idx : step_idx + 1],
                self.cfg.action.history_len,
            )
            temporal_out, state = self._temporal_only_step(
                rgb_t=rgb_recent[:, step_idx],
                dt_t=dt_seq[:, step_idx],
                state=state,
                dt_buffer_for_summary=dt_buffer,
                need_ctx_8x=True,
            )
            act_tokens = self.action_encoder(action_buffer, dt_buffer)
            ego_out = self.ego_filter(
                temporal_out["cam_now"],
                temporal_out["short_ctx"],
                temporal_out["older_ctx"],
                temporal_out["long_ctx"],
                act_tokens,
                state.ego_filter_hidden,
            )
            geom_out = self._geometry_stage1b(
                ctx_8x=temporal_out["ctx_8x"],
                cam_now=temporal_out["cam_now"],
                ego_tokens=ego_out["ego_tokens"],
            )
            state = replace(
                state,
                action_buffer=action_buffer,
                dt_buffer=dt_buffer,
                ego_filter_hidden=ego_out["hidden_next"],
                ego_tokens=ego_out["ego_tokens"],
                pose_belief=torch.cat(
                    [
                        ego_out["pose_delta"],
                        ego_out["kinematics"],
                        ego_out["logvar_pose"],
                    ],
                    dim=-1,
                ),
            )
            per_step.append(
                {
                    "cam_now": temporal_out["cam_now"],
                    "frame_summary": temporal_out["frame_summary"],
                    "pose_delta": ego_out["pose_delta"],
                    "kinematics": ego_out["kinematics"],
                    "logvar_pose": ego_out["logvar_pose"],
                    "depth_mean": geom_out["depth_mean"],
                    "track_offsets": geom_out["track_offsets"],
                }
            )
        return {
            "seq": self._stack_step_dicts(per_step),
            "last": per_step[-1] if per_step else {},
            "final_state": state,
        }

    def forward_stage1c(
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
        *,
        init_state: AtlasState | None = None,
        student_corruption: dict[str, float] | None = None,
        seed_older_with_grad: bool = False,
        seed_mid_with_grad: bool = False,
    ) -> Dict[str, Any]:
        assert_rank(rgb_recent, 5, "rgb_recent")
        assert_rank(dt_recent, 3, "dt_recent")
        assert_rank(rgb_older, 5, "rgb_older")
        assert_rank(dt_older, 3, "dt_older")
        assert_rank(rgb_mid, 5, "rgb_mid")
        assert_rank(dt_mid, 3, "dt_mid")
        assert_rank(actions_hist, 3, "actions_hist")
        assert_rank(dt_hist, 3, "dt_hist")
        if dt_recent.shape[:2] != rgb_recent.shape[:2]:
            raise ValueError("dt_recent must align with rgb_recent exactly.")
        if dt_older.shape[:2] != rgb_older.shape[:2]:
            raise ValueError("dt_older must align with rgb_older exactly.")
        if dt_mid.shape[:2] != rgb_mid.shape[:2]:
            raise ValueError("dt_mid must align with rgb_mid exactly.")
        if dt_hist.shape[:2] != actions_hist.shape[:2]:
            raise ValueError("dt_hist must align with actions_hist exactly.")
        state = self._seed_stage_state(
            rgb_recent=rgb_recent,
            rgb_older=rgb_older,
            dt_older=dt_older,
            rgb_mid=rgb_mid,
            dt_mid=dt_mid,
            actions_hist=actions_hist,
            dt_hist=dt_hist,
            init_state=init_state,
            seed_action_history=True,
            seed_older_with_grad=seed_older_with_grad,
            seed_mid_with_grad=seed_mid_with_grad,
        )
        recent_steps = rgb_recent.shape[1]
        action_seq = actions_hist[:, -recent_steps:]
        dt_seq = dt_recent[:, -recent_steps:]
        per_step: list[dict[str, torch.Tensor]] = []
        for step_idx in range(recent_steps):
            action_buffer = self._append_keep_last(
                state.action_buffer,
                action_seq[:, step_idx : step_idx + 1],
                self.cfg.action.history_len,
            )
            dt_buffer = self._append_keep_last(
                state.dt_buffer,
                dt_seq[:, step_idx : step_idx + 1],
                self.cfg.action.history_len,
            )
            temporal_out, state = self._temporal_only_step(
                rgb_t=rgb_recent[:, step_idx],
                dt_t=dt_seq[:, step_idx],
                state=state,
                dt_buffer_for_summary=dt_buffer,
                need_ctx_8x=True,
            )
            act_tokens = self.action_encoder(action_buffer, dt_buffer)
            ego_out = self.ego_filter(
                temporal_out["cam_now"],
                temporal_out["short_ctx"],
                temporal_out["older_ctx"],
                temporal_out["long_ctx"],
                act_tokens,
                state.ego_filter_hidden,
            )
            frustum_tokens = self._geometry_stage1c_frustum(
                ctx_8x=temporal_out["ctx_8x"],
                cam_now=temporal_out["cam_now"],
                ego_tokens=ego_out["ego_tokens"],
            )
            frustum_tokens = self._apply_stage1c_frustum_dropout(
                frustum_tokens,
                student_corruption,
            )
            batch_size = rgb_recent.shape[0]
            route_tokens = self.route_adapter(
                route_polyline,
                nav_cmd,
                batch_size,
                state.route_tokens.device,
                state.route_tokens.dtype,
            ) if self.cfg.enable_route_tokens else state.route_tokens
            reasoner_tokens, reasoner_ttl = (
                self.reasoner_bridge(
                    self._select_reasoner_step(reasoner_tok, step_idx),
                    state.reasoner_tokens,
                    state.reasoner_ttl,
                )
                if self.cfg.enable_reasoner_tokens
                else (state.reasoner_tokens, state.reasoner_ttl)
            )
            obs_tokens = self.obs_pool(
                temporal_out["cam_now"],
                temporal_out["short_ctx"],
                temporal_out["older_ctx"],
                temporal_out["long_ctx"],
                frustum_tokens,
                ego_out["ego_tokens"],
                act_tokens,
            )
            temporal_tokens = torch.cat(
                [
                    temporal_out["short_ctx"],
                    temporal_out["older_ctx"],
                    temporal_out["long_ctx"],
                ],
                dim=1,
            )
            world_view, world_lifecycle = self.world(
                state=state,
                obs_tokens=obs_tokens,
                temporal_tokens=temporal_tokens,
                pose_delta=ego_out["pose_delta"],
                pose_uncertainty=ego_out["logvar_pose"].exp(),
                ego_tokens_new=ego_out["ego_tokens"],
                route_tokens=route_tokens,
                reasoner_tokens=reasoner_tokens,
                step_dt_s=dt_seq[:, step_idx],
            )
            state = replace(
                state,
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
                    [
                        ego_out["pose_delta"],
                        ego_out["kinematics"],
                        ego_out["logvar_pose"],
                    ],
                    dim=-1,
                ),
                reasoner_ttl=reasoner_ttl,
            )
            per_step.append(
                {
                    "ego_tokens": world_view.ego_tokens,
                    "pose_delta": ego_out["pose_delta"],
                    "kinematics": ego_out["kinematics"],
                    "logvar_pose": ego_out["logvar_pose"],
                    "static_grid": world_view.static_grid,
                    "dynamic_slots": world_view.dynamic_slots,
                    "speculative_slots": world_view.speculative_slots,
                    "dynamic_slot_alive": world_view.dynamic_slot_alive,
                    "speculative_slot_alive": world_view.speculative_slot_alive,
                    "lane_slots": world_view.lane_slots,
                    "map_elem_slots": world_view.map_elem_slots,
                    "route_tokens": world_view.route_tokens,
                    "reasoner_tokens": world_view.reasoner_tokens,
                }
            )
        return {
            "seq": self._stack_step_dicts(per_step),
            "last": per_step[-1] if per_step else {},
            "final_state": state,
        }

    def _advance_temporal_caches(
        self,
        *,
        state: AtlasState,
        cam_tokens_cur: torch.Tensor,
        dt_t: torch.Tensor,
        frame_summary: torch.Tensor,
        dt_buffer: torch.Tensor,
        insert_mid_summary: bool = True,
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
        if insert_mid_summary and (state.step_index + 1) % stride == 0:
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

        rgb_model = self._move_rgb_to_model_device(rgb_t)
        rgb_pad = pad_image_to_size(
            rgb_model,
            self.cfg.image.padded_height,
            self.cfg.image.padded_width,
        )
        vision_features, cam_tokens_cur = self.vision(rgb_model[:, None], pad_to_native=True)
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
                rgb_model.device,
                rgb_model.dtype,
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

        if stage == "stage1a":
            outputs = self.forward_stage1a(
                rgb_recent=rgb_recent,
                dt_recent=dt_recent,
                rgb_older=rgb_older,
                dt_older=dt_older,
                rgb_mid=rgb_mid,
                dt_mid=dt_mid,
                init_state=init_state,
            )
            outputs["stage"] = stage
            outputs["privileged"] = privileged or {}
            return outputs
        if stage == "stage1b":
            outputs = self.forward_stage1b(
                rgb_recent=rgb_recent,
                dt_recent=dt_recent,
                rgb_older=rgb_older,
                dt_older=dt_older,
                rgb_mid=rgb_mid,
                dt_mid=dt_mid,
                actions_hist=actions_hist,
                dt_hist=dt_hist,
                init_state=init_state,
            )
            outputs["stage"] = stage
            outputs["privileged"] = privileged or {}
            return outputs
        if stage == "stage1c":
            outputs = self.forward_stage1c(
                rgb_recent=rgb_recent,
                dt_recent=dt_recent,
                rgb_older=rgb_older,
                dt_older=dt_older,
                rgb_mid=rgb_mid,
                dt_mid=dt_mid,
                actions_hist=actions_hist,
                dt_hist=dt_hist,
                route_polyline=route_polyline,
                nav_cmd=nav_cmd,
                reasoner_tok=reasoner_tok,
                init_state=init_state,
            )
            outputs["stage"] = stage
            outputs["privileged"] = privileged or {}
            return outputs

        batch_size, recent_steps, _, _, _ = rgb_recent.shape
        if dt_recent.shape[:2] != rgb_recent.shape[:2]:
            raise ValueError("dt_recent must align with rgb_recent exactly.")
        if dt_older.shape[:2] != rgb_older.shape[:2]:
            raise ValueError("dt_older must align with rgb_older exactly.")
        if dt_mid.shape[:2] != rgb_mid.shape[:2]:
            raise ValueError("dt_mid must align with rgb_mid exactly.")
        if dt_hist.shape[:2] != actions_hist.shape[:2]:
            raise ValueError("dt_hist must align with actions_hist exactly.")
        state = self._seed_stage_state(
            rgb_recent=rgb_recent,
            rgb_older=rgb_older,
            dt_older=dt_older,
            rgb_mid=rgb_mid,
            dt_mid=dt_mid,
            actions_hist=actions_hist,
            dt_hist=dt_hist,
            init_state=init_state,
            seed_action_history=True,
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
