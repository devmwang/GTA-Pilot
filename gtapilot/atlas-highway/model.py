from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn as nn
import torch.nn.functional as F

from .action_encoder import HighwayActionEncoder
from .backbone import build_regnet_backbone
from .bifpn import HighwayBiFPN
from .config import AtlasHAConfig
from .heads import AtlasHAHeads
from .state import ActionRingBuffer, AtlasHAState, DtRingBuffer
from .temporal import FullContextReader, HeadFusionReader, SummaryContextReader
from .tokenizer import FixedGridVisualTokenizer


class AtlasHA(nn.Module):
    def __init__(self, cfg: AtlasHAConfig | None = None):
        super().__init__()
        self.cfg = cfg or AtlasHAConfig()
        self.cfg.validate()
        self.backbone = build_regnet_backbone(self.cfg)
        self.bifpn = HighwayBiFPN(self.cfg, self.backbone.out_channels)
        self.tokenizer = FixedGridVisualTokenizer(self.cfg)
        self.full_context_reader = FullContextReader(self.cfg)
        self.summary_context_reader = SummaryContextReader(self.cfg)
        self.action_encoder = HighwayActionEncoder(self.cfg)
        self.nav_proj = nn.Sequential(
            nn.Linear(self.cfg.nav_cmd_dim, self.cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(self.cfg.hidden_dim, self.cfg.hidden_dim),
        )
        self.head_fusion = HeadFusionReader(self.cfg, query_count=8)
        self.heads = AtlasHAHeads(self.cfg)

    def init_state(
        self,
        batch_size: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> AtlasHAState:
        return AtlasHAState.init_empty(self.cfg, batch_size, device=device, dtype=dtype)

    def _model_device_dtype(self) -> tuple[torch.device, torch.dtype]:
        param = next(self.parameters())
        return param.device, param.dtype

    def _prepare_rgb(self, scene_rgb: torch.Tensor, ui_mask: torch.Tensor | None) -> torch.Tensor:
        device, dtype = self._model_device_dtype()
        x = scene_rgb.to(device=device, non_blocking=device.type == "cuda")
        if x.dtype == torch.uint8:
            x = x.float().mul_(1.0 / 255.0)
        elif not torch.is_floating_point(x):
            x = x.float()
        x = x.to(dtype=dtype)
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError("scene_rgb must have shape [B, 3, H, W].")
        if ui_mask is not None:
            mask = ui_mask.to(device=device, dtype=dtype)
            if mask.ndim != 4 or mask.shape[1] != 1:
                raise ValueError("ui_mask must have shape [B, 1, H, W].")
            if mask.shape[-2:] != x.shape[-2:]:
                mask = F.interpolate(mask, size=x.shape[-2:], mode="nearest")
            x = x * (1.0 - mask.clamp(0.0, 1.0)) + 0.5 * mask.clamp(0.0, 1.0)
        if x.shape[-2:] != (self.cfg.input_h, self.cfg.input_w):
            x = F.interpolate(
                x,
                size=(self.cfg.input_h, self.cfg.input_w),
                mode="bilinear",
                align_corners=False,
            )
        return x

    def encode_scene(
        self,
        scene_rgb: torch.Tensor,
        ui_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        x = self._prepare_rgb(scene_rgb, ui_mask)
        features = self.backbone(x)
        pyramid = self.bifpn(features)
        source_tokens, frame_tokens, frame_summary = self.tokenizer(pyramid)
        debug = {
            **features,
            **pyramid,
            "source_tokens": source_tokens,
        }
        return frame_tokens, frame_summary, debug

    @staticmethod
    def _append_keep_last(
        history: torch.Tensor,
        new_value: torch.Tensor,
        keep: int,
    ) -> torch.Tensor:
        cat = torch.cat([history, new_value], dim=1)
        return cat[:, -keep:]

    def _build_read_caches(
        self,
        state: AtlasHAState,
        frame_tokens: torch.Tensor,
        frame_summary: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        batch_size = frame_tokens.shape[0]
        full_cache = self._append_keep_last(
            state.full_token_cache,
            frame_tokens[:, None],
            self.cfg.full_context_frames,
        )
        full_valid_new = torch.ones(
            batch_size,
            1,
            device=frame_tokens.device,
            dtype=torch.bool,
        )
        full_valid = self._append_keep_last(
            state.full_valid,
            full_valid_new,
            self.cfg.full_context_frames,
        )

        insert_summary = (
            (state.step_index + 1) % self.cfg.summary_stride_steps == 0
            or not bool(state.summary_valid.any().item())
        )
        if insert_summary:
            summary_cache = self._append_keep_last(
                state.summary_cache,
                frame_summary[:, None],
                self.cfg.summary_context_steps,
            )
            summary_valid = self._append_keep_last(
                state.summary_valid,
                full_valid_new,
                self.cfg.summary_context_steps,
            )
        else:
            summary_cache = state.summary_cache
            summary_valid = state.summary_valid
        return full_cache, full_valid, summary_cache, summary_valid, insert_summary

    def _next_state(
        self,
        state: AtlasHAState,
        *,
        full_cache: torch.Tensor,
        full_valid: torch.Tensor,
        summary_cache: torch.Tensor,
        summary_valid: torch.Tensor,
        actions_hist: torch.Tensor,
        dt_hist: torch.Tensor,
    ) -> AtlasHAState:
        action_ring = ActionRingBuffer(
            values=actions_hist.detach()[:, -self.cfg.action_history_steps :],
            valid=torch.ones(
                actions_hist.shape[0],
                min(actions_hist.shape[1], self.cfg.action_history_steps),
                device=actions_hist.device,
                dtype=torch.bool,
            ),
        )
        if action_ring.values.shape[1] < self.cfg.action_history_steps:
            pad_steps = self.cfg.action_history_steps - action_ring.values.shape[1]
            action_ring = ActionRingBuffer(
                values=torch.cat(
                    [
                        torch.zeros(
                            actions_hist.shape[0],
                            pad_steps,
                            self.cfg.action_dim,
                            device=actions_hist.device,
                            dtype=actions_hist.dtype,
                        ),
                        action_ring.values,
                    ],
                    dim=1,
                ),
                valid=torch.cat(
                    [
                        torch.zeros(
                            actions_hist.shape[0],
                            pad_steps,
                            device=actions_hist.device,
                            dtype=torch.bool,
                        ),
                        action_ring.valid,
                    ],
                    dim=1,
                ),
            )
        dt_ring = DtRingBuffer(
            values=dt_hist.detach()[:, -self.cfg.action_history_steps :],
            valid=action_ring.valid,
        )
        if dt_ring.values.shape[1] < self.cfg.action_history_steps:
            pad_steps = self.cfg.action_history_steps - dt_ring.values.shape[1]
            dt_ring = DtRingBuffer(
                values=torch.cat(
                    [
                        torch.zeros(
                            dt_hist.shape[0],
                            pad_steps,
                            1,
                            device=dt_hist.device,
                            dtype=dt_hist.dtype,
                        ),
                        dt_ring.values,
                    ],
                    dim=1,
                ),
                valid=action_ring.valid,
            )
        return replace(
            state,
            full_token_cache=full_cache.detach(),
            full_kv_cache=[],
            summary_cache=summary_cache.detach(),
            action_ring=action_ring,
            dt_ring=dt_ring,
            full_valid=full_valid.detach(),
            summary_valid=summary_valid.detach(),
            step_index=state.step_index + 1,
        )

    def forward(
        self,
        scene_rgb: torch.Tensor,
        actions_hist: torch.Tensor,
        dt_hist: torch.Tensor,
        ui_mask: torch.Tensor | None = None,
        nav_cmd: torch.Tensor | None = None,
        state: AtlasHAState | None = None,
    ) -> tuple[dict[str, torch.Tensor | None], AtlasHAState]:
        device, dtype = self._model_device_dtype()
        if actions_hist.ndim != 3 or actions_hist.shape[-1] != self.cfg.action_dim:
            raise ValueError("actions_hist must have shape [B, M, 6].")
        if dt_hist.ndim != 3 or dt_hist.shape[-1] != 1:
            raise ValueError("dt_hist must have shape [B, M, 1].")
        if actions_hist.shape[:2] != dt_hist.shape[:2]:
            raise ValueError("actions_hist and dt_hist must align.")

        batch_size = scene_rgb.shape[0]
        actions_hist = actions_hist.to(device=device, dtype=dtype)
        dt_hist = dt_hist.to(device=device, dtype=dtype)
        if state is None:
            state = self.init_state(batch_size, device=device, dtype=dtype)

        frame_tokens, frame_summary, debug = self.encode_scene(scene_rgb, ui_mask)
        full_cache, full_valid, summary_cache, summary_valid, _ = self._build_read_caches(
            state,
            frame_tokens,
            frame_summary,
        )
        full_ctx_tokens = self.full_context_reader(full_cache, full_valid)
        summary_ctx_tokens = self.summary_context_reader(summary_cache, summary_valid)
        action_valid = torch.ones(actions_hist.shape[:2], device=device, dtype=torch.bool)
        action_tokens, action_summary = self.action_encoder(
            actions_hist,
            dt_hist,
            action_valid,
        )

        context_tokens = [frame_tokens, full_ctx_tokens, summary_ctx_tokens, action_tokens]
        nav_token = None
        if self.cfg.use_command_conditioning:
            if nav_cmd is None:
                nav_cmd = torch.zeros(batch_size, self.cfg.nav_cmd_dim, device=device, dtype=dtype)
            else:
                nav_cmd = nav_cmd.to(device=device, dtype=dtype)
            nav_token = self.nav_proj(nav_cmd).unsqueeze(1)
            context_tokens.append(nav_token)
        head_tokens, head_context = self.head_fusion(torch.cat(context_tokens, dim=1))
        outputs = self.heads(head_context)
        outputs.update(
            {
                "current_frame_tokens": frame_tokens,
                "frame_summary": frame_summary,
                "full_ctx_tokens": full_ctx_tokens,
                "summary_ctx_tokens": summary_ctx_tokens,
                "action_tokens": action_tokens,
                "action_summary": action_summary,
                "head_tokens": head_tokens,
                "head_context": head_context,
                "nav_token": nav_token,
                **debug,
            }
        )
        next_state = self._next_state(
            state,
            full_cache=full_cache,
            full_valid=full_valid,
            summary_cache=summary_cache,
            summary_valid=summary_valid,
            actions_hist=actions_hist,
            dt_hist=dt_hist,
        )
        return outputs, next_state
