from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .action_encoder import AtlasCruiseActionEncoder
from .backbone import CruiseVisualTokenizer, build_cruise_backbone
from .config import AtlasCruiseConfig
from .heads import AtlasCruiseHeads
from .state import AtlasCruiseState
from .temporal import AtlasCruiseTemporalEncoder


class AtlasCruise(nn.Module):
    def __init__(self, cfg: AtlasCruiseConfig | None = None):
        super().__init__()
        self.cfg = cfg or AtlasCruiseConfig()
        self.cfg.validate()
        self.backbone = build_cruise_backbone(self.cfg)
        self.tokenizer = CruiseVisualTokenizer(self.cfg, self.backbone.out_channels)
        self.action_encoder = AtlasCruiseActionEncoder(self.cfg)
        self.temporal = AtlasCruiseTemporalEncoder(self.cfg)
        self.heads = AtlasCruiseHeads(self.cfg)

    def init_state(
        self,
        batch_size: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> AtlasCruiseState:
        return AtlasCruiseState.init_empty(self.cfg, batch_size, device=device, dtype=dtype)

    def _model_device_dtype(self) -> tuple[torch.device, torch.dtype]:
        param = next(self.parameters())
        return param.device, param.dtype

    def _prepare_rgb(self, rgb_recent: torch.Tensor, ui_mask_recent: torch.Tensor | None) -> torch.Tensor:
        device, dtype = self._model_device_dtype()
        if rgb_recent.ndim != 5 or rgb_recent.shape[2] != 3:
            raise ValueError("rgb_recent must have shape [B, T, 3, H, W].")
        if rgb_recent.shape[1] != self.cfg.num_visual_frames:
            raise ValueError(f"rgb_recent must have T={self.cfg.num_visual_frames}.")
        x = rgb_recent.to(device=device, non_blocking=device.type == "cuda")
        if x.dtype == torch.uint8:
            x = x.float().mul_(1.0 / 255.0)
        elif not torch.is_floating_point(x):
            x = x.float()
        x = x.to(dtype=dtype)
        if ui_mask_recent is not None:
            if ui_mask_recent.ndim != 5 or ui_mask_recent.shape[2] != 1:
                raise ValueError("ui_mask_recent must have shape [B, T, 1, H, W].")
            mask = ui_mask_recent.to(device=device, dtype=dtype)
            if mask.shape[-2:] != x.shape[-2:]:
                b, t = mask.shape[:2]
                mask = F.interpolate(
                    mask.reshape(b * t, 1, mask.shape[-2], mask.shape[-1]),
                    size=x.shape[-2:],
                    mode="nearest",
                ).view(b, t, 1, x.shape[-2], x.shape[-1])
            mask = mask.clamp(0.0, 1.0)
            x = x * (1.0 - mask) + 0.5 * mask
        if x.shape[-2:] != (self.cfg.input_h, self.cfg.input_w):
            b, t = x.shape[:2]
            x = F.interpolate(
                x.reshape(b * t, 3, x.shape[-2], x.shape[-1]),
                size=(self.cfg.input_h, self.cfg.input_w),
                mode="bilinear",
                align_corners=False,
            ).view(b, t, 3, self.cfg.input_h, self.cfg.input_w)
        return x

    def encode_frames(
        self,
        rgb_recent: torch.Tensor,
        ui_mask_recent: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self._prepare_rgb(rgb_recent, ui_mask_recent)
        b, t = x.shape[:2]
        features = self.backbone(x.reshape(b * t, 3, self.cfg.input_h, self.cfg.input_w))
        source_tokens, frame_tokens, frame_summary = self.tokenizer(features)
        return (
            source_tokens.view(b, t, source_tokens.shape[1], source_tokens.shape[2]),
            frame_tokens.view(b, t, self.cfg.tokens_per_frame, self.cfg.hidden_dim),
            frame_summary.view(b, t, self.cfg.hidden_dim),
        )

    def forward(
        self,
        rgb_recent: torch.Tensor,
        actions_hist: torch.Tensor,
        dt_hist: torch.Tensor,
        ui_mask_recent: torch.Tensor | None = None,
        frame_freshness: torch.Tensor | None = None,
        state: AtlasCruiseState | None = None,
    ) -> dict[str, torch.Tensor]:
        del state
        device, dtype = self._model_device_dtype()
        if actions_hist.ndim != 3 or actions_hist.shape[-1] != self.cfg.action_dim:
            raise ValueError("actions_hist must have shape [B, M, 6].")
        if dt_hist.ndim != 3 or dt_hist.shape[-1] != 1:
            raise ValueError("dt_hist must have shape [B, M, 1].")
        if actions_hist.shape[:2] != dt_hist.shape[:2]:
            raise ValueError("actions_hist and dt_hist must align.")
        _, frame_tokens, frame_summary = self.encode_frames(rgb_recent, ui_mask_recent)
        action_tokens = self.action_encoder(
            actions_hist.to(device=device, dtype=dtype),
            dt_hist.to(device=device, dtype=dtype),
        )
        freshness = None
        if frame_freshness is not None:
            freshness = frame_freshness.to(device=device, dtype=dtype)
        policy_context, temporal_tokens = self.temporal(frame_summary, action_tokens, freshness)
        outputs = self.heads(policy_context)
        outputs.update(
            {
                "frame_tokens": frame_tokens,
                "frame_summary": frame_summary,
                "action_tokens": action_tokens,
                "policy_context": policy_context,
                "temporal_tokens": temporal_tokens,
            }
        )
        return outputs
