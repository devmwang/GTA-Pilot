from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from .config import AtlasCruiseConfig


@dataclass(slots=True)
class AtlasCruiseState:
    frame_tokens: torch.Tensor
    frame_summary: torch.Tensor
    frame_timestamps_ns: torch.Tensor
    actions_hist: torch.Tensor
    dt_hist: torch.Tensor
    step_index: int = 0

    @classmethod
    def init_empty(
        cls,
        cfg: AtlasCruiseConfig,
        batch_size: int,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "AtlasCruiseState":
        return cls(
            frame_tokens=torch.zeros(
                batch_size,
                cfg.num_visual_frames,
                cfg.tokens_per_frame,
                cfg.hidden_dim,
                device=device,
                dtype=dtype,
            ),
            frame_summary=torch.zeros(
                batch_size,
                cfg.num_visual_frames,
                cfg.hidden_dim,
                device=device,
                dtype=dtype,
            ),
            frame_timestamps_ns=torch.zeros(
                batch_size,
                cfg.num_visual_frames,
                device=device,
                dtype=torch.long,
            ),
            actions_hist=torch.zeros(
                batch_size,
                cfg.action_history_steps,
                cfg.action_dim,
                device=device,
                dtype=dtype,
            ),
            dt_hist=torch.zeros(
                batch_size,
                cfg.action_history_steps,
                1,
                device=device,
                dtype=dtype,
            ),
            step_index=0,
        )

    def with_appended_frame(
        self,
        *,
        frame_tokens: torch.Tensor,
        frame_summary: torch.Tensor,
        timestamp_ns: torch.Tensor,
    ) -> "AtlasCruiseState":
        return replace(
            self,
            frame_tokens=torch.cat([self.frame_tokens[:, 1:], frame_tokens[:, None]], dim=1).detach(),
            frame_summary=torch.cat([self.frame_summary[:, 1:], frame_summary[:, None]], dim=1).detach(),
            frame_timestamps_ns=torch.cat([self.frame_timestamps_ns[:, 1:], timestamp_ns[:, None]], dim=1).detach(),
            step_index=self.step_index + 1,
        )

    def with_action_history(
        self,
        *,
        actions_hist: torch.Tensor,
        dt_hist: torch.Tensor,
    ) -> "AtlasCruiseState":
        return replace(
            self,
            actions_hist=actions_hist.detach(),
            dt_hist=dt_hist.detach(),
        )
