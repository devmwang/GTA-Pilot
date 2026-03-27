from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ..config import AtlasConfig
from ..model import Atlas
from ..utils import LearnedQueryPool


class _PrivilegedTokenAdapter(nn.Module):
    def __init__(self, input_dim: int, output_tokens: int, d_model: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.pool = LearnedQueryPool(output_tokens, d_model, heads=max(1, d_model // 64))

    def forward(
        self,
        tensor: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if tensor is None:
            return torch.zeros(
                batch_size,
                0,
                self.proj[-1].out_features,
                device=device,
                dtype=dtype,
            )
        if tensor.ndim == 2:
            tensor = tensor[:, None, :]
        embedded = self.proj(tensor.to(dtype))
        return self.pool(embedded)


class PrivilegedTeacherModel(Atlas):
    def __init__(self, cfg: AtlasConfig):
        super().__init__(cfg)
        d_model = cfg.hidden_dim
        self.lidar_adapter = _PrivilegedTokenAdapter(d_model, 96, d_model)
        self.pose_adapter = _PrivilegedTokenAdapter(3, 4, d_model)
        self.actor_adapter = _PrivilegedTokenAdapter(d_model, 32, d_model)
        self.map_adapter = _PrivilegedTokenAdapter(d_model, 16, d_model)
        self.hidden_actor_adapter = _PrivilegedTokenAdapter(d_model, 32, d_model)
        self.visibility_adapter = _PrivilegedTokenAdapter(4, 8, d_model)
        self.flow_adapter = _PrivilegedTokenAdapter(2, 16, d_model)
        self.risk_adapter = _PrivilegedTokenAdapter(2, 16, d_model)

    def encode_privileged_tokens(
        self,
        privileged: dict[str, Any] | None,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        privileged = privileged or {}
        return {
            "priv_lidar_tokens": self.lidar_adapter(
                privileged.get("lidar_tokens"), batch_size, device, dtype
            ),
            "priv_pose_tokens": self.pose_adapter(
                privileged.get("pose_tokens"), batch_size, device, dtype
            ),
            "priv_actor_tokens": self.actor_adapter(
                privileged.get("actor_tokens"), batch_size, device, dtype
            ),
            "priv_map_tokens": self.map_adapter(
                privileged.get("map_tokens"), batch_size, device, dtype
            ),
            "priv_hidden_actor_tokens": self.hidden_actor_adapter(
                privileged.get("hidden_actor_tokens"), batch_size, device, dtype
            ),
            "priv_visibility_tokens": self.visibility_adapter(
                privileged.get("visibility_tokens"), batch_size, device, dtype
            ),
            "priv_flow_tokens": self.flow_adapter(
                privileged.get("flow_tokens"), batch_size, device, dtype
            ),
            "priv_risk_tokens": self.risk_adapter(
                privileged.get("risk_tokens"), batch_size, device, dtype
            ),
        }

    @staticmethod
    def _teacher_supervision(
        privileged: dict[str, Any] | None,
    ) -> dict[str, Any]:
        privileged = privileged or {}
        return {
            "hidden_actor_trajs": privileged.get("hidden_actor_trajs"),
            "visibility_mask": privileged.get("visibility_mask"),
            "occupancy_flow": privileged.get("occupancy_flow"),
            "speculative_heatmap": privileged.get("speculative_heatmap"),
            "actor_existence": privileged.get("actor_existence"),
            "occluder_risk": privileged.get("occluder_risk"),
        }

    def _attach_teacher_context(
        self,
        outputs: dict[str, Any],
        privileged: dict[str, Any] | None,
    ) -> dict[str, Any]:
        batch_size = outputs["final_state"].static_grid.shape[0]
        device = outputs["final_state"].static_grid.device
        dtype = outputs["final_state"].static_grid.dtype
        outputs["teacher_privileged_tokens"] = self.encode_privileged_tokens(
            privileged,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )
        outputs["teacher_privileged_supervision"] = self._teacher_supervision(privileged)
        return outputs

    def forward_stage1a_teacher(
        self,
        *args: Any,
        privileged: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        outputs = super().forward_stage1a(*args, **kwargs)
        outputs = self._attach_teacher_context(outputs, privileged)
        outputs["teacher_distill_bundle"] = {
            "cam_now_target": outputs["seq"]["cam_now"].detach().clone(),
            "frame_summary_target": outputs["seq"]["frame_summary"].detach().clone(),
        }
        return outputs

    def forward_stage1b_teacher(
        self,
        *args: Any,
        privileged: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        outputs = super().forward_stage1b(*args, **kwargs)
        outputs = self._attach_teacher_context(outputs, privileged)
        outputs["teacher_distill_bundle"] = {
            "cam_now_target": outputs["seq"]["cam_now"].detach().clone(),
            "frame_summary_target": outputs["seq"]["frame_summary"].detach().clone(),
            "pose_delta_target": outputs["seq"]["pose_delta"].detach().clone(),
            "kinematics_target": outputs["seq"]["kinematics"].detach().clone(),
            "depth_target": outputs["seq"]["depth_mean"].detach().clone(),
            "track_target": outputs["seq"]["track_offsets"].detach().clone(),
        }
        return outputs

    def forward_stage1c_teacher(
        self,
        *args: Any,
        privileged: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        outputs = super().forward_stage1c(*args, **kwargs)
        outputs = self._attach_teacher_context(outputs, privileged)
        outputs["teacher_distill_bundle"] = {
            "static_grid_target": outputs["seq"]["static_grid"].detach().clone(),
            "dynamic_slots_target": outputs["seq"]["dynamic_slots"].detach().clone(),
            "speculative_slots_target": outputs["seq"]["speculative_slots"].detach().clone(),
            "dynamic_slot_alive_target": outputs["seq"]["dynamic_slot_alive"].detach().clone(),
            "speculative_slot_alive_target": outputs["seq"]["speculative_slot_alive"].detach().clone(),
            "lane_slots_target": outputs["seq"]["lane_slots"].detach().clone(),
            "map_elem_slots_target": outputs["seq"]["map_elem_slots"].detach().clone(),
            "route_tokens_target": outputs["seq"]["route_tokens"].detach().clone(),
            "reasoner_tokens_target": outputs["seq"]["reasoner_tokens"].detach().clone(),
            "ego_tokens_target": outputs["seq"]["ego_tokens"].detach().clone(),
            **outputs["teacher_privileged_supervision"],
        }
        return outputs

    def forward_stage2plus_teacher(
        self,
        rgb_recent: torch.Tensor,
        dt_recent: torch.Tensor,
        rgb_older: torch.Tensor,
        dt_older: torch.Tensor,
        rgb_mid: torch.Tensor,
        dt_mid: torch.Tensor,
        actions_hist: torch.Tensor,
        dt_hist: torch.Tensor,
        route_polyline: torch.Tensor | None = None,
        nav_cmd: torch.Tensor | None = None,
        reasoner_tok: torch.Tensor | None = None,
        privileged: dict[str, Any] | None = None,
        init_state=None,
        stage: str = "stage2",
        mode: str = "inspect",
    ) -> dict[str, Any]:
        if stage in {"stage1a", "stage1b", "stage1c"}:
            raise RuntimeError(
                "Stage 1 teacher hard cut: use forward_stage1a_teacher(), "
                "forward_stage1b_teacher(), or forward_stage1c_teacher() directly."
            )

        outputs = super().forward_train(
            rgb_recent,
            dt_recent,
            rgb_older,
            dt_older,
            rgb_mid,
            dt_mid,
            actions_hist,
            dt_hist,
            route_polyline=route_polyline,
            nav_cmd=nav_cmd,
            reasoner_tok=reasoner_tok,
            privileged=privileged,
            init_state=init_state,
            stage=stage,
            mode=mode,
        )
        outputs = self._attach_teacher_context(outputs, privileged)
        last = outputs["last"]
        outputs["teacher_distill_bundle"] = {}
        for key in (
            "dynamic_slots",
            "speculative_slots",
            "dyn_flow_bev",
            "occl_risk_bev",
            "provenance",
            "future_spec",
            "hidden_risk_penalty",
        ):
            if key in last:
                outputs["teacher_distill_bundle"][f"{key}_target"] = last[key].detach().clone()
        outputs["teacher_distill_bundle"].update(outputs["teacher_privileged_supervision"])
        return outputs

    def forward_train(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        raise RuntimeError(
            "PrivilegedTeacherModel forward_train() is not part of the public teacher API. "
            "Use forward_stage1a_teacher(), forward_stage1b_teacher(), forward_stage1c_teacher(), "
            "or forward_stage2plus_teacher() explicitly."
        )
