from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .config import AtlasCruiseConfig
from .trajectory_legalizer import estimate_curvature


@dataclass(slots=True)
class AtlasCruiseLossWeights:
    traj_pos: float = 3.0
    traj_yaw: float = 0.5
    traj_speed: float = 1.0
    ego: float = 0.5
    slow: float = 0.5
    fallback: float = 0.2
    control: float = 0.3
    smooth: float = 0.2
    latacc: float = 0.2
    confidence: float = 0.1


def compute_atlas_cruise_losses(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    cfg: AtlasCruiseConfig,
    weights: AtlasCruiseLossWeights | None = None,
) -> dict[str, torch.Tensor]:
    w = weights or AtlasCruiseLossWeights()
    pred = outputs["traj"]
    target = targets["target_traj"].to(device=pred.device, dtype=pred.dtype)
    valid = targets["target_traj_valid"].to(device=pred.device).bool()
    if pred.shape != target.shape:
        raise ValueError(f"traj shape mismatch: pred={tuple(pred.shape)} target={tuple(target.shape)}")
    valid_f = valid.to(dtype=pred.dtype)
    denom = valid_f.sum().clamp_min(1.0)
    pos_loss = (F.huber_loss(pred[..., :2], target[..., :2], reduction="none").sum(dim=-1) * valid_f).sum() / denom
    yaw_delta = _wrap_angle_tensor(pred[..., 2] - target[..., 2])
    yaw_loss = (F.huber_loss(yaw_delta, torch.zeros_like(yaw_delta), reduction="none") * valid_f).sum() / denom
    speed_loss = (F.huber_loss(pred[..., 3], target[..., 3], reduction="none") * valid_f).sum() / denom
    ego_loss = F.huber_loss(
        outputs["ego_kinematics"],
        targets["target_ego"].to(device=pred.device, dtype=pred.dtype),
    )
    slow_loss = F.binary_cross_entropy_with_logits(
        outputs["slow_or_brake_logit"],
        targets["target_slow_or_brake"].to(device=pred.device, dtype=pred.dtype),
    )
    fallback_loss = F.binary_cross_entropy_with_logits(
        outputs["fallback_logit"],
        targets["target_fallback"].to(device=pred.device, dtype=pred.dtype),
    )
    conf_target = (valid_f.mean(dim=1, keepdim=True) >= 0.8).to(dtype=pred.dtype)
    confidence_loss = F.binary_cross_entropy_with_logits(outputs["traj_conf_logit"], conf_target)
    control_loss = torch.zeros((), device=pred.device, dtype=pred.dtype)
    if "control_aux" in outputs and "target_control_aux" in targets:
        control_loss = F.huber_loss(
            outputs["control_aux"],
            targets["target_control_aux"].to(device=pred.device, dtype=pred.dtype),
        )
    smooth_loss = _smoothness_loss(pred)
    latacc_loss = _latacc_penalty(pred, cfg.comfort_lat_acc_mps2)
    total = (
        w.traj_pos * pos_loss
        + w.traj_yaw * yaw_loss
        + w.traj_speed * speed_loss
        + w.ego * ego_loss
        + w.slow * slow_loss
        + w.fallback * fallback_loss
        + w.control * control_loss
        + w.smooth * smooth_loss
        + w.latacc * latacc_loss
        + w.confidence * confidence_loss
    )
    return {
        "traj_pos": pos_loss,
        "traj_yaw": yaw_loss,
        "traj_speed": speed_loss,
        "ego": ego_loss,
        "slow": slow_loss,
        "fallback": fallback_loss,
        "control": control_loss,
        "smooth": smooth_loss,
        "latacc": latacc_loss,
        "confidence": confidence_loss,
        "total": total,
    }


def _smoothness_loss(traj: torch.Tensor) -> torch.Tensor:
    if traj.shape[-2] < 3:
        return torch.zeros((), device=traj.device, dtype=traj.dtype)
    d2 = traj[..., 2:, :3] - 2.0 * traj[..., 1:-1, :3] + traj[..., :-2, :3]
    speed_d = torch.diff(traj[..., 3], dim=-1)
    return F.huber_loss(d2, torch.zeros_like(d2)) + 0.1 * F.huber_loss(speed_d, torch.zeros_like(speed_d))


def _wrap_angle_tensor(angle: torch.Tensor) -> torch.Tensor:
    return torch.remainder(angle + torch.pi, 2.0 * torch.pi) - torch.pi


def _latacc_penalty(traj: torch.Tensor, limit: float) -> torch.Tensor:
    curvature = estimate_curvature(traj)
    a_lat = traj[..., 3].clamp_min(0.0).pow(2) * curvature.abs()
    return F.relu(a_lat - float(limit)).pow(2).mean()
