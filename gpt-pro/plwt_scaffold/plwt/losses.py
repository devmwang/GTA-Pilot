from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

import torch
import torch.nn.functional as F


def _zero_like(outputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
    for value in outputs.values():
        if isinstance(value, torch.Tensor):
            return value.new_zeros(())
    return torch.tensor(0.0)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is None:
        return F.mse_loss(pred, target)
    diff = (pred - target).pow(2)
    while mask.ndim < diff.ndim:
        mask = mask.unsqueeze(-1)
    denom = mask.sum().clamp(min=1.0)
    return (diff * mask).sum() / denom


def sigreg_loss(x: torch.Tensor) -> torch.Tensor:
    # Lightweight Gaussian regularizer inspired by LeWM; aggregate over batch and token axes.
    if x.ndim < 2:
        raise ValueError("sigreg input must have at least two dims")
    feat = x.reshape(-1, x.shape[-1])
    mu = feat.mean(dim=0)
    xc = feat - mu
    cov = (xc.T @ xc) / max(feat.shape[0] - 1, 1)
    eye = torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
    return mu.pow(2).mean() + (cov - eye).pow(2).mean()


def gaussian_nll(pred: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    inv_var = torch.exp(-logvar)
    return 0.5 * ((pred - target).pow(2) * inv_var + logvar).mean()


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    target = target.float()
    inter = (probs * target).sum()
    denom = probs.sum() + target.sum() + eps
    return 1.0 - (2.0 * inter + eps) / denom


def planner_minade_loss(traj: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # traj [B, K, T, 4], target [B, T, 4]
    ade = torch.norm(traj[..., :2] - target[:, None, :, :2], dim=-1).mean(dim=-1)
    return ade.min(dim=1).values.mean()


def planner_fde_loss(traj: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    fde = torch.norm(traj[:, :, -1, :2] - target[:, None, -1, :2], dim=-1)
    return fde.min(dim=1).values.mean()


def planner_ctrl_loss(curvature: torch.Tensor, speed: torch.Tensor, target_curv: torch.Tensor | None, target_speed: torch.Tensor | None) -> torch.Tensor:
    loss = curvature.new_zeros(())
    if target_curv is not None:
        loss = loss + F.l1_loss(curvature, target_curv)
    if target_speed is not None:
        loss = loss + F.l1_loss(speed, target_speed)
    return loss


def planner_rank_loss(score: torch.Tensor, teacher_cost: torch.Tensor) -> torch.Tensor:
    # Pairwise logistic ranking; lower teacher cost should rank higher.
    b, k = score.shape
    diff_s = score[:, :, None] - score[:, None, :]
    diff_c = teacher_cost[:, None, :] - teacher_cost[:, :, None]
    label = torch.sign(diff_c)
    pair_loss = F.softplus(-diff_s * label)
    mask = 1.0 - torch.eye(k, device=score.device, dtype=score.dtype)[None]
    return (pair_loss * mask).sum() / mask.sum().clamp(min=1.0)


@dataclass
class StageWeights:
    weights: Dict[str, float]

    def get(self, name: str, default: float = 0.0) -> float:
        return self.weights.get(name, default)


def stage_weight_preset(stage: str) -> StageWeights:
    presets = {
        "stage1a": {"cam_jepa": 1.0, "sigreg": 0.05},
        "stage1b": {"cam_jepa": 0.3, "sigreg": 0.05, "depth": 1.0, "track": 0.5, "ego": 0.5},
        "stage1c": {"world_jepa": 1.0, "sigreg": 0.05, "mem": 0.5, "ego": 0.5},
        "stage2": {"occ_state": 2.0, "occ_sem": 1.0, "bev": 1.0, "lane": 1.0, "map": 0.4, "actors": 0.4, "ego": 0.3, "provenance": 0.2, "world_jepa": 0.3, "sigreg": 0.02},
        "stage3": {"minade": 1.0, "fde": 0.7, "rank": 1.0, "ctrl": 0.4, "future": 0.4, "occ_state": 0.3, "lane": 0.3, "ego": 0.2, "world_jepa": 0.1},
        "stage4": {"minade": 1.5, "rank": 1.0, "ctrl": 0.3, "future": 0.3, "occ_state": 0.35, "occ_sem": 0.25, "lane": 0.30, "map": 0.10, "actors": 0.15, "ego": 0.15, "provenance": 0.10, "world_jepa": 0.10, "sigreg": 0.01},
    }
    return StageWeights(presets[stage])


def compute_stage_losses(
    stage: str,
    outputs: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    weights: StageWeights | None = None,
) -> dict[str, torch.Tensor]:
    weights = weights or stage_weight_preset(stage)
    z = _zero_like(outputs)
    losses: dict[str, torch.Tensor] = {}

    if "cam_projector_pred" in outputs and "cam_projector_target" in targets:
        losses["cam_jepa"] = masked_mse(outputs["cam_projector_pred"], targets["cam_projector_target"], targets.get("cam_mask"))
    if "world_projector_pred" in outputs and "world_projector_target" in targets:
        losses["world_jepa"] = masked_mse(outputs["world_projector_pred"], targets["world_projector_target"], targets.get("world_mask"))

    sig_terms = []
    for key in ("cam_projector_pred", "world_projector_pred", "proposal_embed"):
        if key in outputs:
            sig_terms.append(sigreg_loss(outputs[key]))
    if sig_terms:
        losses["sigreg"] = torch.stack(sig_terms).mean()

    if "depth_logits" in outputs and "depth_target" in targets:
        depth_bins = outputs["depth_logits"].shape[1]
        tgt = targets["depth_target"].long().clamp(min=0, max=depth_bins - 1)
        losses["depth"] = F.cross_entropy(outputs["depth_logits"], tgt)
    if "track_offsets" in outputs and "track_target" in targets:
        losses["track"] = F.l1_loss(outputs["track_offsets"], targets["track_target"])
    if "ego_out" in outputs and "ego_target" in targets:
        logvar = outputs.get("ego_logvar", torch.zeros_like(outputs["ego_out"]))
        losses["ego"] = gaussian_nll(outputs["ego_out"], logvar, targets["ego_target"])
    if "mem_target" in targets and "static_grid" in outputs:
        losses["mem"] = F.l1_loss(outputs["static_grid"], targets["mem_target"])

    if "occ_state" in outputs and "occ_state_target" in targets:
        losses["occ_state"] = F.cross_entropy(outputs["occ_state"], targets["occ_state_target"].long())
    if "occ_sem" in outputs and "occ_sem_target" in targets:
        losses["occ_sem"] = F.cross_entropy(outputs["occ_sem"], targets["occ_sem_target"].long())
    if "bev_lite" in outputs and "bev_target" in targets:
        losses["bev"] = F.binary_cross_entropy_with_logits(outputs["bev_lite"], targets["bev_target"].float())
    if "provenance" in outputs and "provenance_target" in targets:
        losses["provenance"] = F.cross_entropy(outputs["provenance"], targets["provenance_target"].long())

    if "centerline" in outputs and "lane_centerline_target" in targets:
        losses["lane"] = F.l1_loss(outputs["centerline"], targets["lane_centerline_target"])
    if "map_poly" in outputs and "map_poly_target" in targets:
        losses["map"] = F.l1_loss(outputs["map_poly"], targets["map_poly_target"])
    if "actor_box" in outputs and "actor_box_target" in targets:
        losses["actors"] = F.l1_loss(outputs["actor_box"], targets["actor_box_target"])

    if "traj" in outputs and "traj_target" in targets:
        losses["minade"] = planner_minade_loss(outputs["traj"], targets["traj_target"])
        losses["fde"] = planner_fde_loss(outputs["traj"], targets["traj_target"])
    if "score" in outputs and "teacher_cost" in targets:
        losses["rank"] = planner_rank_loss(outputs["score"], targets["teacher_cost"])
    if "curvature" in outputs:
        losses["ctrl"] = planner_ctrl_loss(outputs["curvature"], outputs["speed"], targets.get("curvature_target"), targets.get("speed_target"))
    if "future_dyn" in outputs and "future_dyn_target" in targets:
        losses["future"] = F.mse_loss(outputs["future_dyn"], targets["future_dyn_target"])

    total = z
    for name, loss in losses.items():
        total = total + weights.get(name) * loss
    losses["total"] = total
    return losses
