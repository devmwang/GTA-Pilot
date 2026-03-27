from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

import torch
import torch.nn.functional as F


def _first_tensor(mapping: Mapping[str, torch.Tensor]) -> torch.Tensor:
    for value in mapping.values():
        if isinstance(value, torch.Tensor):
            return value
    return torch.tensor(0.0)


def _expand_mask(mask: torch.Tensor, target_rank: int) -> torch.Tensor:
    while mask.ndim < target_rank:
        mask = mask.unsqueeze(-1)
    return mask


def _broadcast_mask(mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return _expand_mask(mask, target.ndim).expand_as(target)


def masked_l2_jepa(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    if mask is None:
        return F.mse_loss(pred, target)
    mask = _broadcast_mask(mask, pred).to(pred.dtype)
    loss = (pred - target).pow(2) * mask
    return loss.sum() / mask.sum().clamp(min=1.0)


def sigreg(latents: torch.Tensor, family_mask: torch.Tensor | None = None) -> torch.Tensor:
    if family_mask is not None:
        family_mask = _expand_mask(family_mask, latents.ndim).to(latents.dtype)
        latents = latents * family_mask
    flat = latents.reshape(-1, latents.shape[-1])
    mu = flat.mean(dim=0)
    centered = flat - mu
    cov = (centered.T @ centered) / max(flat.shape[0] - 1, 1)
    eye = torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
    return mu.pow(2).mean() + (cov - eye).pow(2).mean()


def silog_depth(
    pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor | None = None
) -> torch.Tensor:
    if valid is None:
        valid = torch.ones_like(target, dtype=torch.bool)
    elif valid.ndim == pred.ndim - 1:
        valid = valid.unsqueeze(2)
    pred = pred.clamp(min=1e-4)
    target = target.clamp(min=1e-4)
    log_diff = (pred.log() - target.log())[valid]
    if log_diff.numel() == 0:
        return pred.new_zeros(())
    return torch.sqrt(log_diff.pow(2).mean() - 0.15 * log_diff.mean().pow(2) + 1e-6)


def track_epe(
    pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor | None = None
) -> torch.Tensor:
    if pred.ndim == 6:
        epe = torch.linalg.norm(pred - target, dim=3)
    elif pred.ndim == 5:
        epe = torch.linalg.norm(pred - target, dim=-3)
    else:
        epe = torch.linalg.norm(pred - target, dim=-1)
    if valid is None:
        return epe.mean()
    valid = valid.to(epe.dtype)
    return (epe * valid).sum() / valid.sum().clamp(min=1.0)


def gaussian_nll(
    pred_mean: torch.Tensor,
    pred_logvar: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    loss = 0.5 * ((pred_mean - target).pow(2) * torch.exp(-pred_logvar) + pred_logvar)
    if valid is None:
        return loss.mean()
    valid = _broadcast_mask(valid, loss).to(loss.dtype)
    loss = loss * valid
    return loss.sum() / valid.sum().clamp(min=1.0)


def focal_ce_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor | None = None,
    gamma: float = 2.0,
) -> torch.Tensor:
    ce = F.cross_entropy(logits, target.long(), reduction="none")
    pt = torch.exp(-ce)
    loss = ((1 - pt) ** gamma) * ce
    if valid is None:
        return loss.mean()
    valid = valid.to(loss.dtype)
    return (loss * valid).sum() / valid.sum().clamp(min=1.0)


def dice_loss(
    logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor | None = None
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    target = target.float()
    if valid is not None:
        valid = _expand_mask(valid, probs.ndim).to(probs.dtype)
        probs = probs * valid
        target = target * valid
    inter = (probs * target).sum()
    denom = probs.sum() + target.sum() + 1e-6
    return 1.0 - (2.0 * inter + 1e-6) / denom


def lovasz_logits(
    logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor | None = None
) -> torch.Tensor:
    # Lightweight surrogate for scaffold use.
    probs = torch.softmax(logits, dim=1)
    target_oh = F.one_hot(target.long(), num_classes=logits.shape[1]).movedim(-1, 1)
    target_oh = target_oh.to(probs.dtype)
    if valid is not None:
        valid = _expand_mask(valid, probs.ndim).to(probs.dtype)
        probs = probs * valid
        target_oh = target_oh * valid
    return (probs - target_oh).abs().mean()


def hungarian_lane_loss(
    pred: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor],
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    loss = pred["centerline"].new_zeros(())
    if "centerline" in target:
        loss = loss + F.l1_loss(pred["centerline"], target["centerline"])
    if "left_boundary" in target:
        loss = loss + F.l1_loss(pred["left_boundary"], target["left_boundary"])
    if "right_boundary" in target:
        loss = loss + F.l1_loss(pred["right_boundary"], target["right_boundary"])
    if "lane_sem_cls" in target:
        loss = loss + F.cross_entropy(
            pred["lane_sem_cls"].transpose(1, 2), target["lane_sem_cls"].long()
        )
    return loss


def hungarian_actor_loss(
    pred: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor],
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    loss = pred["actor_box"].new_zeros(())
    if "actor_box" in target:
        loss = loss + F.l1_loss(pred["actor_box"], target["actor_box"])
    if "actor_vel" in target:
        loss = loss + F.l1_loss(pred["actor_vel"], target["actor_vel"])
    if "actor_cls" in target:
        loss = loss + F.cross_entropy(
            pred["actor_cls"].transpose(1, 2), target["actor_cls"].long()
        )
    return loss


def min_ade_fde(
    pred_set: torch.Tensor, target: torch.Tensor, valid: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    ade = torch.linalg.norm(pred_set[..., :2] - target[:, None, :, :2], dim=-1).mean(dim=-1)
    fde = torch.linalg.norm(pred_set[:, :, -1, :2] - target[:, None, -1, :2], dim=-1)
    return ade.min(dim=1).values.mean(), fde.min(dim=1).values.mean()


def pairwise_rank_loss(
    scores: torch.Tensor, teacher_costs: torch.Tensor, valid: torch.Tensor | None = None
) -> torch.Tensor:
    _, proposals = scores.shape
    diff_scores = scores[:, :, None] - scores[:, None, :]
    diff_costs = teacher_costs[:, None, :] - teacher_costs[:, :, None]
    labels = torch.sign(diff_costs)
    mask = 1.0 - torch.eye(proposals, device=scores.device, dtype=scores.dtype)[None]
    if valid is not None:
        mask = mask * valid[:, None, None].to(scores.dtype)
    loss = F.softplus(-diff_scores * labels)
    return (loss * mask).sum() / mask.sum().clamp(min=1.0)


def future_dyn_l2(
    pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor | None = None
) -> torch.Tensor:
    if valid is None:
        return F.mse_loss(pred, target)
    valid = _broadcast_mask(valid, pred).to(pred.dtype)
    loss = (pred - target).pow(2) * valid
    return loss.sum() / valid.sum().clamp(min=1.0)


@dataclass
class StageWeights:
    weights: Dict[str, float]

    def get(self, name: str, default: float = 0.0) -> float:
        return self.weights.get(name, default)


def stage_weight_preset(stage: str) -> StageWeights:
    presets = {
        "stage1a": {"cam_jepa": 1.0, "summary_jepa": 0.5, "sigreg": 0.05},
        "stage1b": {"cam_jepa": 0.3, "sigreg": 0.05, "depth": 1.0, "track": 0.5, "ego": 0.5},
        "stage1c": {"world_jepa": 1.0, "sigreg": 0.05, "mem": 0.5, "ego": 0.5},
        "stage1p5": {"lane": 1.0, "map": 0.5, "sigreg": 0.02},
        "stage2": {
            "occ_state": 2.0,
            "occ_sem": 1.0,
            "bev": 1.0,
            "dyn_flow": 0.6,
            "occl_risk": 0.6,
            "dyn_slots_distill": 0.2,
            "spec_slots_distill": 0.2,
            "hidden_risk_distill": 0.15,
            "lane": 1.0,
            "map": 0.4,
            "actors": 0.4,
            "ego": 0.3,
            "provenance": 0.2,
            "world_jepa": 0.3,
            "sigreg": 0.02,
        },
        "stage3": {
            "minade": 1.0,
            "fde": 0.7,
            "rank": 1.0,
            "ctrl": 0.4,
            "future": 0.4,
            "future_spec": 0.2,
            "dyn_slots_distill": 0.2,
            "spec_slots_distill": 0.2,
            "hidden_risk_distill": 0.15,
            "occ_state": 0.3,
            "lane": 0.3,
            "ego": 0.2,
            "world_jepa": 0.1,
        },
        "stage4": {
            "minade": 1.5,
            "rank": 1.0,
            "ctrl": 0.3,
            "future": 0.3,
            "future_spec": 0.2,
            "dyn_slots_distill": 0.2,
            "spec_slots_distill": 0.2,
            "hidden_risk_distill": 0.15,
            "occ_state": 0.35,
            "occ_sem": 0.25,
            "dyn_flow": 0.15,
            "occl_risk": 0.15,
            "lane": 0.30,
            "map": 0.10,
            "actors": 0.15,
            "ego": 0.15,
            "provenance": 0.10,
            "world_jepa": 0.10,
            "sigreg": 0.01,
        },
    }
    return StageWeights(presets[stage])


def compute_stage_losses(
    stage: str,
    outputs: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    weights: StageWeights | None = None,
) -> dict[str, torch.Tensor]:
    weights = weights or stage_weight_preset(stage)
    zero = _first_tensor(outputs).new_zeros(())
    losses: dict[str, torch.Tensor] = {}

    if "cam_projector_pred" in outputs and "cam_projector_target" in targets:
        losses["cam_jepa"] = masked_l2_jepa(
            outputs["cam_projector_pred"],
            targets["cam_projector_target"],
            targets.get("cam_mask"),
        )
    if "summary_projector_pred" in outputs and "summary_projector_target" in targets:
        losses["summary_jepa"] = masked_l2_jepa(
            outputs["summary_projector_pred"],
            targets["summary_projector_target"],
            targets.get("summary_mask"),
        )
    if "world_projector_pred" in outputs and "world_projector_target" in targets:
        losses["world_jepa"] = masked_l2_jepa(
            outputs["world_projector_pred"],
            targets["world_projector_target"],
            targets.get("world_mask"),
        )

    sig_terms = []
    for key in (
        "cam_projector_pred",
        "summary_projector_pred",
        "world_projector_pred",
        "proposal_embed",
    ):
        if key in outputs:
            sig_terms.append(sigreg(outputs[key]))
    if sig_terms:
        losses["sigreg"] = torch.stack(sig_terms).mean()

    if "depth_mean" in outputs and "depth_target" in targets:
        losses["depth"] = silog_depth(
            outputs["depth_mean"], targets["depth_target"], targets.get("depth_valid")
        )
    if "track_offsets" in outputs and "track_target" in targets:
        losses["track"] = track_epe(
            outputs["track_offsets"], targets["track_target"], targets.get("track_valid")
        )
    if "ego_out" in outputs and "ego_target" in targets:
        losses["ego"] = gaussian_nll(
            outputs["ego_out"],
            outputs.get("ego_logvar", torch.zeros_like(outputs["ego_out"])),
            targets["ego_target"],
            targets.get("ego_valid"),
        )
    if "mem_target" in targets and "static_grid" in outputs:
        losses["mem"] = F.l1_loss(outputs["static_grid"], targets["mem_target"])

    if "occ_state" in outputs and "occ_state_target" in targets:
        losses["occ_state"] = focal_ce_logits(
            outputs["occ_state"], targets["occ_state_target"], targets.get("occ_valid")
        ) + 0.5 * lovasz_logits(
            outputs["occ_state"], targets["occ_state_target"], targets.get("occ_valid")
        )
    if "occ_sem" in outputs and "occ_sem_target" in targets:
        losses["occ_sem"] = F.cross_entropy(
            outputs["occ_sem"], targets["occ_sem_target"].long()
        )
    if "bev_lite" in outputs and "bev_target" in targets:
        losses["bev"] = F.binary_cross_entropy_with_logits(
            outputs["bev_lite"], targets["bev_target"].float()
        )
    if "dyn_flow_bev" in outputs and "dyn_flow_target" in targets:
        losses["dyn_flow"] = F.mse_loss(outputs["dyn_flow_bev"], targets["dyn_flow_target"])
    if "occl_risk_bev" in outputs and "occl_risk_target" in targets:
        losses["occl_risk"] = F.binary_cross_entropy_with_logits(
            outputs["occl_risk_bev"],
            targets["occl_risk_target"].float(),
        )
    if "provenance" in outputs and "provenance_target" in targets:
        losses["provenance"] = F.cross_entropy(
            outputs["provenance"], targets["provenance_target"].long()
        )

    lane_target = {
        key: targets[key]
        for key in ("centerline", "left_boundary", "right_boundary", "lane_sem_cls")
        if key in targets
    }
    if "centerline" in outputs and lane_target:
        lane_pred = {
            "centerline": outputs["centerline"],
            "left_boundary": outputs["left_boundary"],
            "right_boundary": outputs["right_boundary"],
            "lane_sem_cls": outputs["lane_sem_cls"],
        }
        losses["lane"] = hungarian_lane_loss(lane_pred, lane_target, targets.get("lane_valid"))
    if "map_poly" in outputs and "map_poly" in targets:
        losses["map"] = F.l1_loss(outputs["map_poly"], targets["map_poly"])

    actor_target = {
        key: targets[key] for key in ("actor_box", "actor_vel", "actor_cls") if key in targets
    }
    if "actor_box" in outputs and actor_target:
        actor_pred = {
            "actor_box": outputs["actor_box"],
            "actor_vel": outputs["actor_vel"],
            "actor_cls": outputs["actor_cls"],
        }
        losses["actors"] = hungarian_actor_loss(
            actor_pred, actor_target, targets.get("actor_valid")
        )

    if "traj" in outputs and "traj_target" in targets:
        losses["minade"], losses["fde"] = min_ade_fde(
            outputs["traj"], targets["traj_target"], targets.get("traj_valid")
        )
    if "score" in outputs and "teacher_cost" in targets:
        losses["rank"] = pairwise_rank_loss(
            outputs["score"], targets["teacher_cost"], targets.get("rank_valid")
        )
    if "curvature" in outputs and "curvature_target" in targets:
        losses["ctrl"] = F.l1_loss(outputs["curvature"], targets["curvature_target"])
        if "speed_target" in targets:
            losses["ctrl"] = losses["ctrl"] + F.l1_loss(
                outputs["speed"], targets["speed_target"]
            )
    if "future_dyn" in outputs and "future_dyn_target" in targets:
        losses["future"] = future_dyn_l2(
            outputs["future_dyn"],
            targets["future_dyn_target"],
            targets.get("future_valid"),
        )
    if "future_spec" in outputs and "future_spec_target" in targets:
        losses["future_spec"] = future_dyn_l2(
            outputs["future_spec"],
            targets["future_spec_target"],
            targets.get("future_spec_valid"),
        )
    if "dynamic_slots" in outputs and "dynamic_slots_target" in targets:
        losses["dyn_slots_distill"] = future_dyn_l2(
            outputs["dynamic_slots"],
            targets["dynamic_slots_target"],
            targets.get("dynamic_slots_valid"),
        )
    if "speculative_slots" in outputs and "speculative_slots_target" in targets:
        losses["spec_slots_distill"] = future_dyn_l2(
            outputs["speculative_slots"],
            targets["speculative_slots_target"],
            targets.get("speculative_slots_valid"),
        )
    if "hidden_risk_penalty" in outputs and "hidden_risk_penalty_target" in targets:
        losses["hidden_risk_distill"] = F.l1_loss(
            outputs["hidden_risk_penalty"],
            targets["hidden_risk_penalty_target"],
        )

    total = zero
    for name, loss in losses.items():
        total = total + weights.get(name) * loss
    losses["total"] = total
    return losses


def compute_stage1a_losses(
    outputs: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    weights: StageWeights | None = None,
) -> dict[str, torch.Tensor]:
    weights = weights or stage_weight_preset("stage1a")
    pred_cam = F.layer_norm(
        outputs["cam_projector_pred"],
        outputs["cam_projector_pred"].shape[-1:],
    )
    tgt_cam = F.layer_norm(
        targets["cam_projector_target"],
        targets["cam_projector_target"].shape[-1:],
    )
    pred_sum = F.layer_norm(
        outputs["summary_projector_pred"],
        outputs["summary_projector_pred"].shape[-1:],
    )
    tgt_sum = F.layer_norm(
        targets["summary_projector_target"],
        targets["summary_projector_target"].shape[-1:],
    )
    losses = {
        "cam_jepa": masked_l2_jepa(pred_cam, tgt_cam, targets["cam_mask"]),
        "summary_jepa": masked_l2_jepa(pred_sum, tgt_sum, targets["summary_mask"]),
        "sigreg": 0.5 * (sigreg(pred_cam) + sigreg(pred_sum)),
    }
    total = pred_cam.new_zeros(())
    for name, loss in losses.items():
        total = total + weights.get(name) * loss
    losses["total"] = total
    return losses


def compute_stage1b_losses(
    outputs: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    weights: StageWeights | None = None,
) -> dict[str, torch.Tensor]:
    weights = weights or stage_weight_preset("stage1b")
    ego_valid = targets.get("pose_valid_recent")
    kin_valid = None if ego_valid is None else ego_valid
    losses = {
        "ego": gaussian_nll(
            outputs["pose_delta_seq"],
            outputs["logvar_pose_seq"],
            targets["pose_delta_recent"],
            ego_valid,
        ),
        "kinematics": gaussian_nll(
            outputs["kinematics_seq"],
            torch.zeros_like(outputs["kinematics_seq"]),
            targets["kinematics_recent"],
            kin_valid,
        ),
        "depth": silog_depth(
            outputs["depth_mean_seq"],
            targets["depth_target_recent"],
            targets.get("depth_valid_recent"),
        ),
        "track": track_epe(
            outputs["track_offsets_seq"],
            targets["track_target_recent_sparse"],
            targets.get("track_valid_recent_sparse"),
        ),
    }
    total = outputs["pose_delta_seq"].new_zeros(())
    for name, loss in losses.items():
        key = "ego" if name in {"ego", "kinematics"} else name
        total = total + weights.get(key) * loss
    losses["total"] = total
    return losses


def compute_stage1c_losses(
    outputs: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    weights: StageWeights | None = None,
) -> dict[str, torch.Tensor]:
    weights = weights or stage_weight_preset("stage1c")
    losses = {
        "world_jepa": masked_l2_jepa(
            outputs["world_projector_pred"],
            targets["world_projector_target"],
            targets["world_mask"],
        ),
        "mem": F.smooth_l1_loss(
            outputs["static_grid_seq"],
            targets["teacher_static_grid_seq"],
        ),
    }
    if "pose_delta_seq" in outputs and "pose_delta_recent" in targets:
        losses["ego"] = gaussian_nll(
            outputs["pose_delta_seq"],
            outputs["logvar_pose_seq"],
            targets["pose_delta_recent"],
            targets.get("pose_valid_recent"),
        )
    if "kinematics_seq" in outputs and "kinematics_recent" in targets:
        losses["ego"] = losses.get("ego", outputs["kinematics_seq"].new_zeros(())) + gaussian_nll(
            outputs["kinematics_seq"],
            torch.zeros_like(outputs["kinematics_seq"]),
            targets["kinematics_recent"],
            targets.get("pose_valid_recent"),
        )
    losses["sigreg"] = sigreg(outputs["world_projector_pred"])
    total = outputs["world_projector_pred"].new_zeros(())
    for name, loss in losses.items():
        total = total + weights.get(name) * loss
    losses["total"] = total
    return losses
