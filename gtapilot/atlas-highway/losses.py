from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .config import AtlasHAConfig


@dataclass(slots=True)
class AtlasHALossWeights:
    traj: float = 3.0
    candidate: float = 1.0
    lane: float = 1.0
    lead: float = 0.7
    adjacent: float = 0.5
    ego: float = 0.5
    legal_speed: float = 0.5
    latacc: float = 0.2
    curvature_limit: float = 0.2
    scene_type: float = 0.3
    takeover: float = 0.2
    smoothness: float = 0.1


def compute_atlas_ha_losses(
    outputs: dict[str, torch.Tensor | None],
    targets: dict[str, torch.Tensor],
    cfg: AtlasHAConfig,
    weights: AtlasHALossWeights | None = None,
) -> dict[str, torch.Tensor]:
    w = weights or AtlasHALossWeights()
    losses: dict[str, torch.Tensor] = {}
    traj = outputs["traj_candidates"]
    assert traj is not None
    candidate_logits = outputs["candidate_logits"]
    assert candidate_logits is not None
    target_traj = targets["target_traj"].to(device=traj.device, dtype=traj.dtype)
    if "target_candidate" in targets:
        target_candidate = targets["target_candidate"].to(device=traj.device).long().view(-1)
        selected = traj[torch.arange(traj.shape[0], device=traj.device), target_candidate]
        losses["traj"] = F.huber_loss(selected, target_traj)
        losses["candidate"] = F.cross_entropy(candidate_logits, target_candidate)
    else:
        per_k = F.huber_loss(
            traj,
            target_traj[:, None].expand_as(traj),
            reduction="none",
        ).mean(dim=(-1, -2))
        losses["traj"] = per_k.min(dim=1).values.mean()
        losses["candidate"] = torch.zeros((), device=traj.device, dtype=traj.dtype)

    lane_lat = outputs["lane_lat_pred"]
    lane_valid_logit = outputs["lane_valid_logit"]
    lane_conf = outputs["lane_conf"]
    assert lane_lat is not None and lane_valid_logit is not None and lane_conf is not None
    target_lane_valid = targets["target_lane_valid"].to(device=traj.device, dtype=traj.dtype)
    target_lane_lat = targets["target_lane_lat"].to(device=traj.device, dtype=traj.dtype)
    target_lane_conf = targets["target_lane_conf"].to(device=traj.device, dtype=traj.dtype)
    masked_lane = F.huber_loss(lane_lat, target_lane_lat, reduction="none") * target_lane_valid
    losses["lane_lat"] = masked_lane.sum() / target_lane_valid.sum().clamp_min(1.0)
    losses["lane_valid"] = F.binary_cross_entropy_with_logits(lane_valid_logit, target_lane_valid)
    losses["lane_conf"] = F.binary_cross_entropy(lane_conf.clamp(1e-4, 1.0 - 1e-4), target_lane_conf)
    losses["lane"] = losses["lane_lat"] + losses["lane_valid"] + losses["lane_conf"]

    lead_present_logit = outputs["lead_present_logit"]
    lead_state = outputs["lead_state"]
    assert lead_present_logit is not None and lead_state is not None
    target_lead_present = targets["target_lead_present"].to(device=traj.device, dtype=traj.dtype)
    target_lead_state = targets["target_lead_state"].to(device=traj.device, dtype=traj.dtype)
    losses["lead_present"] = F.binary_cross_entropy_with_logits(lead_present_logit, target_lead_present)
    lead_mask = target_lead_present.expand_as(lead_state)
    losses["lead_state"] = (
        F.huber_loss(lead_state, target_lead_state, reduction="none") * lead_mask
    ).sum() / lead_mask.sum().clamp_min(1.0)
    losses["lead"] = losses["lead_present"] + losses["lead_state"]

    losses["adjacent"] = _adjacent_loss(outputs, targets, traj)
    ego = outputs["ego_kinematics"]
    assert ego is not None
    losses["ego"] = F.huber_loss(ego, targets["target_ego"].to(device=traj.device, dtype=traj.dtype))
    losses["legal_speed"] = _legal_speed_loss(outputs, targets, traj)
    losses["latacc"] = _latacc_penalty(traj)
    losses["curvature_limit"] = _curvature_limit_penalty(traj)
    losses["scene_type"] = _scene_type_loss(outputs, targets, traj)
    losses["takeover"] = _takeover_loss(outputs, targets, traj)
    losses["smoothness"] = _smoothness_loss(traj)
    losses["total"] = (
        w.traj * losses["traj"]
        + w.candidate * losses["candidate"]
        + w.lane * losses["lane"]
        + w.lead * losses["lead"]
        + w.adjacent * losses["adjacent"]
        + w.ego * losses["ego"]
        + w.legal_speed * losses["legal_speed"]
        + w.latacc * losses["latacc"]
        + w.curvature_limit * losses["curvature_limit"]
        + w.scene_type * losses["scene_type"]
        + w.takeover * losses["takeover"]
        + w.smoothness * losses["smoothness"]
    )
    del cfg
    return losses


def _adjacent_loss(
    outputs: dict[str, torch.Tensor | None],
    targets: dict[str, torch.Tensor],
    ref: torch.Tensor,
) -> torch.Tensor:
    terms = []
    for side, key in (("left", "target_adjacent_left_visible"), ("right", "target_adjacent_right_visible")):
        if key in targets:
            pred = outputs[f"adjacent_{side}"]
            assert pred is not None
            terms.append(F.huber_loss(pred, targets[key].to(device=ref.device, dtype=ref.dtype)))
    if "target_lane_change_teacher_ok" in targets:
        logits = torch.stack(
            [outputs["adjacent_left"][:, 0], outputs["adjacent_right"][:, 0]],
            dim=-1,
        )
        target = targets["target_lane_change_teacher_ok"].to(device=ref.device, dtype=ref.dtype)
        terms.append(F.binary_cross_entropy(logits.clamp(1e-4, 1.0 - 1e-4), target))
    return sum(terms) if terms else torch.zeros((), device=ref.device, dtype=ref.dtype)


def _legal_speed_loss(
    outputs: dict[str, torch.Tensor | None],
    targets: dict[str, torch.Tensor],
    traj: torch.Tensor,
) -> torch.Tensor:
    if "target_legal_speed" not in targets:
        return torch.zeros((), device=traj.device, dtype=traj.dtype)
    target_candidate = targets.get("target_candidate")
    idx = target_candidate.to(device=traj.device).long().view(-1) if target_candidate is not None else traj[..., 3].mean(dim=-1).argmax(dim=1)
    pred = traj[torch.arange(traj.shape[0], device=traj.device), idx, :, 3]
    return F.huber_loss(pred, targets["target_legal_speed"].to(device=traj.device, dtype=traj.dtype))


def _latacc_penalty(traj: torch.Tensor, limit: float = 3.0) -> torch.Tensor:
    curvature = _curvature(traj)
    a_lat = traj[..., 3] ** 2 * curvature.abs()
    return F.relu(a_lat - limit).pow(2).mean()


def _curvature_limit_penalty(traj: torch.Tensor, kappa_max: float = 0.20) -> torch.Tensor:
    return F.relu(_curvature(traj).abs() - kappa_max).pow(2).mean()


def _curvature(traj: torch.Tensor) -> torch.Tensor:
    long_m = traj[..., 0]
    lat_m = traj[..., 1]
    dx = torch.gradient(long_m, dim=-1)[0]
    dy = torch.gradient(lat_m, dim=-1)[0]
    ddx = torch.gradient(dx, dim=-1)[0]
    ddy = torch.gradient(dy, dim=-1)[0]
    return (dx * ddy - dy * ddx) / (dx * dx + dy * dy).clamp_min(1e-4).pow(1.5)


def _scene_type_loss(
    outputs: dict[str, torch.Tensor | None],
    targets: dict[str, torch.Tensor],
    ref: torch.Tensor,
) -> torch.Tensor:
    if "target_soft_scene_type" not in targets:
        return torch.zeros((), device=ref.device, dtype=ref.dtype)
    logits = outputs["scene_type_logits"]
    assert logits is not None
    target = targets["target_soft_scene_type"].to(device=ref.device)
    if target.ndim > 1:
        return -(target.to(dtype=ref.dtype) * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
    return F.cross_entropy(logits, target.long())


def _takeover_loss(
    outputs: dict[str, torch.Tensor | None],
    targets: dict[str, torch.Tensor],
    ref: torch.Tensor,
) -> torch.Tensor:
    if "target_takeover_required" not in targets:
        return torch.zeros((), device=ref.device, dtype=ref.dtype)
    logit = outputs["takeover_required_logit"]
    assert logit is not None
    target = targets["target_takeover_required"].to(device=ref.device, dtype=ref.dtype)
    return F.binary_cross_entropy_with_logits(logit, target)


def _smoothness_loss(traj: torch.Tensor) -> torch.Tensor:
    if traj.shape[-2] < 3:
        return torch.zeros((), device=traj.device, dtype=traj.dtype)
    d2 = traj[..., 2:, :3] - 2.0 * traj[..., 1:-1, :3] + traj[..., :-2, :3]
    speed_d = torch.diff(traj[..., 3], dim=-1)
    return d2.pow(2).mean() + 0.1 * speed_d.pow(2).mean()
