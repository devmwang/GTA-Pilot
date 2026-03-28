from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..config import AtlasConfig
from ..state import AtlasWorldMemoryView
from ..utils import CrossAttentionBlock, LearnedQueryPool, flatten_hw, sinusoidal_embedding


def _alive_weights(alive: torch.Tensor | None, ref: torch.Tensor) -> torch.Tensor | None:
    if alive is None:
        return None
    return alive.to(dtype=ref.dtype, device=ref.device).clamp(0.0, 1.0)


def _alive_padding_mask(alive: torch.Tensor | None) -> torch.Tensor | None:
    if alive is None:
        return None
    if alive.dtype == torch.bool:
        return ~alive
    return alive <= 1e-4


class HorizonConditionedQueries(nn.Module):
    def __init__(self, num_queries: int, dim: int):
        super().__init__()
        self.base_queries = nn.Parameter(torch.randn(1, num_queries, dim) * 0.02)
        self.horizon_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, batch_size: int, horizons_s: torch.Tensor) -> torch.Tensor:
        horizons_s = horizons_s.to(dtype=self.base_queries.dtype).view(1, -1, 1)
        horizon_embed = self.horizon_mlp(
            sinusoidal_embedding(horizons_s, self.base_queries.shape[-1])
        )
        query = self.base_queries[:, None] + horizon_embed[:, :, None, :]
        return query.expand(batch_size, -1, -1, -1)


class LatentFutureProjector(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        num_queries: int,
        heads: int,
        dropout: float = 0.0,
        decoder_blocks: int = 2,
    ):
        super().__init__()
        self.query_bank = HorizonConditionedQueries(num_queries=num_queries, dim=dim)
        self.blocks = nn.ModuleList(
            [CrossAttentionBlock(dim, heads, dropout) for _ in range(decoder_blocks)]
        )
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, src_tokens: torch.Tensor, horizons_s: torch.Tensor) -> torch.Tensor:
        if src_tokens.ndim != 4:
            raise ValueError("src_tokens must have shape [B, T, N, D].")
        batch, steps, _, dim = src_tokens.shape
        src_flat = src_tokens.reshape(batch * steps, src_tokens.shape[2], dim)
        horizons_s = horizons_s.to(device=src_tokens.device, dtype=src_tokens.dtype)
        query = self.query_bank(batch * steps, horizons_s).reshape(
            batch * steps,
            horizons_s.numel(),
            self.query_bank.base_queries.shape[1],
            dim,
        )
        preds: list[torch.Tensor] = []
        for horizon_idx in range(horizons_s.numel()):
            q = query[:, horizon_idx]
            for block in self.blocks:
                q = block(q, src_flat)
            preds.append(self.out(self.norm(q)))
        return torch.stack(preds, dim=1).reshape(
            batch,
            steps,
            horizons_s.numel(),
            self.query_bank.base_queries.shape[1],
            dim,
        )


@dataclass(slots=True)
class WorldReadoutLayout:
    static_queries: int
    dynamic_queries: int
    speculative_queries: int
    lane_queries: int
    map_queries: int
    route_queries: int
    reasoner_queries: int
    ego_queries: int

    @property
    def total_queries(self) -> int:
        return (
            self.static_queries
            + self.dynamic_queries
            + self.speculative_queries
            + self.lane_queries
            + self.map_queries
            + self.route_queries
            + self.reasoner_queries
            + self.ego_queries
        )


class WorldTargetReadout(nn.Module):
    def __init__(
        self,
        cfg: AtlasConfig,
        *,
        static_queries: int = 16,
        dynamic_queries: int = 16,
        speculative_queries: int = 8,
        lane_queries: int = 8,
        map_queries: int = 8,
        route_queries: int = 4,
        reasoner_queries: int = 4,
        ego_queries: int = 4,
    ):
        super().__init__()
        self.cfg = cfg
        self.layout = WorldReadoutLayout(
            static_queries=static_queries,
            dynamic_queries=dynamic_queries,
            speculative_queries=speculative_queries,
            lane_queries=lane_queries,
            map_queries=map_queries,
            route_queries=route_queries,
            reasoner_queries=reasoner_queries,
            ego_queries=ego_queries,
        )
        heads = min(cfg.temporal.num_heads, max(1, cfg.hidden_dim // 64))
        self.static_pool = LearnedQueryPool(static_queries, cfg.hidden_dim, heads=heads)
        self.dynamic_pool = LearnedQueryPool(dynamic_queries, cfg.hidden_dim, heads=heads)
        self.speculative_pool = LearnedQueryPool(speculative_queries, cfg.hidden_dim, heads=heads)
        self.lane_pool = LearnedQueryPool(lane_queries, cfg.hidden_dim, heads=heads)
        self.map_pool = LearnedQueryPool(map_queries, cfg.hidden_dim, heads=heads)
        self.route_pool = LearnedQueryPool(route_queries, cfg.hidden_dim, heads=heads)
        self.reasoner_pool = LearnedQueryPool(reasoner_queries, cfg.hidden_dim, heads=heads)
        self.ego_pool = LearnedQueryPool(ego_queries, cfg.hidden_dim, heads=heads)
        self.norm = nn.LayerNorm(cfg.hidden_dim)

    def forward(
        self,
        *,
        static_grid: torch.Tensor,
        dynamic_slots: torch.Tensor,
        speculative_slots: torch.Tensor,
        lane_slots: torch.Tensor,
        map_elem_slots: torch.Tensor,
        route_tokens: torch.Tensor,
        reasoner_tokens: torch.Tensor,
        ego_tokens: torch.Tensor,
        dynamic_alive: torch.Tensor | None = None,
        speculative_alive: torch.Tensor | None = None,
    ) -> torch.Tensor:
        static_tokens = flatten_hw(static_grid)
        dynamic_weights = _alive_weights(dynamic_alive, dynamic_slots)
        speculative_weights = _alive_weights(speculative_alive, speculative_slots)
        if dynamic_weights is not None:
            dynamic_slots = dynamic_slots * dynamic_weights.unsqueeze(-1)
        if speculative_weights is not None:
            speculative_slots = speculative_slots * speculative_weights.unsqueeze(-1)
        return self.norm(
            torch.cat(
                [
                    self.static_pool(static_tokens),
                    self.dynamic_pool(
                        dynamic_slots,
                        key_padding_mask=_alive_padding_mask(dynamic_alive),
                    ),
                    self.speculative_pool(
                        speculative_slots,
                        key_padding_mask=_alive_padding_mask(speculative_alive),
                    ),
                    self.lane_pool(lane_slots),
                    self.map_pool(map_elem_slots),
                    self.route_pool(route_tokens),
                    self.reasoner_pool(reasoner_tokens),
                    self.ego_pool(ego_tokens),
                ],
                dim=1,
            )
        )


class Stage1APretrainHeads(nn.Module):
    def __init__(self, cfg: AtlasConfig):
        super().__init__()
        self.cam_projector = LatentFutureProjector(
            dim=cfg.hidden_dim,
            num_queries=cfg.temporal.recent_cam_tokens,
            heads=cfg.temporal.num_heads,
            dropout=cfg.temporal.dropout,
            decoder_blocks=2,
        )
        self.summary_projector = LatentFutureProjector(
            dim=cfg.hidden_dim,
            num_queries=cfg.temporal.mid_summary_tokens,
            heads=cfg.temporal.num_heads,
            dropout=cfg.temporal.dropout,
            decoder_blocks=2,
        )

    def forward(
        self,
        *,
        cam_src: torch.Tensor,
        summary_src: torch.Tensor,
        cam_horizons_s: torch.Tensor,
        summary_horizons_s: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return {
            "cam_projector_pred": self.cam_projector(cam_src, cam_horizons_s),
            "summary_projector_pred": self.summary_projector(
                summary_src,
                summary_horizons_s,
            ),
        }


class Stage1CPretrainHeads(nn.Module):
    def __init__(self, cfg: AtlasConfig):
        super().__init__()
        self.world_readout = WorldTargetReadout(cfg)
        self.world_projector = LatentFutureProjector(
            dim=cfg.hidden_dim,
            num_queries=self.world_readout.layout.total_queries,
            heads=cfg.temporal.num_heads,
            dropout=cfg.temporal.dropout,
            decoder_blocks=2,
        )

    def readout(self, world_view: AtlasWorldMemoryView) -> torch.Tensor:
        return self.world_readout(
            static_grid=world_view.static_grid,
            dynamic_slots=world_view.dynamic_slots,
            speculative_slots=world_view.speculative_slots,
            lane_slots=world_view.lane_slots,
            map_elem_slots=world_view.map_elem_slots,
            route_tokens=world_view.route_tokens,
            reasoner_tokens=world_view.reasoner_tokens,
            ego_tokens=world_view.ego_tokens,
            dynamic_alive=world_view.dynamic_slot_alive,
            speculative_alive=world_view.speculative_slot_alive,
        )

    def readout_seq(
        self,
        *,
        static_grid_seq: torch.Tensor,
        dynamic_slots_seq: torch.Tensor,
        speculative_slots_seq: torch.Tensor,
        lane_slots_seq: torch.Tensor,
        map_elem_slots_seq: torch.Tensor,
        route_tokens_seq: torch.Tensor,
        reasoner_tokens_seq: torch.Tensor,
        ego_tokens_seq: torch.Tensor,
        dynamic_alive_seq: torch.Tensor | None = None,
        speculative_alive_seq: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if static_grid_seq.ndim != 5:
            raise ValueError("static_grid_seq must have shape [B, T, C, H, W].")
        batch, steps = static_grid_seq.shape[:2]
        static_grid = static_grid_seq.reshape(batch * steps, *static_grid_seq.shape[2:])
        dynamic_slots = dynamic_slots_seq.reshape(batch * steps, *dynamic_slots_seq.shape[2:])
        speculative_slots = speculative_slots_seq.reshape(batch * steps, *speculative_slots_seq.shape[2:])
        lane_slots = lane_slots_seq.reshape(batch * steps, *lane_slots_seq.shape[2:])
        map_elem_slots = map_elem_slots_seq.reshape(batch * steps, *map_elem_slots_seq.shape[2:])
        route_tokens = route_tokens_seq.reshape(batch * steps, *route_tokens_seq.shape[2:])
        reasoner_tokens = reasoner_tokens_seq.reshape(batch * steps, *reasoner_tokens_seq.shape[2:])
        ego_tokens = ego_tokens_seq.reshape(batch * steps, *ego_tokens_seq.shape[2:])
        dynamic_alive = None if dynamic_alive_seq is None else dynamic_alive_seq.reshape(batch * steps, dynamic_alive_seq.shape[-1])
        speculative_alive = None if speculative_alive_seq is None else speculative_alive_seq.reshape(batch * steps, speculative_alive_seq.shape[-1])
        readout = self.world_readout(
            static_grid=static_grid,
            dynamic_slots=dynamic_slots,
            speculative_slots=speculative_slots,
            lane_slots=lane_slots,
            map_elem_slots=map_elem_slots,
            route_tokens=route_tokens,
            reasoner_tokens=reasoner_tokens,
            ego_tokens=ego_tokens,
            dynamic_alive=dynamic_alive,
            speculative_alive=speculative_alive,
        )
        return readout.reshape(batch, steps, readout.shape[1], readout.shape[2])

    def project(self, src_tokens: torch.Tensor, horizons_s: torch.Tensor) -> torch.Tensor:
        return self.world_projector(src_tokens, horizons_s)
