from __future__ import annotations

from .aux_heads import BEVLiteHead


class ProvenanceHead(BEVLiteHead):
    def forward(self, world):  # type: ignore[override]
        return {"provenance": super().forward(world)["provenance"]}


__all__ = ["ProvenanceHead"]
