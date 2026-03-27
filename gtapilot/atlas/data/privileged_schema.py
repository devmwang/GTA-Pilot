from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PrivilegedClipManifest:
    clip_id: str
    schema_version: int = 1
    source_video_file: str | None = None
    frame_count: int = 0
    grid_height_8x: int = 136
    grid_width_8x: int = 240
    track_lag_indices: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 31])
    files: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PrivilegedClipManifest":
        return cls(
            clip_id=str(payload["clip_id"]),
            schema_version=int(payload.get("schema_version", 1)),
            source_video_file=payload.get("source_video_file"),
            frame_count=int(payload.get("frame_count", 0)),
            grid_height_8x=int(payload.get("grid_height_8x", 136)),
            grid_width_8x=int(payload.get("grid_width_8x", 240)),
            track_lag_indices=[int(value) for value in payload.get("track_lag_indices", [1, 2, 4, 8, 16, 31])],
            files=dict(payload.get("files", {})),
            metadata=dict(payload.get("metadata", {})),
        )

    def save(self, path: str | Path) -> None:
        import json

        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "PrivilegedClipManifest":
        import json

        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
