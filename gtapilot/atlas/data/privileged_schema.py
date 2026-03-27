from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PrivilegedClipManifest:
    clip_id: str
    schema_version: int = 2
    source_video_file: str | None = None
    source_metadata_file: str | None = None
    source_metadata_sha1: str | None = None
    frame_count: int = 0
    grid_height_8x: int = 136
    grid_width_8x: int = 240
    track_lag_indices: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 31])
    frame_ids_file: str | None = None
    capture_timestamps_ns_file: str | None = None
    video_frame_indices_file: str | None = None
    files: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PrivilegedClipManifest":
        return cls(
            clip_id=str(payload["clip_id"]),
            schema_version=int(payload.get("schema_version", 2)),
            source_video_file=payload.get("source_video_file"),
            source_metadata_file=payload.get("source_metadata_file"),
            source_metadata_sha1=payload.get("source_metadata_sha1"),
            frame_count=int(payload.get("frame_count", 0)),
            grid_height_8x=int(payload.get("grid_height_8x", 136)),
            grid_width_8x=int(payload.get("grid_width_8x", 240)),
            track_lag_indices=[int(value) for value in payload.get("track_lag_indices", [1, 2, 4, 8, 16, 31])],
            frame_ids_file=payload.get("frame_ids_file"),
            capture_timestamps_ns_file=payload.get("capture_timestamps_ns_file"),
            video_frame_indices_file=payload.get("video_frame_indices_file"),
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
