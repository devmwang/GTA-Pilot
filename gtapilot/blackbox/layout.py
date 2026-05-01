from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from gtapilot.blackbox.constants import (
    ACTIONS_JOURNAL_FILE_NAME,
    ACTIONS_METADATA_FILE_NAME,
    BACKUPS_DIR_NAME,
    FRAMES_JOURNAL_FILE_NAME,
    PRIVILEGED_DIR_NAME,
    VIDEO_FILE_NAME,
    VIDEO_METADATA_FILE_NAME,
)


@dataclass(slots=True, frozen=True)
class BlackboxClipPaths:
    clip_id: str
    recordings_root: Path
    clip_dir: Path
    video_path: Path
    metadata_path: Path
    actions_path: Path
    privileged_dir: Path
    backups_dir: Path
    frames_journal_path: Path
    actions_journal_path: Path
    session_timestamp: str


def build_session_paths(output_dir: Path) -> BlackboxClipPaths:
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    clip_id = f"capture_{session_timestamp}"
    clip_dir = output_dir / clip_id
    return BlackboxClipPaths(
        clip_id=clip_id,
        recordings_root=output_dir,
        clip_dir=clip_dir,
        video_path=clip_dir / VIDEO_FILE_NAME,
        metadata_path=clip_dir / VIDEO_METADATA_FILE_NAME,
        actions_path=clip_dir / ACTIONS_METADATA_FILE_NAME,
        privileged_dir=clip_dir / PRIVILEGED_DIR_NAME,
        backups_dir=clip_dir / BACKUPS_DIR_NAME,
        frames_journal_path=clip_dir / FRAMES_JOURNAL_FILE_NAME,
        actions_journal_path=clip_dir / ACTIONS_JOURNAL_FILE_NAME,
        session_timestamp=session_timestamp,
    )


def resolve_clip_paths(recordings_root: Path, clip_name: str) -> BlackboxClipPaths:
    clip_dir = (recordings_root / clip_name).resolve()
    return BlackboxClipPaths(
        clip_id=clip_name,
        recordings_root=recordings_root.resolve(),
        clip_dir=clip_dir,
        video_path=clip_dir / VIDEO_FILE_NAME,
        metadata_path=clip_dir / VIDEO_METADATA_FILE_NAME,
        actions_path=clip_dir / ACTIONS_METADATA_FILE_NAME,
        privileged_dir=clip_dir / PRIVILEGED_DIR_NAME,
        backups_dir=clip_dir / BACKUPS_DIR_NAME,
        frames_journal_path=clip_dir / FRAMES_JOURNAL_FILE_NAME,
        actions_journal_path=clip_dir / ACTIONS_JOURNAL_FILE_NAME,
        session_timestamp=clip_name.removeprefix("capture_"),
    )
