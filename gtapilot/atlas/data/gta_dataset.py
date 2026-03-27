from __future__ import annotations

import bisect
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import AtlasConfig
from .schema import AtlasTemporalClipIndex, BlackboxFrameRecord


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fallback_dt_seconds(frames: list[BlackboxFrameRecord], index: int, nominal_fps: float) -> float:
    if index > 0:
        dt_ns = max(
            1,
            frames[index].capture_timestamp_ns - frames[index - 1].capture_timestamp_ns,
        )
        return float(dt_ns) / 1_000_000_000.0
    return 1.0 / max(1.0, float(nominal_fps))


def _frame_dt_tensor(
    frames: list[BlackboxFrameRecord],
    indices: list[int],
    nominal_fps: float,
    *,
    relative_to_selected_prev: bool = False,
) -> torch.Tensor:
    values: list[float] = []
    for offset, frame_index in enumerate(indices):
        if relative_to_selected_prev and offset > 0:
            prev_index = indices[offset - 1]
            dt_ns = max(
                1,
                frames[frame_index].capture_timestamp_ns
                - frames[prev_index].capture_timestamp_ns,
            )
            values.append(float(dt_ns) / 1_000_000_000.0)
            continue
        values.append(_fallback_dt_seconds(frames, frame_index, nominal_fps))
    return torch.tensor(values, dtype=torch.float32).unsqueeze(-1)


def _action_history_from_frames(
    frames: list[BlackboxFrameRecord],
    indices: list[int],
) -> torch.Tensor:
    return torch.tensor(
        np.stack([frames[frame_index].action_vector for frame_index in indices]),
        dtype=torch.float32,
    )


class AtlasBlackboxClipDataset(Dataset[dict[str, Any]]):
    """
    Builds Atlas temporal training samples directly from blackbox MKV+metadata recordings.

    Each sample is a three-tier temporal bundle:
      - recent dense RGB frames
      - older dense-rate RGB frames to be compressed into history tokens
      - sparse mid-history RGB keyframes for long summary context
      - long action history aligned through per-frame blackbox metadata
    """

    def __init__(
        self,
        recordings_root: str | Path,
        cfg: AtlasConfig,
        *,
        metadata_paths: list[str | Path] | None = None,
    ):
        self.recordings_root = Path(recordings_root)
        self.cfg = cfg
        self.metadata_paths = (
            [Path(path) for path in metadata_paths]
            if metadata_paths is not None
            else sorted(self.recordings_root.glob("capture_*_metadata.json"))
        )
        self.samples = self._build_indices()

    def _build_indices(self) -> list[AtlasTemporalClipIndex]:
        samples: list[AtlasTemporalClipIndex] = []
        for metadata_path in self.metadata_paths:
            manifest = _read_json(metadata_path)
            frames = [
                BlackboxFrameRecord.from_dict(frame_payload)
                for frame_payload in manifest.get("frames", [])
            ]
            if not frames:
                continue

            nominal_fps = float(manifest.get("video_nominal_fps", 0.0) or 0.0)
            if nominal_fps <= 0.0:
                nominal_fps = float(frames[0].frame_metadata.get("nominal_fps", 1.0))

            video_path = metadata_path.with_name(str(manifest["video_file_name"]))
            capture_times = [frame.capture_timestamp_ns for frame in frames]
            recent_count = self.cfg.temporal.recent_full_frames
            older_count = self.cfg.temporal.older_compressed_frames
            action_count = self.cfg.action.history_len
            mid_count = self.cfg.temporal.mid_summary_frames
            mid_interval_ns = int(1_000_000_000 / max(1, self.cfg.temporal.mid_summary_hz))

            earliest_target = max(recent_count + older_count - 1, action_count - 1)
            for target_index in range(earliest_target, len(frames)):
                recent_start = target_index - recent_count + 1
                older_start = recent_start - older_count
                older_end = recent_start
                if older_start < 0:
                    continue

                target_time_ns = capture_times[older_start]
                mid_indices: list[int] = []
                valid_mid = True
                for offset in range(mid_count, 0, -1):
                    desired_time_ns = target_time_ns - offset * mid_interval_ns
                    history_stop = older_start
                    candidate = bisect.bisect_right(
                        capture_times,
                        desired_time_ns,
                        0,
                        history_stop,
                    ) - 1
                    if candidate < 0:
                        valid_mid = False
                        break
                    mid_indices.append(candidate)
                if not valid_mid:
                    continue

                action_start = target_index - action_count + 1
                if action_start < 0:
                    continue

                samples.append(
                    AtlasTemporalClipIndex(
                        clip_id=metadata_path.stem.replace("_metadata", ""),
                        metadata_path=str(metadata_path),
                        video_path=str(video_path),
                        target_frame_index=target_index,
                        recent_frame_indices=list(range(recent_start, target_index + 1)),
                        older_frame_indices=list(range(older_start, older_end)),
                        mid_frame_indices=mid_indices,
                        action_frame_indices=list(range(action_start, target_index + 1)),
                        nominal_fps=nominal_fps,
                        frame_source=frames[target_index].frame_source,
                    )
                )
        return samples

    @staticmethod
    def _load_rgb_frames(video_path: Path, frame_indices: list[int]) -> torch.Tensor:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise FileNotFoundError(f"Unable to open blackbox video: {video_path}")

        frames: list[torch.Tensor] = []
        try:
            for frame_index in frame_indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                ok, frame_bgr = capture.read()
                if not ok or frame_bgr is None:
                    raise RuntimeError(
                        f"Unable to read frame {frame_index} from {video_path.name}."
                    )
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frames.append(
                    torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
                )
        finally:
            capture.release()
        return torch.stack(frames, dim=0)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        manifest = _read_json(sample.metadata_file)
        frames = [
            BlackboxFrameRecord.from_dict(frame_payload)
            for frame_payload in manifest.get("frames", [])
        ]
        video_path = sample.video_file
        nominal_fps = float(sample.nominal_fps)
        rgb_recent = self._load_rgb_frames(video_path, sample.recent_frame_indices)
        rgb_older = self._load_rgb_frames(video_path, sample.older_frame_indices)
        rgb_mid = self._load_rgb_frames(video_path, sample.mid_frame_indices)

        return {
            "clip_id": sample.clip_id,
            "metadata_path": sample.metadata_path,
            "video_path": sample.video_path,
            "target_frame_index": sample.target_frame_index,
            "frame_source": sample.frame_source,
            "rgb_recent": rgb_recent,
            "dt_recent": _frame_dt_tensor(
                frames,
                sample.recent_frame_indices,
                nominal_fps,
            ),
            "rgb_older": rgb_older,
            "dt_older": _frame_dt_tensor(
                frames,
                sample.older_frame_indices,
                nominal_fps,
            ),
            "rgb_mid": rgb_mid,
            "dt_mid": _frame_dt_tensor(
                frames,
                sample.mid_frame_indices,
                nominal_fps,
                relative_to_selected_prev=True,
            ),
            "actions_hist": _action_history_from_frames(
                frames,
                sample.action_frame_indices,
            ),
            "dt_hist": _frame_dt_tensor(
                frames,
                sample.action_frame_indices,
                nominal_fps,
            ),
            "route_polyline": None,
            "nav_cmd": None,
            "reasoner_tok": None,
        }
