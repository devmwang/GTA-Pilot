from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import cv2
import torch

# Keep the worker-local clip cache modest: Stage 1 samples already materialize large
# recent/older/mid unions, and overly large decoded-frame caches can consume several
# additional gigabytes of host RAM per active clip. Bound both the per-clip frame cache
# and the number of live clip decoders so host RAM cannot grow without limit.
_MAX_CACHED_FRAMES = 32
_MAX_DECODER_SESSIONS = 2
_DECODER_SESSIONS: OrderedDict[Path, "_ClipDecoder"] = OrderedDict()


class _ClipDecoder:
    def __init__(self, video_path: Path):
        self.video_path = video_path
        self.capture: cv2.VideoCapture | None = None
        self.next_frame_index = 0
        self.cache: OrderedDict[int, torch.Tensor] = OrderedDict()

    def _ensure_open(self) -> cv2.VideoCapture:
        if self.capture is None:
            capture = cv2.VideoCapture(str(self.video_path))
            if not capture.isOpened():
                raise FileNotFoundError(f"Unable to open video: {self.video_path}")
            self.capture = capture
            self.next_frame_index = 0
        return self.capture

    def _reopen_at(self, frame_index: int) -> cv2.VideoCapture:
        if self.capture is not None:
            self.capture.release()
        capture = cv2.VideoCapture(str(self.video_path))
        if not capture.isOpened():
            raise FileNotFoundError(f"Unable to open video: {self.video_path}")
        if frame_index > 0:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        self.capture = capture
        self.next_frame_index = frame_index
        return capture

    def _cache_frame(self, frame_index: int, frame_rgb: torch.Tensor) -> None:
        self.cache[frame_index] = frame_rgb
        self.cache.move_to_end(frame_index)
        while len(self.cache) > _MAX_CACHED_FRAMES:
            self.cache.popitem(last=False)

    def _decode_frame(self, frame_index: int) -> torch.Tensor:
        cached = self.cache.get(frame_index)
        if cached is not None:
            self.cache.move_to_end(frame_index)
            return cached

        capture = self._ensure_open()
        if frame_index < self.next_frame_index:
            capture = self._reopen_at(frame_index)
        elif frame_index > self.next_frame_index:
            while self.next_frame_index < frame_index:
                ok, _ = capture.read()
                if not ok:
                    raise RuntimeError(
                        f"Unable to advance to frame {frame_index} in {self.video_path}"
                    )
                self.next_frame_index += 1

        ok, frame_bgr = capture.read()
        if not ok or frame_bgr is None:
            raise RuntimeError(f"Unable to decode frame {frame_index} from {self.video_path}")
        self.next_frame_index = frame_index + 1
        frame_rgb = torch.from_numpy(
            cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        ).permute(2, 0, 1).contiguous()
        self._cache_frame(frame_index, frame_rgb)
        return frame_rgb

    def decode_union(self, frame_indices: list[int]) -> dict[int, torch.Tensor]:
        requested = sorted(set(int(index) for index in frame_indices))
        return {frame_index: self._decode_frame(frame_index) for frame_index in requested}

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.cache.clear()
        self.next_frame_index = 0


def _decoder_for(video_path: str | Path) -> _ClipDecoder:
    resolved = Path(video_path).resolve()
    decoder = _DECODER_SESSIONS.get(resolved)
    if decoder is None:
        decoder = _ClipDecoder(resolved)
        _DECODER_SESSIONS[resolved] = decoder
        while len(_DECODER_SESSIONS) > _MAX_DECODER_SESSIONS:
            stale_path, stale_decoder = _DECODER_SESSIONS.popitem(last=False)
            if stale_path != resolved:
                stale_decoder.close()
    else:
        _DECODER_SESSIONS.move_to_end(resolved)
    return decoder


def decode_rgb_frame_union(
    video_path: str | Path,
    frame_indices: list[int],
) -> dict[int, torch.Tensor]:
    return _decoder_for(video_path).decode_union(frame_indices)
