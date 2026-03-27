from __future__ import annotations

import ctypes
import time

_TIMERR_NOERROR = 0
_WINMM = ctypes.WinDLL("winmm") if hasattr(ctypes, "WinDLL") else None

if _WINMM is not None:
    _WINMM.timeBeginPeriod.argtypes = [ctypes.c_uint]
    _WINMM.timeBeginPeriod.restype = ctypes.c_uint
    _WINMM.timeEndPeriod.argtypes = [ctypes.c_uint]
    _WINMM.timeEndPeriod.restype = ctypes.c_uint


class HighResolutionTimer:
    def __init__(self, period_ms: int = 1):
        self.period_ms = int(period_ms)
        self._active = False

    def __enter__(self) -> "HighResolutionTimer":
        self.start()
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        self.close()

    def start(self) -> None:
        if self._active or _WINMM is None:
            return
        self._active = (
            int(_WINMM.timeBeginPeriod(ctypes.c_uint(self.period_ms)))
            == _TIMERR_NOERROR
        )

    def close(self) -> None:
        if not self._active or _WINMM is None:
            return
        _WINMM.timeEndPeriod(ctypes.c_uint(self.period_ms))
        self._active = False


def sleep_until(deadline: float) -> None:
    remaining = deadline - time.perf_counter()
    if remaining > 0.0:
        time.sleep(remaining)


def advance_fixed_deadline(next_deadline: float, interval_seconds: float) -> float:
    next_deadline += interval_seconds
    now = time.perf_counter()
    if now - next_deadline > interval_seconds:
        return now + interval_seconds
    return next_deadline
