from __future__ import annotations

from time import perf_counter
from typing import Callable


class FPSCounter:
    """Calculate frames per second using an injectable time source."""

    def __init__(self, time_provider: Callable[[], float] = perf_counter) -> None:
        self._time_provider = time_provider
        self._start_time = self._time_provider()
        self._frame_count = 0
        self._fps = 0.0

    def reset(self) -> None:
        """Clear collected frame count and timing state."""
        self._start_time = self._time_provider()
        self._frame_count = 0
        self._fps = 0.0

    def update(self, frames: int = 1) -> float:
        """Add processed frames and return the current FPS estimate."""
        if frames <= 0:
            raise ValueError("Frame count increment must be positive.")

        self._frame_count += frames
        elapsed_seconds = self._time_provider() - self._start_time
        if elapsed_seconds <= 0.0:
            self._fps = 0.0
        else:
            self._fps = self._frame_count / elapsed_seconds
        return self._fps

    def get_fps(self) -> float:
        """Return the last calculated FPS value."""
        return self._fps
