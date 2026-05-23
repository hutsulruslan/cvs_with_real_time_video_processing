from __future__ import annotations

from threading import Condition, Lock

from edge_vision.core.frame import FramePacket
from edge_vision.core.result import FrameResult


class LatestFrameBuffer:
    """Keep only the newest captured frame for realtime processing."""

    def __init__(self) -> None:
        self._condition = Condition(Lock())
        self._frame: FramePacket | None = None
        self._dropped_frames = 0

    def put(self, frame: FramePacket) -> None:
        """Store a frame, replacing any unconsumed frame already present."""
        with self._condition:
            if self._frame is not None:
                self._dropped_frames += 1
            self._frame = frame
            self._condition.notify()

    def get_latest(self) -> FramePacket | None:
        """Return the latest frame without removing it."""
        with self._condition:
            return self._frame

    def pop_latest(self) -> FramePacket | None:
        """Return the latest frame and clear the buffer."""
        with self._condition:
            frame = self._frame
            self._frame = None
            return frame

    def wait_pop_latest(self, timeout: float | None = None) -> FramePacket | None:
        """Wait briefly for the latest frame, then return and clear it."""
        with self._condition:
            if self._frame is None:
                self._condition.wait(timeout)
            frame = self._frame
            self._frame = None
            return frame

    def clear(self) -> None:
        """Clear the stored frame while preserving counters."""
        with self._condition:
            self._frame = None

    def reset(self) -> None:
        """Clear the stored frame and reset counters."""
        with self._condition:
            self._frame = None
            self._dropped_frames = 0

    @property
    def dropped_frames(self) -> int:
        """Number of frames overwritten before they were consumed."""
        with self._condition:
            return self._dropped_frames


class LatestResultStore:
    """Keep only the newest completed frame result for rendering."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._result: FrameResult | None = None
        self._replaced_results = 0

    def put(self, result: FrameResult) -> None:
        """Store a result, replacing any previous result."""
        with self._lock:
            if self._result is not None:
                self._replaced_results += 1
            self._result = result

    def get_latest(self) -> FrameResult | None:
        """Return the latest result without removing it."""
        with self._lock:
            return self._result

    def clear(self) -> None:
        """Clear the stored result while preserving counters."""
        with self._lock:
            self._result = None

    def reset(self) -> None:
        """Clear the stored result and reset counters."""
        with self._lock:
            self._result = None
            self._replaced_results = 0

    @property
    def replaced_results(self) -> int:
        """Number of stored results replaced by newer results."""
        with self._lock:
            return self._replaced_results
