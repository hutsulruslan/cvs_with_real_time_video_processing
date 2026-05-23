from __future__ import annotations

from dataclasses import dataclass
from threading import Condition, Lock
from time import perf_counter
from typing import Callable

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


@dataclass(frozen=True, slots=True)
class LowLatencyRuntimeSnapshot:
    """Thread-safe snapshot of low-latency runtime counters."""

    captured_frames: int
    processed_frames: int
    latest_captured_frame_id: int | None
    latest_processed_frame_id: int | None
    capture_fps: float | None
    inference_fps: float | None


class LowLatencyRuntimeStats:
    """Collect capture and inference counters for low-latency runs."""

    def __init__(
        self,
        time_provider_s: Callable[[], float] = perf_counter,
    ) -> None:
        self._lock = Lock()
        self._time_provider_s = time_provider_s
        self._captured_frames = 0
        self._processed_frames = 0
        self._latest_captured_frame_id: int | None = None
        self._latest_processed_frame_id: int | None = None
        self._capture_started_at_s: float | None = None
        self._capture_completed_at_s: float | None = None
        self._inference_started_at_s: float | None = None
        self._inference_completed_at_s: float | None = None

    def reset(self) -> None:
        """Clear counters and timing markers."""
        with self._lock:
            self._captured_frames = 0
            self._processed_frames = 0
            self._latest_captured_frame_id = None
            self._latest_processed_frame_id = None
            self._capture_started_at_s = None
            self._capture_completed_at_s = None
            self._inference_started_at_s = None
            self._inference_completed_at_s = None

    def start_capture(self) -> None:
        """Mark capture worker start time."""
        with self._lock:
            self._capture_started_at_s = self._time_provider_s()
            self._capture_completed_at_s = None

    def finish_capture(self) -> None:
        """Mark capture worker completion time."""
        with self._lock:
            self._capture_completed_at_s = self._time_provider_s()

    def record_captured_frame(self, frame: FramePacket) -> None:
        """Record one frame read by the capture worker."""
        with self._lock:
            self._captured_frames += 1
            self._latest_captured_frame_id = frame.frame_id

    def start_inference(self) -> None:
        """Mark inference worker start time."""
        with self._lock:
            self._inference_started_at_s = self._time_provider_s()
            self._inference_completed_at_s = None

    def finish_inference(self) -> None:
        """Mark inference worker completion time."""
        with self._lock:
            self._inference_completed_at_s = self._time_provider_s()

    def record_processed_result(self, result: FrameResult) -> int:
        """Record one processed result and return the processed count."""
        with self._lock:
            self._processed_frames += 1
            self._latest_processed_frame_id = result.frame_id
            return self._processed_frames

    def snapshot(self) -> LowLatencyRuntimeSnapshot:
        """Return immutable counters and derived rates."""
        with self._lock:
            return LowLatencyRuntimeSnapshot(
                captured_frames=self._captured_frames,
                processed_frames=self._processed_frames,
                latest_captured_frame_id=self._latest_captured_frame_id,
                latest_processed_frame_id=self._latest_processed_frame_id,
                capture_fps=_rate(
                    self._captured_frames,
                    self._capture_started_at_s,
                    self._capture_completed_at_s,
                ),
                inference_fps=_rate(
                    self._processed_frames,
                    self._inference_started_at_s,
                    self._inference_completed_at_s,
                ),
            )

    def report_fields(
        self,
        *,
        dropped_frames: int,
        replaced_results: int,
    ) -> dict[str, int | float | None]:
        """Return report-ready low-latency counters."""
        snapshot = self.snapshot()
        dropped_ratio = (
            dropped_frames / snapshot.captured_frames
            if snapshot.captured_frames > 0
            else None
        )
        return {
            "captured_frames": snapshot.captured_frames,
            "dropped_frames": dropped_frames,
            "dropped_frame_ratio": dropped_ratio,
            "latest_captured_frame_id": snapshot.latest_captured_frame_id,
            "latest_processed_frame_id": snapshot.latest_processed_frame_id,
            "capture_fps": snapshot.capture_fps,
            "inference_fps": snapshot.inference_fps,
            "replaced_results": replaced_results,
        }


def _rate(
    count: int,
    started_at_s: float | None,
    completed_at_s: float | None,
) -> float | None:
    if count <= 0 or started_at_s is None or completed_at_s is None:
        return None
    elapsed_s = completed_at_s - started_at_s
    if elapsed_s <= 0:
        return None
    return count / elapsed_s
