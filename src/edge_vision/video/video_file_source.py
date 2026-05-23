from __future__ import annotations

from pathlib import Path
from time import perf_counter, perf_counter_ns, sleep
from typing import Any, Callable

from edge_vision.core.errors import VideoSourceError
from edge_vision.core.frame import FramePacket
from edge_vision.video.video_source import VideoSource


CaptureFactory = Callable[[str], Any]
TimeProvider = Callable[[], int]
PaceClock = Callable[[], float]
Sleeper = Callable[[float], None]


class VideoFileSource(VideoSource):
    """Video source that reads frames from a local video file."""

    def __init__(
        self,
        file_path: str | Path,
        capture_factory: CaptureFactory | None = None,
        time_provider_ns: TimeProvider | None = None,
        file_source_fps: float | None = None,
        time_provider_s: PaceClock | None = None,
        sleep_func: Sleeper | None = None,
    ) -> None:
        if file_source_fps is not None and (
            isinstance(file_source_fps, bool) or file_source_fps <= 0
        ):
            raise ValueError("file_source_fps must be positive or None.")
        self._file_path = Path(file_path)
        self._capture_factory = capture_factory
        self._time_provider_ns = time_provider_ns or perf_counter_ns
        self._file_source_fps = file_source_fps
        self._time_provider_s = time_provider_s or perf_counter
        self._sleep_func = sleep_func or sleep
        self._capture: Any | None = None
        self._frame_id = 0
        self._next_frame_time_s: float | None = None

    def open(self) -> None:
        if not self._file_path.exists():
            raise VideoSourceError(f"Video file does not exist: {self._file_path}")

        factory = self._capture_factory or _load_capture_factory()
        capture = factory(str(self._file_path))
        if not capture.isOpened():
            raise VideoSourceError(f"Cannot open video file: {self._file_path}")

        self._capture = capture
        self._frame_id = 0
        self._next_frame_time_s = None

    def read(self) -> FramePacket | None:
        if self._capture is None:
            raise VideoSourceError("Video file source is not opened.")

        self._wait_for_next_frame()
        success, frame = self._capture.read()
        if not success:
            return None

        timestamp_ns = self._time_provider_ns()
        packet = FramePacket(
            frame_id=self._frame_id,
            timestamp_ms=timestamp_ns / 1_000_000.0,
            original_frame=frame,
            timestamp_ns=timestamp_ns,
        )
        self._frame_id += 1
        return packet

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def _wait_for_next_frame(self) -> None:
        if self._file_source_fps is None:
            return

        now_s = self._time_provider_s()
        if self._next_frame_time_s is not None:
            wait_s = self._next_frame_time_s - now_s
            if wait_s > 0:
                self._sleep_func(wait_s)
                now_s = self._time_provider_s()
        self._next_frame_time_s = now_s + (1.0 / self._file_source_fps)


def _load_capture_factory() -> CaptureFactory:
    try:
        import cv2
    except ImportError as error:
        raise VideoSourceError("OpenCV is required for video file input.") from error
    return cv2.VideoCapture
