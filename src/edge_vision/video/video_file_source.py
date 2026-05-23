from __future__ import annotations

from pathlib import Path
from time import perf_counter_ns
from typing import Any, Callable

from edge_vision.core.errors import VideoSourceError
from edge_vision.core.frame import FramePacket
from edge_vision.video.video_source import VideoSource


CaptureFactory = Callable[[str], Any]
TimeProvider = Callable[[], int]


class VideoFileSource(VideoSource):
    """Video source that reads frames from a local video file."""

    def __init__(
        self,
        file_path: str | Path,
        capture_factory: CaptureFactory | None = None,
        time_provider_ns: TimeProvider | None = None,
    ) -> None:
        self._file_path = Path(file_path)
        self._capture_factory = capture_factory
        self._time_provider_ns = time_provider_ns or perf_counter_ns
        self._capture: Any | None = None
        self._frame_id = 0

    def open(self) -> None:
        if not self._file_path.exists():
            raise VideoSourceError(f"Video file does not exist: {self._file_path}")

        factory = self._capture_factory or _load_capture_factory()
        capture = factory(str(self._file_path))
        if not capture.isOpened():
            raise VideoSourceError(f"Cannot open video file: {self._file_path}")

        self._capture = capture
        self._frame_id = 0

    def read(self) -> FramePacket | None:
        if self._capture is None:
            raise VideoSourceError("Video file source is not opened.")

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


def _load_capture_factory() -> CaptureFactory:
    try:
        import cv2
    except ImportError as error:
        raise VideoSourceError("OpenCV is required for video file input.") from error
    return cv2.VideoCapture
