from __future__ import annotations

from time import perf_counter_ns
from typing import Any, Callable

from edge_vision.core.errors import VideoSourceError
from edge_vision.core.frame import FramePacket
from edge_vision.video.video_source import VideoSource


CaptureFactory = Callable[[int], Any]
TimeProvider = Callable[[], int]
CAP_PROP_FRAME_WIDTH = 3
CAP_PROP_FRAME_HEIGHT = 4


class OpenCVCameraSource(VideoSource):
    """Video source that reads frames from an OpenCV camera index."""

    def __init__(
        self,
        camera_index: int,
        width: int,
        height: int,
        capture_factory: CaptureFactory | None = None,
        time_provider_ns: TimeProvider | None = None,
    ) -> None:
        self._camera_index = camera_index
        self._width = width
        self._height = height
        self._capture_factory = capture_factory
        self._time_provider_ns = time_provider_ns or perf_counter_ns
        self._capture: Any | None = None
        self._frame_id = 0

    def open(self) -> None:
        factory = self._capture_factory or _load_capture_factory()
        capture = factory(self._camera_index)
        if not capture.isOpened():
            raise VideoSourceError(f"Cannot open camera index {self._camera_index}.")

        capture.set(CAP_PROP_FRAME_WIDTH, self._width)
        capture.set(CAP_PROP_FRAME_HEIGHT, self._height)
        self._capture = capture
        self._frame_id = 0

    def read(self) -> FramePacket | None:
        if self._capture is None:
            raise VideoSourceError("Camera source is not opened.")

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
        raise VideoSourceError("OpenCV is required for camera input.") from error
    return cv2.VideoCapture
