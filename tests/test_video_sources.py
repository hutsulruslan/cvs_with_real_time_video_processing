from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from edge_vision.config.settings import VideoSettings
from edge_vision.core.errors import VideoSourceError
from edge_vision.video.opencv_camera_source import OpenCVCameraSource
from edge_vision.video.source_factory import create_video_source
from edge_vision.video.video_file_source import VideoFileSource
from edge_vision.video.video_stream_source import VideoStreamSource


def test_camera_source_reads_frames_and_releases_capture() -> None:
    fake_capture = FakeCapture(frames=["frame-a", "frame-b"])
    source = OpenCVCameraSource(
        camera_index=1,
        width=640,
        height=480,
        capture_factory=lambda index: fake_capture,
        time_provider_ns=SequenceClockNs([10_000_000, 20_000_000]).now,
    )

    source.open()
    first = source.read()
    second = source.read()
    end = source.read()
    source.release()

    assert first is not None
    assert second is not None
    assert first.frame_id == 0
    assert second.frame_id == 1
    assert first.timestamp_ms == 10.0
    assert first.timestamp_ns == 10_000_000
    assert second.timestamp_ms == 20.0
    assert second.timestamp_ns == 20_000_000
    assert first.original_frame == "frame-a"
    assert end is None
    assert fake_capture.released is True
    assert fake_capture.properties[3] == 640
    assert fake_capture.properties[4] == 480


def test_camera_source_rejects_unopened_capture() -> None:
    source = OpenCVCameraSource(
        camera_index=5,
        width=640,
        height=480,
        capture_factory=lambda index: FakeCapture(opened=False),
    )

    with pytest.raises(VideoSourceError, match="camera index"):
        source.open()


def test_video_file_source_reads_frames_from_existing_path(tmp_path: Path) -> None:
    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"placeholder")
    fake_capture = FakeCapture(frames=["frame-a"])
    source = VideoFileSource(
        file_path=video_path,
        capture_factory=lambda path: fake_capture,
        time_provider_ns=SequenceClockNs([30_000_000]).now,
    )

    source.open()
    packet = source.read()
    end = source.read()
    source.release()

    assert packet is not None
    assert packet.frame_id == 0
    assert packet.timestamp_ms == 30.0
    assert packet.timestamp_ns == 30_000_000
    assert packet.original_frame == "frame-a"
    assert end is None
    assert fake_capture.released is True


def test_video_file_source_rejects_missing_path(tmp_path: Path) -> None:
    source = VideoFileSource(
        file_path=tmp_path / "missing.mp4",
        capture_factory=lambda path: FakeCapture(),
    )

    with pytest.raises(VideoSourceError, match="does not exist"):
        source.open()


def test_create_video_source_uses_configured_source_type() -> None:
    camera_settings = VideoSettings(
        source_type="camera",
        camera_index=0,
        file_path="assets/samples/sample_video.mp4",
        width=640,
        height=480,
    )
    file_settings = VideoSettings(
        source_type="file",
        camera_index=0,
        file_path="assets/samples/sample_video.mp4",
        width=640,
        height=480,
    )
    stream_settings = VideoSettings(
        source_type="stream",
        camera_index=0,
        file_path="assets/samples/sample_video.mp4",
        width=640,
        height=480,
        stream_url="http://example.local:8080/video",
    )

    assert isinstance(create_video_source(camera_settings), OpenCVCameraSource)
    assert isinstance(create_video_source(file_settings), VideoFileSource)
    assert isinstance(create_video_source(stream_settings), VideoStreamSource)


def test_create_video_source_keeps_picamera2_explicitly_unimplemented() -> None:
    settings = VideoSettings(
        source_type="picamera2",
        camera_index=0,
        file_path="assets/samples/sample_video.mp4",
        width=640,
        height=480,
    )

    with pytest.raises(VideoSourceError, match="not implemented"):
        create_video_source(settings)


class FakeCapture:
    def __init__(self, frames: list[Any] | None = None, opened: bool = True) -> None:
        self._frames = list(frames or [])
        self._opened = opened
        self.properties: dict[int, Any] = {}
        self.released = False

    def isOpened(self) -> bool:
        return self._opened

    def read(self) -> tuple[bool, Any | None]:
        if not self._frames:
            return False, None
        return True, self._frames.pop(0)

    def release(self) -> None:
        self.released = True

    def set(self, property_id: int, value: Any) -> None:
        self.properties[property_id] = value


class SequenceClockNs:
    def __init__(self, values: list[int]) -> None:
        self._values = list(values)

    def now(self) -> int:
        return self._values.pop(0)
