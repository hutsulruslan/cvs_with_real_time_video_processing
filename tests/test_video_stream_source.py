from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from edge_vision.core.errors import VideoSourceError
from edge_vision.video.video_stream_source import VideoStreamSource


def test_video_stream_source_reads_frames_and_releases_capture() -> None:
    fake_capture = FakeCapture(frames=["frame-a", "frame-b"])
    source = VideoStreamSource(
        stream_url="http://example.local:8080/video",
        capture_factory=lambda url: fake_capture,
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
    assert first.original_frame == "frame-a"
    assert end is None
    assert fake_capture.released is True


def test_video_stream_source_rejects_missing_stream_url() -> None:
    source = VideoStreamSource(
        stream_url="",
        capture_factory=lambda url: FakeCapture(),
    )

    with pytest.raises(VideoSourceError, match="stream_url"):
        source.open()


def test_video_stream_source_rejects_unopened_stream() -> None:
    source = VideoStreamSource(
        stream_url="rtsp://example.local/stream",
        capture_factory=lambda url: FakeCapture(opened=False),
    )

    with pytest.raises(VideoSourceError, match="video stream"):
        source.open()


class FakeCapture:
    def __init__(self, frames: list[Any] | None = None, opened: bool = True) -> None:
        self._frames = list(frames or [])
        self._opened = opened
        self.released = False

    def isOpened(self) -> bool:
        return self._opened

    def read(self) -> tuple[bool, Any | None]:
        if not self._frames:
            return False, None
        return True, self._frames.pop(0)

    def release(self) -> None:
        self.released = True
