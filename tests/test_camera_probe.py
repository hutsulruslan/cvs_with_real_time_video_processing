from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from edge_vision.tools.camera_probe import (
    CameraProbeResult,
    format_probe_results,
    probe_camera_indexes,
)


def test_probe_camera_indexes_reports_frame_size_and_releases_capture() -> None:
    capture = FakeCapture(frame=np.zeros((480, 640, 3), dtype=np.uint8))

    results = probe_camera_indexes(
        max_index=0,
        capture_factory=lambda index: capture,
    )

    assert results == [CameraProbeResult(0, True, True, 640, 480)]
    assert capture.released is True


def test_probe_camera_indexes_continues_when_index_fails() -> None:
    captures = {
        0: FakeCapture(opened=False),
        2: FakeCapture(frame=np.zeros((240, 320, 3), dtype=np.uint8)),
    }

    def capture_factory(index: int) -> FakeCapture:
        if index == 1:
            raise RuntimeError("camera unavailable")
        return captures[index]

    results = probe_camera_indexes(max_index=2, capture_factory=capture_factory)

    assert [result.index for result in results] == [0, 1, 2]
    assert results[0] == CameraProbeResult(0, False, False, None, None)
    assert results[1].error == "camera unavailable"
    assert results[2] == CameraProbeResult(2, True, True, 320, 240)
    assert captures[0].released is True
    assert captures[2].released is True


def test_probe_camera_indexes_uses_capture_size_when_frame_has_no_shape() -> None:
    capture = FakeCapture(frame="frame", width=800, height=600)

    results = probe_camera_indexes(
        max_index=0,
        capture_factory=lambda index: capture,
    )

    assert results == [CameraProbeResult(0, True, True, 800, 600)]


def test_probe_camera_indexes_reports_read_error_without_crashing() -> None:
    capture = FakeCapture(read_error=RuntimeError("read failed"), width=640, height=480)

    results = probe_camera_indexes(
        max_index=0,
        capture_factory=lambda index: capture,
    )

    assert results == [CameraProbeResult(0, True, False, 640, 480, "read failed")]
    assert capture.released is True


def test_probe_camera_indexes_handles_negative_max_index() -> None:
    results = probe_camera_indexes(max_index=-1, capture_factory=lambda index: None)

    assert results == []


def test_format_probe_results_is_readable() -> None:
    results = [
        CameraProbeResult(0, True, True, 640, 480),
        CameraProbeResult(1, False, False, None, None, "camera unavailable"),
    ]

    formatted = format_probe_results(results)

    assert "Camera probe results:" in formatted
    assert "index 0" in formatted
    assert "opened=yes" in formatted
    assert "size=640x480" in formatted
    assert "error=camera unavailable" in formatted


class FakeCapture:
    def __init__(
        self,
        frame: Any | None = None,
        opened: bool = True,
        width: int = 0,
        height: int = 0,
        read_error: Exception | None = None,
    ) -> None:
        self._frame = frame
        self._opened = opened
        self._width = width
        self._height = height
        self._read_error = read_error
        self.released = False

    def isOpened(self) -> bool:
        return self._opened

    def read(self) -> tuple[bool, Any | None]:
        if self._read_error is not None:
            raise self._read_error
        return self._frame is not None, self._frame

    def get(self, property_id: int) -> int:
        if property_id == 3:
            return self._width
        if property_id == 4:
            return self._height
        return 0

    def release(self) -> None:
        self.released = True
