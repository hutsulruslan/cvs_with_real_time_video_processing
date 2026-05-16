from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from edge_vision.visualization.window_display import WindowDisplay


def test_show_calls_imshow_with_configured_window_name_and_frame() -> None:
    fake_cv2 = FakeCV2(wait_key_code=-1)
    frame = _frame()

    WindowDisplay(window_name="Test Window", cv2_module=fake_cv2).show(frame)

    assert fake_cv2.imshow_calls == [("Test Window", frame)]


def test_show_returns_true_when_quit_key_is_pressed() -> None:
    fake_cv2 = FakeCV2(wait_key_code=ord("q"))

    quit_requested = WindowDisplay(cv2_module=fake_cv2).show(_frame())

    assert quit_requested is True


def test_show_returns_false_when_another_key_is_pressed() -> None:
    fake_cv2 = FakeCV2(wait_key_code=ord("x"))

    quit_requested = WindowDisplay(cv2_module=fake_cv2).show(_frame())

    assert quit_requested is False


def test_show_masks_opencv_wait_key_return_value() -> None:
    fake_cv2 = FakeCV2(wait_key_code=ord("q") + 256)

    quit_requested = WindowDisplay(cv2_module=fake_cv2).show(_frame())

    assert quit_requested is True


def test_close_calls_destroy_window() -> None:
    fake_cv2 = FakeCV2(wait_key_code=-1)
    display = WindowDisplay(window_name="Close Me", cv2_module=fake_cv2)

    display.close()

    assert fake_cv2.destroy_window_calls == ["Close Me"]


def test_show_rejects_invalid_frame_shape() -> None:
    display = WindowDisplay(cv2_module=FakeCV2(wait_key_code=-1))

    with pytest.raises(ValueError, match="three color channels"):
        display.show(np.zeros((20, 20), dtype=np.uint8))


class FakeCV2:
    def __init__(self, wait_key_code: int) -> None:
        self._wait_key_code = wait_key_code
        self.imshow_calls: list[tuple[str, Any]] = []
        self.wait_key_delays: list[int] = []
        self.destroy_window_calls: list[str] = []

    def imshow(self, window_name: str, frame: np.ndarray) -> None:
        self.imshow_calls.append((window_name, frame))

    def waitKey(self, delay_ms: int) -> int:
        self.wait_key_delays.append(delay_ms)
        return self._wait_key_code

    def destroyWindow(self, window_name: str) -> None:
        self.destroy_window_calls.append(window_name)


def _frame() -> np.ndarray:
    return np.zeros((20, 30, 3), dtype=np.uint8)
