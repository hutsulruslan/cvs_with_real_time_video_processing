from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


class WindowDisplay:
    """Show rendered frames in an OpenCV window and detect quit requests."""

    def __init__(
        self,
        window_name: str = "Edge Vision System",
        quit_key: str = "q",
        cv2_module: Any | None = None,
    ) -> None:
        _validate_window_name(window_name)
        _validate_quit_key(quit_key)
        self._window_name = window_name
        self._quit_key_code = ord(quit_key) & 0xFF
        self._cv2 = cv2_module if cv2_module is not None else _load_cv2()

    def show(self, frame: NDArray[Any]) -> bool:
        """Show a frame and return True when the quit key was pressed."""
        _validate_frame(frame)
        self._cv2.imshow(self._window_name, frame)
        key_code = self._cv2.waitKey(1) & 0xFF
        return key_code == self._quit_key_code

    def close(self) -> None:
        """Close the configured OpenCV window."""
        self._cv2.destroyWindow(self._window_name)


def _load_cv2() -> Any:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("OpenCV is required for window display.") from error
    return cv2


def _validate_frame(frame: NDArray[Any]) -> None:
    if not isinstance(frame, np.ndarray):
        raise ValueError("Frame must be a NumPy array.")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Frame must have height, width, and three color channels.")


def _validate_window_name(window_name: str) -> None:
    if not isinstance(window_name, str) or not window_name.strip():
        raise ValueError("Window name must be a non-empty string.")


def _validate_quit_key(quit_key: str) -> None:
    if not isinstance(quit_key, str) or len(quit_key) != 1:
        raise ValueError("Quit key must be a single character.")
