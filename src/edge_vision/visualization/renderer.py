from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from edge_vision.core.detection import Detection


class Renderer:
    """Draw detections and optional FPS text on image frames."""

    def __init__(self, show_fps: bool = True) -> None:
        self._show_fps = show_fps

    def render(
        self,
        frame: NDArray[Any],
        detections: list[Detection],
        fps: float | None = None,
    ) -> NDArray[Any]:
        """Return a rendered copy of the input frame."""
        _validate_frame(frame)
        rendered = frame.copy()

        for detection in detections:
            self._draw_detection(rendered, detection)

        if self._show_fps and fps is not None:
            cv2.putText(
                rendered,
                f"FPS: {fps:.2f}",
                (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        return rendered

    def _draw_detection(self, frame: NDArray[Any], detection: Detection) -> None:
        height, width = frame.shape[:2]
        x_min, x_max = sorted((detection.x_min, detection.x_max))
        y_min, y_max = sorted((detection.y_min, detection.y_max))
        x_min = _clamp(x_min, 0, width - 1)
        x_max = _clamp(x_max, 0, width - 1)
        y_min = _clamp(y_min, 0, height - 1)
        y_max = _clamp(y_max, 0, height - 1)

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        self._draw_label(frame, detection, x_min, y_min)

    def _draw_label(
        self, frame: NDArray[Any], detection: Detection, x_min: int, y_min: int
    ) -> None:
        height, width = frame.shape[:2]
        label = f"{detection.class_name} {detection.confidence:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        text_x = _clamp(x_min, 0, max(0, width - text_width - 4))
        text_y = y_min - 6
        if text_y - text_height - baseline < 0:
            text_y = y_min + text_height + baseline + 6
        text_y = _clamp(text_y, text_height + baseline, height - 1)

        box_top = _clamp(text_y - text_height - baseline, 0, height - 1)
        box_bottom = _clamp(text_y + baseline, 0, height - 1)
        box_right = _clamp(text_x + text_width + 4, 0, width - 1)

        cv2.rectangle(
            frame,
            (text_x, box_top),
            (box_right, box_bottom),
            (0, 255, 0),
            -1,
        )
        cv2.putText(
            frame,
            label,
            (text_x + 2, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )


def _validate_frame(frame: NDArray[Any]) -> None:
    if not isinstance(frame, np.ndarray):
        raise ValueError("Frame must be a NumPy array.")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Frame must have height, width, and three color channels.")


def _clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(value, upper))
