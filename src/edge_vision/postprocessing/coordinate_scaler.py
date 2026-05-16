from __future__ import annotations

from collections.abc import Iterable

from edge_vision.core.detection import Detection
from edge_vision.core.frame import PreprocessedFrame


def scale_detection_coordinates(
    detection: Detection, frame: PreprocessedFrame
) -> Detection:
    """Scale one detection from model input coordinates to original frame size."""
    _validate_frame_bounds(frame)
    x_min, x_max = sorted((detection.x_min, detection.x_max))
    y_min, y_max = sorted((detection.y_min, detection.y_max))

    return Detection(
        class_id=detection.class_id,
        class_name=detection.class_name,
        confidence=detection.confidence,
        x_min=_clamp(round(x_min * frame.scale_x), 0, frame.original_width - 1),
        y_min=_clamp(round(y_min * frame.scale_y), 0, frame.original_height - 1),
        x_max=_clamp(round(x_max * frame.scale_x), 0, frame.original_width - 1),
        y_max=_clamp(round(y_max * frame.scale_y), 0, frame.original_height - 1),
    )


def scale_detections_coordinates(
    detections: Iterable[Detection], frame: PreprocessedFrame
) -> list[Detection]:
    """Scale a list of detections to original frame coordinates."""
    return [scale_detection_coordinates(detection, frame) for detection in detections]


def _validate_frame_bounds(frame: PreprocessedFrame) -> None:
    if frame.original_width <= 0 or frame.original_height <= 0:
        raise ValueError("Original frame dimensions must be positive.")
    if frame.input_width <= 0 or frame.input_height <= 0:
        raise ValueError("Model input dimensions must be positive.")
    if frame.scale_x <= 0.0 or frame.scale_y <= 0.0:
        raise ValueError("Frame scale values must be positive.")


def _clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(value, upper))
