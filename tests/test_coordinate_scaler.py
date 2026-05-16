from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from edge_vision.core.detection import Detection
from edge_vision.core.frame import PreprocessedFrame
from edge_vision.postprocessing.coordinate_scaler import (
    scale_detection_coordinates,
    scale_detections_coordinates,
)


def test_scale_detection_coordinates_to_original_frame_size() -> None:
    detection = _detection(x_min=10, y_min=20, x_max=100, y_max=120)

    scaled = scale_detection_coordinates(detection, _frame())

    assert (scaled.x_min, scaled.y_min, scaled.x_max, scaled.y_max) == (
        20,
        40,
        200,
        240,
    )


def test_scale_detection_coordinates_clamps_negative_values() -> None:
    detection = _detection(x_min=-10, y_min=-5, x_max=20, y_max=30)

    scaled = scale_detection_coordinates(detection, _frame())

    assert scaled.x_min == 0
    assert scaled.y_min == 0
    assert scaled.x_max == 40
    assert scaled.y_max == 60


def test_scale_detection_coordinates_clamps_values_to_frame_boundary() -> None:
    detection = _detection(x_min=10, y_min=20, x_max=400, y_max=300)

    scaled = scale_detection_coordinates(detection, _frame())

    assert scaled.x_max == 639
    assert scaled.y_max == 479


def test_scale_detection_coordinates_preserves_detection_metadata() -> None:
    detection = Detection(
        class_id=7,
        class_name="person",
        confidence=0.88,
        x_min=1,
        y_min=2,
        x_max=3,
        y_max=4,
    )

    scaled = scale_detection_coordinates(detection, _frame())

    assert scaled.class_id == 7
    assert scaled.class_name == "person"
    assert scaled.confidence == 0.88


def test_scale_detection_coordinates_does_not_mutate_original_detection() -> None:
    detection = _detection(x_min=10, y_min=20, x_max=100, y_max=120)
    original = detection

    scaled = scale_detection_coordinates(detection, _frame())

    assert detection == original
    assert scaled is not detection


def test_scale_detection_coordinates_normalizes_reversed_coordinates() -> None:
    detection = _detection(x_min=100, y_min=120, x_max=10, y_max=20)

    scaled = scale_detection_coordinates(detection, _frame())

    assert (scaled.x_min, scaled.y_min, scaled.x_max, scaled.y_max) == (
        20,
        40,
        200,
        240,
    )


def test_scale_detections_coordinates_handles_empty_input() -> None:
    assert scale_detections_coordinates([], _frame()) == []


def _frame() -> PreprocessedFrame:
    return PreprocessedFrame(
        input_tensor=np.zeros((1, 240, 320, 3), dtype=np.uint8),
        original_width=640,
        original_height=480,
        input_width=320,
        input_height=240,
        scale_x=2.0,
        scale_y=2.0,
    )


def _detection(x_min: int, y_min: int, x_max: int, y_max: int) -> Detection:
    return Detection(
        class_id=1,
        class_name="object",
        confidence=0.9,
        x_min=x_min,
        y_min=y_min,
        x_max=x_max,
        y_max=y_max,
    )
