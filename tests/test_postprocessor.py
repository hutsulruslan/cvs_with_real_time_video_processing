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
from edge_vision.postprocessing.box_smoother import BoxSmoother
from edge_vision.postprocessing.postprocessor import DetectionPostProcessor


def test_postprocessor_filters_limits_and_scales_detections() -> None:
    detections = [
        _detection(confidence=0.2, class_name="low", x_min=1),
        _detection(confidence=0.9, class_name="middle", x_min=10),
        _detection(confidence=0.95, class_name="top", x_min=20),
        _detection(confidence=0.8, class_name="kept-but-limited", x_min=30),
    ]
    postprocessor = DetectionPostProcessor(
        confidence_threshold=0.8,
        max_detections=2,
    )

    processed = postprocessor.process(detections, _frame())

    assert [detection.class_name for detection in processed] == ["top", "middle"]
    assert processed[0].x_min == 40
    assert processed[1].x_min == 20


def test_postprocessor_returns_empty_list_when_all_detections_are_filtered() -> None:
    postprocessor = DetectionPostProcessor(confidence_threshold=0.8)

    processed = postprocessor.process([_detection(confidence=0.1)], _frame())

    assert processed == []


def test_postprocessor_applies_nms_before_scaling() -> None:
    detections = [
        _detection(confidence=0.7, class_name="lower", x_min=10),
        _detection(confidence=0.9, class_name="higher", x_min=11),
    ]
    postprocessor = DetectionPostProcessor(
        confidence_threshold=0.5,
        nms_threshold=0.5,
    )

    processed = postprocessor.process(detections, _frame())

    assert [detection.class_name for detection in processed] == ["higher"]
    assert processed[0].x_min == 22


def test_postprocessor_can_smooth_scaled_boxes() -> None:
    postprocessor = DetectionPostProcessor(
        confidence_threshold=0.5,
        box_smoother=BoxSmoother(alpha=0.5, iou_threshold=0.1),
    )

    first = postprocessor.process([_detection(confidence=0.9, x_min=10)], _frame())
    second = postprocessor.process([_detection(confidence=0.9, x_min=14)], _frame())

    assert first[0].x_min == 20
    assert second[0].x_min == 24


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


def _detection(
    confidence: float,
    class_name: str = "object",
    x_min: int = 10,
) -> Detection:
    return Detection(
        class_id=1,
        class_name=class_name,
        confidence=confidence,
        x_min=x_min,
        y_min=10,
        x_max=x_min + 20,
        y_max=30,
    )
