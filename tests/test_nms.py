from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from edge_vision.core.detection import Detection
from edge_vision.postprocessing.nms import (
    apply_non_max_suppression,
    calculate_iou,
)


def test_calculate_iou_for_overlapping_boxes() -> None:
    assert calculate_iou(
        _detection(0, 0, 20, 20),
        _detection(10, 10, 30, 30),
    ) == 100 / 700


def test_nms_suppresses_lower_confidence_same_class_overlap() -> None:
    detections = [
        _detection(0, 0, 20, 20, confidence=0.7, class_id=1),
        _detection(1, 1, 21, 21, confidence=0.9, class_id=1),
        _detection(0, 0, 20, 20, confidence=0.6, class_id=2),
    ]

    kept = apply_non_max_suppression(detections, iou_threshold=0.5)

    assert [detection.confidence for detection in kept] == [0.9, 0.6]
    assert [detection.class_id for detection in kept] == [1, 2]


def _detection(
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int,
    confidence: float = 0.9,
    class_id: int = 1,
) -> Detection:
    return Detection(
        class_id=class_id,
        class_name=f"class-{class_id}",
        confidence=confidence,
        x_min=x_min,
        y_min=y_min,
        x_max=x_max,
        y_max=y_max,
    )
