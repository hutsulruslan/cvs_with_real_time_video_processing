from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from edge_vision.core.detection import Detection
from edge_vision.postprocessing.box_smoother import BoxSmoother


def test_box_smoother_smooths_matching_detection_coordinates() -> None:
    smoother = BoxSmoother(alpha=0.5, iou_threshold=0.1)

    first = smoother.smooth([_detection(x_min=10, x_max=30)])
    second = smoother.smooth([_detection(x_min=14, x_max=34)])

    assert first[0].x_min == 10
    assert second[0].x_min == 12
    assert second[0].x_max == 32


def test_box_smoother_does_not_mutate_input_detections() -> None:
    smoother = BoxSmoother(alpha=0.5, iou_threshold=0.1)
    first = _detection(x_min=10, x_max=30)
    second = _detection(x_min=14, x_max=34)

    smoother.smooth([first])
    smoothed = smoother.smooth([second])

    assert second.x_min == 14
    assert second.x_max == 34
    assert smoothed[0] is not second


def test_box_smoother_keeps_unmatched_detection_unchanged() -> None:
    smoother = BoxSmoother(alpha=0.5, iou_threshold=0.9)

    smoother.smooth([_detection(x_min=10, x_max=30)])
    second = smoother.smooth([_detection(x_min=80, x_max=100)])

    assert second[0].x_min == 80
    assert second[0].x_max == 100


def _detection(x_min: int, x_max: int) -> Detection:
    return Detection(
        class_id=1,
        class_name="object",
        confidence=0.9,
        x_min=x_min,
        y_min=10,
        x_max=x_max,
        y_max=30,
    )
