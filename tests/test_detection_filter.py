from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from edge_vision.core.detection import Detection
from edge_vision.postprocessing.detection_filter import (
    filter_by_confidence,
    limit_detections,
)


def test_filter_by_confidence_keeps_detections_above_threshold() -> None:
    detections = [_detection(0.2), _detection(0.7), _detection(0.9)]

    filtered = filter_by_confidence(detections, confidence_threshold=0.5)

    assert [detection.confidence for detection in filtered] == [0.7, 0.9]


def test_filter_by_confidence_keeps_detection_equal_to_threshold() -> None:
    detections = [_detection(0.4), _detection(0.5)]

    filtered = filter_by_confidence(detections, confidence_threshold=0.5)

    assert [detection.confidence for detection in filtered] == [0.5]


def test_filter_by_confidence_handles_empty_input() -> None:
    assert filter_by_confidence([], confidence_threshold=0.5) == []


def test_limit_detections_keeps_highest_confidence_detections() -> None:
    detections = [_detection(0.6), _detection(0.9), _detection(0.7)]

    limited = limit_detections(detections, max_detections=2)

    assert [detection.confidence for detection in limited] == [0.9, 0.7]


def test_limit_detections_is_stable_for_equal_confidence_values() -> None:
    first = _detection(0.8, class_name="first")
    second = _detection(0.8, class_name="second")
    third = _detection(0.7, class_name="third")

    limited = limit_detections([first, second, third], max_detections=2)

    assert limited == [first, second]


def test_limit_detections_does_not_limit_when_value_is_none_or_non_positive() -> None:
    detections = [_detection(0.6), _detection(0.9)]

    assert limit_detections(detections, max_detections=None) == detections
    assert limit_detections(detections, max_detections=0) == detections
    assert limit_detections(detections, max_detections=-1) == detections


def test_detection_filter_does_not_mutate_original_list() -> None:
    detections = [_detection(0.6), _detection(0.9), _detection(0.7)]
    original = list(detections)

    filter_by_confidence(detections, confidence_threshold=0.8)
    limit_detections(detections, max_detections=1)

    assert detections == original


def _detection(confidence: float, class_name: str = "object") -> Detection:
    return Detection(
        class_id=1,
        class_name=class_name,
        confidence=confidence,
        x_min=1,
        y_min=2,
        x_max=30,
        y_max=40,
    )
