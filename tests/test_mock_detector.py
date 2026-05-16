from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from edge_vision.core.detection import Detection
from edge_vision.core.frame import PreprocessedFrame
from edge_vision.inference.detector import ObjectDetector
from edge_vision.inference.mock_detector import MockObjectDetector


def test_mock_detector_implements_detector_interface() -> None:
    detector = MockObjectDetector()

    assert isinstance(detector, ObjectDetector)


def test_mock_detector_returns_default_detection_objects() -> None:
    detector = MockObjectDetector()

    detections = detector.detect(_preprocessed_frame())

    assert len(detections) == 1
    assert isinstance(detections[0], Detection)
    assert detections[0].class_name == "mock-object"
    assert detections[0].confidence == 0.9


def test_mock_detector_returns_configured_detections() -> None:
    expected_detection = Detection(
        class_id=2,
        class_name="configured-object",
        confidence=0.75,
        x_min=1,
        y_min=2,
        x_max=30,
        y_max=40,
    )
    detector = MockObjectDetector(detections=[expected_detection])

    detections = detector.detect(_preprocessed_frame())

    assert detections == [expected_detection]


def test_empty_mock_detector_returns_empty_list() -> None:
    detector = MockObjectDetector(detections=[])

    assert detector.detect(_preprocessed_frame()) == []


def test_mock_detector_returns_a_new_list_each_time() -> None:
    detector = MockObjectDetector()

    first_result = detector.detect(_preprocessed_frame())
    first_result.clear()
    second_result = detector.detect(_preprocessed_frame())

    assert len(second_result) == 1


def test_mock_detector_rejects_non_detection_values() -> None:
    with pytest.raises(TypeError, match="Detection"):
        MockObjectDetector(detections=[object()])


def _preprocessed_frame() -> PreprocessedFrame:
    return PreprocessedFrame(
        input_tensor=np.zeros((1, 320, 320, 3), dtype=np.uint8),
        original_width=640,
        original_height=480,
        input_width=320,
        input_height=320,
        scale_x=2.0,
        scale_y=1.5,
    )
