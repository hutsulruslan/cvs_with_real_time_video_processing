from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from edge_vision.core.frame import PreprocessedFrame
from edge_vision.inference.labels_loader import LabelMap
from edge_vision.inference.tflite_detector import (
    TFLiteObjectDetector,
    parse_ssd_outputs,
)


def test_parse_ssd_outputs_converts_normalized_boxes_to_input_pixels() -> None:
    detections = parse_ssd_outputs(
        boxes=np.array([[[0.25, 0.10, 0.75, 0.50]]]),
        classes=np.array([[1]]),
        scores=np.array([[0.84]]),
        num_detections=np.array([1]),
        labels=LabelMap(("background", "person")),
        input_width=320,
        input_height=240,
    )

    assert len(detections) == 1
    assert detections[0].class_id == 1
    assert detections[0].class_name == "person"
    assert detections[0].confidence == 0.84
    assert (detections[0].x_min, detections[0].y_min) == (32, 60)
    assert (detections[0].x_max, detections[0].y_max) == (160, 180)


def test_parse_ssd_outputs_uses_num_detections_limit_and_fallback_labels() -> None:
    detections = parse_ssd_outputs(
        boxes=np.array([[[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.0, 1.0]]]),
        classes=np.array([[5, 1]]),
        scores=np.array([[0.9, 0.7]]),
        num_detections=np.array([1]),
        labels=LabelMap(("background", "person")),
        input_width=100,
        input_height=80,
    )

    assert len(detections) == 1
    assert detections[0].class_name == "class_5"
    assert detections[0].confidence == 0.9


def test_tflite_detector_uses_fake_interpreter_without_real_runtime() -> None:
    input_tensor = np.zeros((1, 320, 320, 3), dtype=np.uint8)
    frame = PreprocessedFrame(input_tensor, 640, 480, 320, 320, 2.0, 1.5)
    interpreter = FakeInterpreter(
        outputs={
            1: np.array([[[0.0, 0.0, 1.0, 1.0]]]),
            2: np.array([[1]]),
            3: np.array([[0.95]]),
            4: np.array([1]),
        }
    )

    detections = TFLiteObjectDetector(
        interpreter=interpreter,
        labels=LabelMap(("background", "object")),
    ).detect(frame)

    assert interpreter.invoked is True
    np.testing.assert_array_equal(interpreter.input_tensor, input_tensor)
    assert detections[0].class_name == "object"
    assert detections[0].confidence == 0.95
    assert (detections[0].x_max, detections[0].y_max) == (320, 320)


class FakeInterpreter:
    def __init__(self, outputs: dict[int, Any]) -> None:
        self._outputs = outputs
        self.input_tensor: np.ndarray | None = None
        self.invoked = False

    def get_input_details(self) -> list[dict[str, int]]:
        return [{"index": 0}]

    def get_output_details(self) -> list[dict[str, int]]:
        return [{"index": 1}, {"index": 2}, {"index": 3}, {"index": 4}]

    def set_tensor(self, index: int, tensor: np.ndarray) -> None:
        assert index == 0
        self.input_tensor = tensor.copy()

    def invoke(self) -> None:
        self.invoked = True

    def get_tensor(self, index: int) -> Any:
        return self._outputs[index]
