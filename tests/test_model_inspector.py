from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from edge_vision.inference.model_inspector import (
    TensorInfo,
    format_inspection_result,
    inspect_interpreter,
)


def test_inspect_interpreter_returns_input_and_output_tensor_info() -> None:
    result = inspect_interpreter(FakeInterpreter())

    assert result.inputs == [
        TensorInfo("serving_default_image:0", 0, [1, 320, 320, 3], "uint8")
    ]
    assert result.outputs[0] == TensorInfo("boxes", 1, [1, 10, 4], "float32")
    assert result.outputs[1] == TensorInfo("classes", 2, [1, 10], "float32")


def test_format_inspection_result_mentions_supported_outputs() -> None:
    result = inspect_interpreter(FakeInterpreter())

    formatted = format_inspection_result(result)

    assert "Inputs:" in formatted
    assert "Outputs:" in formatted
    assert "boxes, classes, scores, num_detections" in formatted


class FakeInterpreter:
    def get_input_details(self) -> list[dict]:
        return [
            {
                "name": "serving_default_image:0",
                "index": 0,
                "shape": np.array([1, 320, 320, 3]),
                "dtype": np.uint8,
            }
        ]

    def get_output_details(self) -> list[dict]:
        return [
            {"name": "boxes", "index": 1, "shape": [1, 10, 4], "dtype": np.float32},
            {"name": "classes", "index": 2, "shape": [1, 10], "dtype": np.float32},
            {"name": "scores", "index": 3, "shape": [1, 10], "dtype": np.float32},
            {"name": "num_detections", "index": 4, "shape": [1], "dtype": np.float32},
        ]
