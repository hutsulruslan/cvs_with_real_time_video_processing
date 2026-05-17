from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from edge_vision.inference.model_loader import load_tflite_interpreter


@dataclass(frozen=True, slots=True)
class TensorInfo:
    """Small description of one TFLite tensor."""

    name: str
    index: int
    shape: list[int]
    dtype: str


@dataclass(frozen=True, slots=True)
class ModelInspectionResult:
    """Input and output tensors exposed by a TFLite model."""

    inputs: list[TensorInfo]
    outputs: list[TensorInfo]


def inspect_tflite_model(model_path: str | Path) -> ModelInspectionResult:
    """Load a TFLite model and return its input and output tensor details."""
    interpreter = load_tflite_interpreter(model_path)
    return inspect_interpreter(interpreter)


def inspect_interpreter(interpreter: Any) -> ModelInspectionResult:
    """Inspect an already constructed interpreter, useful for tests."""
    return ModelInspectionResult(
        inputs=_tensor_infos(interpreter.get_input_details()),
        outputs=_tensor_infos(interpreter.get_output_details()),
    )


def format_inspection_result(result: ModelInspectionResult) -> str:
    """Format model tensor details for manual inspection."""
    sections = [
        _format_section("Inputs", result.inputs),
        _format_section("Outputs", result.outputs),
        "Expected supported outputs: boxes, classes, scores, num_detections.",
    ]
    return "\n".join(sections)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect TFLite model tensors.")
    parser.add_argument("model_path", help="Path to a .tflite model file.")
    args = parser.parse_args(argv)

    result = inspect_tflite_model(args.model_path)
    print(format_inspection_result(result))
    return 0


def _tensor_infos(details: list[dict[str, Any]]) -> list[TensorInfo]:
    return [_tensor_info(detail) for detail in details]


def _tensor_info(detail: dict[str, Any]) -> TensorInfo:
    return TensorInfo(
        name=str(detail.get("name", "")),
        index=int(detail["index"]),
        shape=_shape_as_list(detail.get("shape", [])),
        dtype=_dtype_name(detail.get("dtype", "unknown")),
    )


def _shape_as_list(shape: Any) -> list[int]:
    return [int(value) for value in np.asarray(shape).reshape(-1)]


def _dtype_name(dtype: Any) -> str:
    try:
        return str(np.dtype(dtype).name)
    except TypeError:
        return str(dtype)


def _format_section(title: str, tensors: list[TensorInfo]) -> str:
    lines = [f"{title}:"]
    for tensor in tensors:
        lines.append(
            f"- index={tensor.index}, name={tensor.name}, "
            f"shape={tensor.shape}, dtype={tensor.dtype}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
