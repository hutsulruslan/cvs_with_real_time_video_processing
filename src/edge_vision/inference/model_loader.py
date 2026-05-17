from __future__ import annotations

from pathlib import Path
from typing import Any

from edge_vision.core.errors import EdgeVisionError


class TFLiteRuntimeError(EdgeVisionError):
    """Raised when a TFLite model cannot be loaded or prepared."""


def load_tflite_interpreter(model_path: str | Path) -> Any:
    """Load and allocate a TFLite interpreter using a lazily imported runtime."""
    path = Path(model_path)
    if not path.exists():
        raise TFLiteRuntimeError(f"TFLite model file does not exist: {path}")

    interpreter_class = _load_interpreter_class()
    interpreter = interpreter_class(model_path=str(path))
    interpreter.allocate_tensors()
    return interpreter


def _load_interpreter_class() -> Any:
    try:
        from tflite_runtime.interpreter import Interpreter

        return Interpreter
    except ImportError:
        return _load_ai_edge_litert_interpreter()


def _load_ai_edge_litert_interpreter() -> Any:
    try:
        from ai_edge_litert.interpreter import Interpreter

        return Interpreter
    except ImportError:
        return _load_tensorflow_lite_interpreter()


def _load_tensorflow_lite_interpreter() -> Any:
    try:
        import tensorflow as tensorflow
    except ImportError as error:
        raise TFLiteRuntimeError(
            "TFLite runtime is not installed. Install tflite-runtime, "
            "ai-edge-litert, TensorFlow, or another LiteRT-compatible runtime "
            "to use model.runtime=tflite."
        ) from error

    return tensorflow.lite.Interpreter
