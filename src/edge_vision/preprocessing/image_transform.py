from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from edge_vision.core.errors import PreprocessingError


def resize_frame(
    frame: NDArray[Any], target_width: int, target_height: int
) -> NDArray[Any]:
    """Resize a frame to the model input size."""
    _validate_frame(frame)
    _validate_size(target_width, target_height)
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)


def convert_bgr_to_rgb(frame: NDArray[Any]) -> NDArray[Any]:
    """Convert an OpenCV BGR frame to RGB channel order."""
    _validate_frame(frame)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def normalize_to_unit_float(frame: NDArray[Any]) -> NDArray[np.float32]:
    """Convert image values to float32 in the [0.0, 1.0] range."""
    _validate_frame(frame)
    return frame.astype(np.float32) / 255.0


def add_batch_dimension(frame: NDArray[Any]) -> NDArray[Any]:
    """Add a leading batch dimension expected by most TFLite models."""
    _validate_frame(frame)
    return np.expand_dims(frame, axis=0)


def _validate_frame(frame: NDArray[Any]) -> None:
    if not isinstance(frame, np.ndarray):
        raise PreprocessingError("Frame must be a NumPy array.")
    if frame.ndim != 3:
        raise PreprocessingError("Frame must have height, width, and channel dimensions.")
    if frame.shape[2] != 3:
        raise PreprocessingError("Frame must have exactly three color channels.")
    if frame.shape[0] <= 0 or frame.shape[1] <= 0:
        raise PreprocessingError("Frame dimensions must be positive.")


def _validate_size(width: int, height: int) -> None:
    if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
        raise PreprocessingError("Target width must be a positive integer.")
    if not isinstance(height, int) or isinstance(height, bool) or height <= 0:
        raise PreprocessingError("Target height must be a positive integer.")
