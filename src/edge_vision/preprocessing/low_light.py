from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from edge_vision.config.settings import LowLightSettings
from edge_vision.core.errors import PreprocessingError


def enhance_low_light(
    frame: NDArray[Any],
    settings: LowLightSettings,
) -> NDArray[np.uint8]:
    """Apply optional lightweight low-light enhancement to a BGR frame."""
    _validate_bgr_uint8_frame(frame)
    if not settings.enabled or settings.mode == "off":
        return frame
    if settings.mode == "gamma":
        return apply_gamma_correction(frame, settings.gamma)
    if settings.mode == "clahe":
        return apply_clahe(frame, settings.clahe_clip_limit, settings.clahe_tile_grid_size)
    if settings.mode == "gamma_clahe":
        corrected = apply_gamma_correction(frame, settings.gamma)
        return apply_clahe(corrected, settings.clahe_clip_limit, settings.clahe_tile_grid_size)
    if settings.mode == "auto":
        if estimate_brightness(frame) >= settings.brightness_threshold:
            return frame
        corrected = apply_gamma_correction(frame, settings.gamma)
        return apply_clahe(corrected, settings.clahe_clip_limit, settings.clahe_tile_grid_size)
    raise PreprocessingError(f"Unsupported low-light mode: {settings.mode}")


def apply_gamma_correction(frame: NDArray[Any], gamma: float) -> NDArray[np.uint8]:
    """Brighten or darken a BGR frame with lookup-table gamma correction."""
    _validate_bgr_uint8_frame(frame)
    if not isinstance(gamma, (int, float)) or isinstance(gamma, bool) or gamma <= 0:
        raise PreprocessingError("Low-light gamma must be a positive number.")

    inverse_gamma = 1.0 / float(gamma)
    lookup_table = np.array(
        [((value / 255.0) ** inverse_gamma) * 255.0 for value in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(frame, lookup_table)


def apply_clahe(
    frame: NDArray[Any],
    clip_limit: float,
    tile_grid_size: int,
) -> NDArray[np.uint8]:
    """Apply CLAHE on the luminance channel of a BGR frame."""
    _validate_bgr_uint8_frame(frame)
    if not isinstance(clip_limit, (int, float)) or isinstance(clip_limit, bool) or clip_limit <= 0:
        raise PreprocessingError("CLAHE clip limit must be a positive number.")
    if not isinstance(tile_grid_size, int) or isinstance(tile_grid_size, bool) or tile_grid_size <= 0:
        raise PreprocessingError("CLAHE tile grid size must be a positive integer.")

    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y_channel, cr_channel, cb_channel = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=(tile_grid_size, tile_grid_size),
    )
    enhanced_y = clahe.apply(y_channel)
    enhanced_ycrcb = cv2.merge((enhanced_y, cr_channel, cb_channel))
    return cv2.cvtColor(enhanced_ycrcb, cv2.COLOR_YCrCb2BGR)


def estimate_brightness(frame: NDArray[Any]) -> float:
    """Estimate average luminance for auto low-light decisions."""
    _validate_bgr_uint8_frame(frame)
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    return float(ycrcb[:, :, 0].mean())


def _validate_bgr_uint8_frame(frame: NDArray[Any]) -> None:
    if not isinstance(frame, np.ndarray):
        raise PreprocessingError("Frame must be a NumPy array.")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise PreprocessingError("Frame must be a BGR image with three channels.")
    if frame.dtype != np.uint8:
        raise PreprocessingError("Low-light enhancement expects a uint8 BGR frame.")
