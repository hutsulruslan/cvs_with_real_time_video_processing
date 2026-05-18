from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from edge_vision.config.settings import LowLightSettings
from edge_vision.preprocessing.low_light import (
    apply_clahe,
    apply_gamma_correction,
    enhance_low_light,
)


def test_gamma_correction_changes_dark_frame_without_mutating_original() -> None:
    frame = np.full((8, 8, 3), 30, dtype=np.uint8)
    original = frame.copy()

    enhanced = apply_gamma_correction(frame, gamma=1.5)

    assert enhanced.dtype == np.uint8
    assert enhanced.shape == frame.shape
    assert enhanced.mean() > frame.mean()
    np.testing.assert_array_equal(frame, original)


def test_clahe_returns_same_shape_and_dtype() -> None:
    frame = _gradient_frame()

    enhanced = apply_clahe(frame, clip_limit=2.0, tile_grid_size=4)

    assert enhanced.dtype == np.uint8
    assert enhanced.shape == frame.shape


def test_gamma_clahe_returns_same_shape_and_dtype() -> None:
    frame = _gradient_frame()
    settings = LowLightSettings(enabled=True, mode="gamma_clahe", gamma=1.5)

    enhanced = enhance_low_light(frame, settings)

    assert enhanced.dtype == np.uint8
    assert enhanced.shape == frame.shape


def test_auto_mode_leaves_bright_frame_unchanged() -> None:
    frame = np.full((8, 8, 3), 230, dtype=np.uint8)
    settings = LowLightSettings(
        enabled=True,
        mode="auto",
        brightness_threshold=65,
    )

    enhanced = enhance_low_light(frame, settings)

    np.testing.assert_array_equal(enhanced, frame)


def test_auto_mode_enhances_dark_frame() -> None:
    frame = np.full((8, 8, 3), 25, dtype=np.uint8)
    settings = LowLightSettings(
        enabled=True,
        mode="auto",
        gamma=1.5,
        brightness_threshold=65,
        clahe_tile_grid_size=4,
    )

    enhanced = enhance_low_light(frame, settings)

    assert enhanced.mean() > frame.mean()


def test_enhancement_modes_do_not_mutate_original_frame() -> None:
    for settings in [
        LowLightSettings(enabled=True, mode="gamma", gamma=1.5),
        LowLightSettings(enabled=True, mode="clahe", clahe_tile_grid_size=4),
        LowLightSettings(enabled=True, mode="gamma_clahe", gamma=1.5, clahe_tile_grid_size=4),
        LowLightSettings(enabled=True, mode="auto", gamma=1.5, brightness_threshold=65, clahe_tile_grid_size=4),
    ]:
        frame = np.full((8, 8, 3), 25, dtype=np.uint8)
        original = frame.copy()

        enhance_low_light(frame, settings)

        np.testing.assert_array_equal(frame, original)


def test_disabled_low_light_returns_unchanged_frame() -> None:
    frame = _gradient_frame()

    enhanced = enhance_low_light(frame, LowLightSettings(enabled=False, mode="gamma"))

    np.testing.assert_array_equal(enhanced, frame)


def _gradient_frame() -> np.ndarray:
    row = np.linspace(0, 255, 8, dtype=np.uint8)
    gray = np.tile(row, (8, 1))
    return np.stack((gray, gray, gray), axis=2)
