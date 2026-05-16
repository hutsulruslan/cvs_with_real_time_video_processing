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
from edge_vision.visualization.renderer import Renderer


def test_renderer_returns_array_with_same_shape() -> None:
    frame = _black_frame()

    rendered = Renderer(show_fps=False).render(frame, [_detection()])

    assert isinstance(rendered, np.ndarray)
    assert rendered.shape == frame.shape


def test_renderer_does_not_mutate_original_frame() -> None:
    frame = _black_frame()
    original = frame.copy()

    Renderer(show_fps=False).render(frame, [_detection()])

    np.testing.assert_array_equal(frame, original)


def test_rendering_with_detections_changes_output_frame() -> None:
    frame = _black_frame()

    rendered = Renderer(show_fps=False).render(frame, [_detection()])

    assert not np.array_equal(rendered, frame)


def test_rendering_empty_detections_returns_valid_frame() -> None:
    frame = _black_frame()

    rendered = Renderer(show_fps=False).render(frame, [])

    assert rendered.shape == frame.shape
    np.testing.assert_array_equal(rendered, frame)


def test_rendering_with_fps_enabled_changes_output_frame() -> None:
    frame = _black_frame()

    rendered = Renderer(show_fps=True).render(frame, [], fps=24.5)

    assert not np.array_equal(rendered, frame)


def test_detection_label_near_top_left_edge_does_not_crash() -> None:
    frame = _black_frame()
    detection = _detection(x_min=0, y_min=0, x_max=20, y_max=20)

    rendered = Renderer(show_fps=False).render(frame, [detection])

    assert rendered.shape == frame.shape
    assert rendered.sum() > 0


def test_renderer_rejects_invalid_frame_shape() -> None:
    with pytest.raises(ValueError, match="three color channels"):
        Renderer().render(np.zeros((20, 20), dtype=np.uint8), [])


def _black_frame() -> np.ndarray:
    return np.zeros((80, 120, 3), dtype=np.uint8)


def _detection(
    x_min: int = 10,
    y_min: int = 15,
    x_max: int = 60,
    y_max: int = 50,
) -> Detection:
    return Detection(
        class_id=1,
        class_name="person",
        confidence=0.85,
        x_min=x_min,
        y_min=y_min,
        x_max=x_max,
        y_max=y_max,
    )
