from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from edge_vision.config.settings import LowLightSettings, ModelSettings
from edge_vision.core.errors import PreprocessingError
from edge_vision.core.frame import FramePacket
from edge_vision.preprocessing.image_transform import (
    add_batch_dimension,
    convert_bgr_to_rgb,
    normalize_to_unit_float,
    resize_frame,
)
from edge_vision.preprocessing.preprocessor import FramePreprocessor


def test_preprocess_returns_batched_rgb_tensor_and_scale_values() -> None:
    frame = np.array(
        [
            [[10, 20, 30], [40, 50, 60], [70, 80, 90]],
            [[11, 21, 31], [41, 51, 61], [71, 81, 91]],
        ],
        dtype=np.uint8,
    )
    packet = FramePacket(frame_id=7, timestamp_ms=123.0, original_frame=frame)
    preprocessor = FramePreprocessor(input_width=3, input_height=2)

    result = preprocessor.preprocess(packet)

    assert result.input_tensor.shape == (1, 2, 3, 3)
    assert result.input_tensor.dtype == np.uint8
    assert result.original_width == 3
    assert result.original_height == 2
    assert result.input_width == 3
    assert result.input_height == 2
    assert result.scale_x == 1.0
    assert result.scale_y == 1.0
    np.testing.assert_array_equal(result.input_tensor[0, 0, 0], [30, 20, 10])


def test_preprocess_does_not_mutate_original_frame() -> None:
    frame = np.full((4, 5, 3), 128, dtype=np.uint8)
    original = frame.copy()
    packet = FramePacket(frame_id=1, timestamp_ms=10.0, original_frame=frame)

    FramePreprocessor(input_width=3, input_height=2).preprocess(packet)

    np.testing.assert_array_equal(frame, original)


def test_preprocess_tracks_scale_values_for_resized_frame() -> None:
    frame = np.zeros((4, 8, 3), dtype=np.uint8)
    packet = FramePacket(frame_id=1, timestamp_ms=10.0, original_frame=frame)

    result = FramePreprocessor(input_width=2, input_height=2).preprocess(packet)

    assert result.input_tensor.shape == (1, 2, 2, 3)
    assert result.scale_x == 4.0
    assert result.scale_y == 2.0


def test_preprocess_can_normalize_to_float_tensor() -> None:
    frame = np.full((2, 2, 3), 255, dtype=np.uint8)
    packet = FramePacket(frame_id=1, timestamp_ms=10.0, original_frame=frame)
    preprocessor = FramePreprocessor(input_width=2, input_height=2, normalize_input=True)

    result = preprocessor.preprocess(packet)

    assert result.input_tensor.dtype == np.float32
    np.testing.assert_allclose(result.input_tensor, 1.0)


def test_preprocessor_can_be_created_from_model_settings() -> None:
    settings = ModelSettings(
        runtime="tflite",
        model_path="assets/models/model.tflite",
        labels_path="assets/models/labels.txt",
        input_width=320,
        input_height=240,
        confidence_threshold=0.4,
        nms_threshold=0.5,
        normalize=True,
    )

    preprocessor = FramePreprocessor.from_model_settings(settings)

    assert preprocessor.input_width == 320
    assert preprocessor.input_height == 240
    assert preprocessor.normalize_input is True


def test_preprocessor_accepts_low_light_settings_from_factory() -> None:
    settings = ModelSettings(
        runtime="tflite",
        model_path="assets/models/model.tflite",
        labels_path="assets/models/labels.txt",
        input_width=2,
        input_height=2,
        confidence_threshold=0.4,
        nms_threshold=0.5,
        normalize=False,
    )
    low_light = LowLightSettings(enabled=True, mode="gamma", gamma=1.5)

    preprocessor = FramePreprocessor.from_model_settings(settings, low_light)

    assert preprocessor.low_light_settings == low_light


def test_preprocess_with_low_light_keeps_uint8_tensor_when_not_normalized() -> None:
    frame = np.full((4, 4, 3), 25, dtype=np.uint8)
    packet = FramePacket(frame_id=1, timestamp_ms=10.0, original_frame=frame)
    preprocessor = FramePreprocessor(
        input_width=2,
        input_height=2,
        normalize_input=False,
        low_light_settings=LowLightSettings(enabled=True, mode="gamma", gamma=1.5),
    )

    result = preprocessor.preprocess(packet)

    assert result.input_tensor.shape == (1, 2, 2, 3)
    assert result.input_tensor.dtype == np.uint8
    assert result.input_tensor.mean() > frame.mean()


def test_image_transform_helpers_prepare_expected_shapes_and_values() -> None:
    frame = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)

    resized = resize_frame(frame, target_width=1, target_height=1)
    rgb = convert_bgr_to_rgb(frame)
    normalized = normalize_to_unit_float(frame)
    batched = add_batch_dimension(frame)

    assert resized.shape == (1, 1, 3)
    np.testing.assert_array_equal(rgb[0, 0], [3, 2, 1])
    assert normalized.dtype == np.float32
    np.testing.assert_allclose(normalized[0, 0], [1 / 255, 2 / 255, 3 / 255])
    assert batched.shape == (1, 1, 2, 3)


def test_preprocessor_rejects_invalid_frame_shape() -> None:
    invalid_frame = np.zeros((10, 10), dtype=np.uint8)
    packet = FramePacket(frame_id=1, timestamp_ms=10.0, original_frame=invalid_frame)

    with pytest.raises(PreprocessingError, match="channel"):
        FramePreprocessor(input_width=3, input_height=3).preprocess(packet)


def test_preprocessor_rejects_invalid_input_size() -> None:
    with pytest.raises(PreprocessingError, match="width"):
        FramePreprocessor(input_width=0, input_height=320)
