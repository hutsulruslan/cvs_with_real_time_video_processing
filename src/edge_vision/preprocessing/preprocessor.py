from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from edge_vision.config.settings import LowLightSettings, ModelSettings
from edge_vision.core.errors import PreprocessingError
from edge_vision.core.frame import FramePacket, PreprocessedFrame
from edge_vision.preprocessing.image_transform import (
    add_batch_dimension,
    convert_bgr_to_rgb,
    normalize_to_unit_float,
    resize_frame,
)
from edge_vision.preprocessing.low_light import enhance_low_light


@dataclass(frozen=True, slots=True)
class FramePreprocessor:
    """Prepare original video frames for model inference."""

    input_width: int
    input_height: int
    normalize_input: bool = False
    low_light_settings: LowLightSettings = field(default_factory=LowLightSettings)

    def __post_init__(self) -> None:
        _validate_input_size(self.input_width, self.input_height)

    @classmethod
    def from_model_settings(
        cls,
        settings: ModelSettings,
        low_light_settings: LowLightSettings | None = None,
    ) -> FramePreprocessor:
        return cls(
            input_width=settings.input_width,
            input_height=settings.input_height,
            normalize_input=settings.normalize,
            low_light_settings=low_light_settings or LowLightSettings(),
        )

    def preprocess(self, packet: FramePacket) -> PreprocessedFrame:
        frame = packet.original_frame
        original_height, original_width = frame.shape[:2]

        enhanced_bgr = enhance_low_light(frame, self.low_light_settings)
        resized_bgr = resize_frame(enhanced_bgr, self.input_width, self.input_height)
        resized_rgb = convert_bgr_to_rgb(resized_bgr)
        tensor_frame = self._prepare_tensor_frame(resized_rgb)

        return PreprocessedFrame(
            input_tensor=add_batch_dimension(tensor_frame),
            original_width=original_width,
            original_height=original_height,
            input_width=self.input_width,
            input_height=self.input_height,
            scale_x=original_width / self.input_width,
            scale_y=original_height / self.input_height,
        )

    def _prepare_tensor_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.normalize_input:
            return normalize_to_unit_float(frame)
        return np.ascontiguousarray(frame)


def _validate_input_size(width: int, height: int) -> None:
    if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
        raise PreprocessingError("Input width must be a positive integer.")
    if not isinstance(height, int) or isinstance(height, bool) or height <= 0:
        raise PreprocessingError("Input height must be a positive integer.")
