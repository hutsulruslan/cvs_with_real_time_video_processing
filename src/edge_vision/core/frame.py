from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class FramePacket:
    """Original frame and metadata read from a video source."""

    frame_id: int
    timestamp_ms: float
    original_frame: NDArray[Any]


@dataclass(frozen=True, slots=True)
class PreprocessedFrame:
    """Model input tensor and information required to restore coordinates."""

    input_tensor: NDArray[Any]
    original_width: int
    original_height: int
    input_width: int
    input_height: int
    scale_x: float
    scale_y: float
