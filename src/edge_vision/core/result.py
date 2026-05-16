from __future__ import annotations

from dataclasses import dataclass

from edge_vision.core.detection import Detection


@dataclass(frozen=True, slots=True)
class FrameResult:
    """Detection and performance data produced for one processed frame."""

    frame_id: int
    timestamp_ms: float
    detections: list[Detection]
    fps: float
    inference_ms: float
    total_frame_ms: float
