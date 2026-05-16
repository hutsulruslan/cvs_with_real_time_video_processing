from __future__ import annotations

from dataclasses import dataclass

from edge_vision.core.detection import Detection
from edge_vision.core.frame import PreprocessedFrame
from edge_vision.postprocessing.coordinate_scaler import scale_detections_coordinates
from edge_vision.postprocessing.detection_filter import (
    filter_by_confidence,
    limit_detections,
)


@dataclass(frozen=True, slots=True)
class DetectionPostProcessor:
    """Filter detections and scale them to original frame coordinates."""

    confidence_threshold: float
    max_detections: int | None = None

    def process(
        self, detections: list[Detection], frame: PreprocessedFrame
    ) -> list[Detection]:
        filtered = filter_by_confidence(detections, self.confidence_threshold)
        limited = limit_detections(filtered, self.max_detections)
        return scale_detections_coordinates(limited, frame)
