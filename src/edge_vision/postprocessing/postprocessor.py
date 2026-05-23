from __future__ import annotations

from dataclasses import dataclass

from edge_vision.core.detection import Detection
from edge_vision.core.frame import PreprocessedFrame
from edge_vision.postprocessing.box_smoother import BoxSmoother
from edge_vision.postprocessing.coordinate_scaler import scale_detections_coordinates
from edge_vision.postprocessing.detection_filter import (
    filter_by_confidence,
    limit_detections,
)
from edge_vision.postprocessing.nms import apply_non_max_suppression


@dataclass(frozen=True, slots=True)
class DetectionPostProcessor:
    """Filter detections and scale them to original frame coordinates."""

    confidence_threshold: float
    max_detections: int | None = None
    nms_threshold: float | None = None
    box_smoother: BoxSmoother | None = None

    def process(
        self, detections: list[Detection], frame: PreprocessedFrame
    ) -> list[Detection]:
        filtered = filter_by_confidence(detections, self.confidence_threshold)
        suppressed = apply_non_max_suppression(filtered, self.nms_threshold)
        limited = limit_detections(suppressed, self.max_detections)
        scaled = scale_detections_coordinates(limited, frame)
        if self.box_smoother is None:
            return scaled
        return self.box_smoother.smooth(scaled)
