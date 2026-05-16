from __future__ import annotations

from collections.abc import Iterable

from edge_vision.core.detection import Detection
from edge_vision.core.frame import PreprocessedFrame


DEFAULT_DETECTIONS = (
    Detection(
        class_id=1,
        class_name="mock-object",
        confidence=0.9,
        x_min=10,
        y_min=20,
        x_max=110,
        y_max=120,
    ),
)


class MockObjectDetector:
    """Deterministic detector for tests and early pipeline integration."""

    def __init__(self, detections: Iterable[Detection] | None = None) -> None:
        configured_detections = DEFAULT_DETECTIONS if detections is None else detections
        self._detections = tuple(configured_detections)
        for detection in self._detections:
            if not isinstance(detection, Detection):
                raise TypeError("Mock detections must be Detection instances.")

    def detect(self, preprocessed_frame: PreprocessedFrame) -> list[Detection]:
        """Return configured detections without reading any external model."""
        _ = preprocessed_frame
        return list(self._detections)
