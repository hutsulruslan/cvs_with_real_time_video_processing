from __future__ import annotations

from typing import Protocol, runtime_checkable

from edge_vision.core.detection import Detection
from edge_vision.core.frame import PreprocessedFrame


@runtime_checkable
class ObjectDetector(Protocol):
    """Interface for object detectors used by the processing pipeline."""

    def detect(self, preprocessed_frame: PreprocessedFrame) -> list[Detection]:
        """Run object detection on a preprocessed frame."""
        ...
