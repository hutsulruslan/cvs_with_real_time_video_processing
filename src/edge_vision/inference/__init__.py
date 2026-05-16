"""Object detection interfaces and lightweight test detectors."""

from edge_vision.inference.detector import ObjectDetector
from edge_vision.inference.mock_detector import MockObjectDetector

__all__ = [
    "MockObjectDetector",
    "ObjectDetector",
]
