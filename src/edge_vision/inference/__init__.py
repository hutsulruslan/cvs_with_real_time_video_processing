"""Object detection interfaces and lightweight test detectors."""

from edge_vision.inference.detector import ObjectDetector
from edge_vision.inference.labels_loader import LabelMap, LabelsError, load_labels
from edge_vision.inference.mock_detector import MockObjectDetector
from edge_vision.inference.tflite_detector import TFLiteObjectDetector

__all__ = [
    "LabelMap",
    "LabelsError",
    "MockObjectDetector",
    "ObjectDetector",
    "TFLiteObjectDetector",
    "load_labels",
]
