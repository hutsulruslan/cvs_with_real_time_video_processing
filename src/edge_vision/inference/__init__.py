"""Object detection interfaces and lightweight test detectors."""

from edge_vision.inference.detector import ObjectDetector
from edge_vision.inference.labels_loader import LabelMap, LabelsError, load_labels
from edge_vision.inference.mock_detector import MockObjectDetector
from edge_vision.inference.model_inspector import (
    ModelInspectionResult,
    TensorInfo,
    inspect_interpreter,
    inspect_tflite_model,
)
from edge_vision.inference.tflite_detector import TFLiteObjectDetector

__all__ = [
    "LabelMap",
    "LabelsError",
    "ModelInspectionResult",
    "MockObjectDetector",
    "ObjectDetector",
    "TensorInfo",
    "TFLiteObjectDetector",
    "inspect_interpreter",
    "inspect_tflite_model",
    "load_labels",
]
