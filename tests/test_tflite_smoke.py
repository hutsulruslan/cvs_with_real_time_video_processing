from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from edge_vision.config.settings import ModelSettings
from edge_vision.core.detection import Detection
from edge_vision.core.frame import FramePacket
from edge_vision.inference.tflite_detector import TFLiteObjectDetector
from edge_vision.preprocessing.preprocessor import FramePreprocessor


def test_local_tflite_model_runs_non_gui_smoke_check() -> None:
    pytest.importorskip("ai_edge_litert.interpreter")
    model_path = PROJECT_ROOT / "assets" / "models" / "model.tflite"
    labels_path = PROJECT_ROOT / "assets" / "models" / "labels.txt"
    if not model_path.exists() or not labels_path.exists():
        pytest.skip("Local TFLite model files are not available.")

    settings = ModelSettings(
        runtime="tflite",
        model_path=str(model_path),
        labels_path=str(labels_path),
        input_width=320,
        input_height=320,
        confidence_threshold=0.4,
        nms_threshold=0.5,
        normalize=False,
    )
    packet = FramePacket(1, 10.0, np.zeros((480, 640, 3), dtype=np.uint8))
    preprocessed = FramePreprocessor.from_model_settings(settings).preprocess(packet)

    assert preprocessed.input_tensor.shape == (1, 320, 320, 3)
    assert preprocessed.input_tensor.dtype == np.uint8

    detections = TFLiteObjectDetector(
        model_path=settings.model_path,
        labels_path=settings.labels_path,
    ).detect(preprocessed)

    assert isinstance(detections, list)
    assert all(isinstance(detection, Detection) for detection in detections)
