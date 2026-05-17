from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from edge_vision.core.detection import Detection
from edge_vision.core.frame import PreprocessedFrame
from edge_vision.inference.labels_loader import LabelMap, load_labels
from edge_vision.inference.model_loader import load_tflite_interpreter


class TFLiteObjectDetector:
    """Object detector for SSD/EfficientDet-style TFLite detection models."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        labels_path: str | Path | None = None,
        *,
        interpreter: Any | None = None,
        labels: LabelMap | None = None,
    ) -> None:
        self._interpreter = interpreter or _load_required_interpreter(model_path)
        self._labels = labels or _load_optional_labels(labels_path)
        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()

    def detect(self, preprocessed_frame: PreprocessedFrame) -> list[Detection]:
        """Run TFLite inference and return detections in model input coordinates."""
        self._interpreter.set_tensor(
            self._input_details[0]["index"],
            preprocessed_frame.input_tensor,
        )
        self._interpreter.invoke()

        outputs = [
            self._interpreter.get_tensor(output["index"])
            for output in self._output_details
        ]
        if len(outputs) < 4:
            raise ValueError("SSD-style TFLite detector requires four output tensors.")

        return parse_ssd_outputs(
            boxes=outputs[0],
            classes=outputs[1],
            scores=outputs[2],
            num_detections=outputs[3],
            labels=self._labels,
            input_width=preprocessed_frame.input_width,
            input_height=preprocessed_frame.input_height,
        )


def parse_ssd_outputs(
    *,
    boxes: Any,
    classes: Any,
    scores: Any,
    num_detections: Any,
    labels: LabelMap,
    input_width: int,
    input_height: int,
) -> list[Detection]:
    """Convert normalized SSD TFLite outputs to Detection objects."""
    box_array = _first_batch(boxes)
    class_array = _first_batch(classes)
    score_array = _first_batch(scores)
    detection_count = _detection_count(num_detections, box_array, class_array, score_array)

    detections: list[Detection] = []
    for index in range(detection_count):
        y_min, x_min, y_max, x_max = box_array[index]
        class_id = int(class_array[index])
        detections.append(
            Detection(
                class_id=class_id,
                class_name=labels.get_label(class_id),
                confidence=float(score_array[index]),
                x_min=round(float(x_min) * input_width),
                y_min=round(float(y_min) * input_height),
                x_max=round(float(x_max) * input_width),
                y_max=round(float(y_max) * input_height),
            )
        )
    return detections


def _load_required_interpreter(model_path: str | Path | None) -> Any:
    if model_path is None:
        raise ValueError("model_path is required when interpreter is not provided.")
    return load_tflite_interpreter(model_path)


def _load_optional_labels(labels_path: str | Path | None) -> LabelMap:
    if labels_path is None:
        return LabelMap(())
    return load_labels(labels_path)


def _first_batch(values: Any) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim > 1 and array.shape[0] == 1:
        return array[0]
    return array


def _detection_count(
    num_detections: Any,
    boxes: np.ndarray,
    classes: np.ndarray,
    scores: np.ndarray,
) -> int:
    requested_count = int(np.asarray(num_detections).reshape(-1)[0])
    return min(requested_count, len(boxes), len(classes), len(scores))
