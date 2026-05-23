from __future__ import annotations

from edge_vision.core.detection import Detection
from edge_vision.postprocessing.nms import calculate_iou


class BoxSmoother:
    """Apply lightweight EMA smoothing to matched detections."""

    def __init__(self, alpha: float = 0.6, iou_threshold: float = 0.3) -> None:
        if alpha < 0.0 or alpha > 1.0:
            raise ValueError("alpha must be between 0.0 and 1.0.")
        if iou_threshold < 0.0 or iou_threshold > 1.0:
            raise ValueError("iou_threshold must be between 0.0 and 1.0.")
        self._alpha = alpha
        self._iou_threshold = iou_threshold
        self._previous_detections: list[Detection] = []

    def smooth(self, detections: list[Detection]) -> list[Detection]:
        """Return smoothed detections without mutating the input list."""
        if not detections:
            self._previous_detections = []
            return []

        matched_previous: set[int] = set()
        smoothed: list[Detection] = []
        for detection in detections:
            previous_index = self._best_match_index(detection, matched_previous)
            if previous_index is None:
                smoothed_detection = detection
            else:
                matched_previous.add(previous_index)
                smoothed_detection = self._smooth_detection(
                    detection,
                    self._previous_detections[previous_index],
                )
            smoothed.append(smoothed_detection)

        self._previous_detections = smoothed
        return smoothed

    def reset(self) -> None:
        """Clear smoothing state."""
        self._previous_detections = []

    def _best_match_index(
        self,
        detection: Detection,
        matched_previous: set[int],
    ) -> int | None:
        best_index: int | None = None
        best_iou = 0.0
        for index, previous in enumerate(self._previous_detections):
            if index in matched_previous or previous.class_id != detection.class_id:
                continue
            iou = calculate_iou(detection, previous)
            if iou >= self._iou_threshold and iou > best_iou:
                best_iou = iou
                best_index = index
        return best_index

    def _smooth_detection(
        self,
        detection: Detection,
        previous: Detection,
    ) -> Detection:
        return Detection(
            class_id=detection.class_id,
            class_name=detection.class_name,
            confidence=detection.confidence,
            x_min=self._smooth_coordinate(detection.x_min, previous.x_min),
            y_min=self._smooth_coordinate(detection.y_min, previous.y_min),
            x_max=self._smooth_coordinate(detection.x_max, previous.x_max),
            y_max=self._smooth_coordinate(detection.y_max, previous.y_max),
        )

    def _smooth_coordinate(self, current: int, previous: int) -> int:
        return round((self._alpha * current) + ((1.0 - self._alpha) * previous))
