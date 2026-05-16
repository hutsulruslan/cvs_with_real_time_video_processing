from __future__ import annotations

from collections.abc import Iterable

from edge_vision.core.detection import Detection


def filter_by_confidence(
    detections: Iterable[Detection], confidence_threshold: float
) -> list[Detection]:
    """Keep detections whose confidence is greater than or equal to threshold."""
    return [
        detection
        for detection in detections
        if detection.confidence >= confidence_threshold
    ]


def limit_detections(
    detections: Iterable[Detection], max_detections: int | None
) -> list[Detection]:
    """Keep the top detections by confidence when a positive limit is provided."""
    detection_list = list(detections)
    if max_detections is None or max_detections <= 0:
        return detection_list

    ranked = sorted(
        enumerate(detection_list),
        key=lambda item: (-item[1].confidence, item[0]),
    )
    return [detection for _, detection in ranked[:max_detections]]
