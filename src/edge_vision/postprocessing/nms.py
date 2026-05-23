from __future__ import annotations

from collections.abc import Iterable

from edge_vision.core.detection import Detection


def apply_non_max_suppression(
    detections: Iterable[Detection],
    iou_threshold: float | None,
) -> list[Detection]:
    """Suppress overlapping detections of the same class."""
    detection_list = list(detections)
    if iou_threshold is None or iou_threshold >= 1.0:
        return detection_list

    ranked = sorted(
        enumerate(detection_list),
        key=lambda item: (-item[1].confidence, item[0]),
    )
    kept: list[Detection] = []
    for _, candidate in ranked:
        if all(
            candidate.class_id != kept_detection.class_id
            or calculate_iou(candidate, kept_detection) <= iou_threshold
            for kept_detection in kept
        ):
            kept.append(candidate)
    return kept


def calculate_iou(first: Detection, second: Detection) -> float:
    """Return intersection-over-union for two axis-aligned detections."""
    first_x_min, first_x_max = sorted((first.x_min, first.x_max))
    first_y_min, first_y_max = sorted((first.y_min, first.y_max))
    second_x_min, second_x_max = sorted((second.x_min, second.x_max))
    second_y_min, second_y_max = sorted((second.y_min, second.y_max))

    intersection_width = max(
        0,
        min(first_x_max, second_x_max) - max(first_x_min, second_x_min),
    )
    intersection_height = max(
        0,
        min(first_y_max, second_y_max) - max(first_y_min, second_y_min),
    )
    intersection_area = intersection_width * intersection_height
    if intersection_area == 0:
        return 0.0

    first_area = max(0, first_x_max - first_x_min) * max(0, first_y_max - first_y_min)
    second_area = max(0, second_x_max - second_x_min) * max(0, second_y_max - second_y_min)
    union_area = first_area + second_area - intersection_area
    if union_area <= 0:
        return 0.0
    return intersection_area / union_area
