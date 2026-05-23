from __future__ import annotations

from typing import Any

from edge_vision.core.detection import Detection
from edge_vision.core.result import FrameResult


CSV_FIELD_NAMES = [
    "frame_id",
    "timestamp_ms",
    "fps",
    "inference_ms",
    "total_frame_ms",
    "class_id",
    "class_name",
    "confidence",
    "x_min",
    "y_min",
    "x_max",
    "y_max",
    "timestamp_ns",
    "source_frame_id",
    "source_timestamp_ms",
    "source_timestamp_ns",
    "completed_timestamp_ns",
    "result_age_ms",
    "end_to_end_latency_ms",
    "inference_ran",
]


def frame_result_to_dict(result: FrameResult) -> dict[str, Any]:
    """Convert a frame result to a JSON-serializable dictionary."""
    return {
        "frame_id": result.frame_id,
        "timestamp_ms": result.timestamp_ms,
        "fps": result.fps,
        "inference_ms": result.inference_ms,
        "total_frame_ms": result.total_frame_ms,
        "timestamp_ns": result.timestamp_ns,
        "source_frame_id": result.source_frame_id,
        "source_timestamp_ms": result.source_timestamp_ms,
        "source_timestamp_ns": result.source_timestamp_ns,
        "completed_timestamp_ns": result.completed_timestamp_ns,
        "result_age_ms": result.result_age_ms,
        "end_to_end_latency_ms": result.end_to_end_latency_ms,
        "inference_ran": result.inference_ran,
        "detections": [
            detection_to_dict(detection) for detection in result.detections
        ],
    }


def detection_to_dict(detection: Detection) -> dict[str, Any]:
    """Convert a detection to a JSON-serializable dictionary."""
    return {
        "class_id": detection.class_id,
        "class_name": detection.class_name,
        "confidence": detection.confidence,
        "x_min": detection.x_min,
        "y_min": detection.y_min,
        "x_max": detection.x_max,
        "y_max": detection.y_max,
    }


def frame_result_to_csv_rows(result: FrameResult) -> list[dict[str, Any]]:
    """Convert a frame result to CSV rows, preserving frames without detections."""
    if not result.detections:
        return [_base_csv_row(result)]

    rows = []
    for detection in result.detections:
        row = _base_csv_row(result)
        row.update(detection_to_dict(detection))
        rows.append(row)
    return rows


def _base_csv_row(result: FrameResult) -> dict[str, Any]:
    return {
        "frame_id": result.frame_id,
        "timestamp_ms": result.timestamp_ms,
        "fps": result.fps,
        "inference_ms": result.inference_ms,
        "total_frame_ms": result.total_frame_ms,
        "class_id": "",
        "class_name": "",
        "confidence": "",
        "x_min": "",
        "y_min": "",
        "x_max": "",
        "y_max": "",
        "timestamp_ns": result.timestamp_ns,
        "source_frame_id": result.source_frame_id,
        "source_timestamp_ms": result.source_timestamp_ms,
        "source_timestamp_ns": result.source_timestamp_ns,
        "completed_timestamp_ns": result.completed_timestamp_ns,
        "result_age_ms": result.result_age_ms,
        "end_to_end_latency_ms": result.end_to_end_latency_ms,
        "inference_ran": result.inference_ran,
    }
