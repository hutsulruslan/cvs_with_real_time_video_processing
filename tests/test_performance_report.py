from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from edge_vision.core.detection import Detection
from edge_vision.core.result import FrameResult
from edge_vision.metrics.performance_report import (
    PerformanceReportBuilder,
    format_performance_report,
)


def test_empty_performance_report_handles_zero_frames() -> None:
    summary = PerformanceReportBuilder().build()

    assert summary.processed_frames == 0
    assert summary.total_detections == 0
    assert summary.average_fps is None
    assert summary.min_inference_ms is None


def test_performance_report_computes_summary_metrics() -> None:
    builder = PerformanceReportBuilder()

    builder.add_result(_result(frame_id=1, detections=2, fps=10.0, inference_ms=8.0, total_ms=30.0))
    builder.add_result(_result(frame_id=2, detections=1, fps=20.0, inference_ms=12.0, total_ms=50.0))

    summary = builder.build()

    assert summary.processed_frames == 2
    assert summary.total_detections == 3
    assert summary.average_fps == 15.0
    assert summary.min_fps == 10.0
    assert summary.max_fps == 20.0
    assert summary.average_inference_ms == 10.0
    assert summary.min_inference_ms == 8.0
    assert summary.max_inference_ms == 12.0
    assert summary.average_total_frame_ms == 40.0
    assert summary.min_total_frame_ms == 30.0
    assert summary.max_total_frame_ms == 50.0


def test_format_performance_report_contains_key_metrics() -> None:
    builder = PerformanceReportBuilder()
    builder.add_result(_result(frame_id=1, detections=1, fps=12.345, inference_ms=3.2, total_ms=9.8))

    report = format_performance_report(builder.build())

    assert "Performance report:" in report
    assert "processed_frames: 1" in report
    assert "total_detections: 1" in report
    assert "average_fps: 12.35" in report
    assert "average_inference_ms: 3.20" in report
    assert "average_total_frame_ms: 9.80" in report


def test_format_empty_report_uses_not_available_values() -> None:
    report = format_performance_report(PerformanceReportBuilder().build())

    assert "processed_frames: 0" in report
    assert "average_fps: n/a" in report


def _result(
    frame_id: int,
    detections: int,
    fps: float,
    inference_ms: float,
    total_ms: float,
) -> FrameResult:
    return FrameResult(
        frame_id=frame_id,
        timestamp_ms=float(frame_id),
        detections=[_detection(index) for index in range(detections)],
        fps=fps,
        inference_ms=inference_ms,
        total_frame_ms=total_ms,
    )


def _detection(index: int) -> Detection:
    return Detection(index, f"class_{index}", 0.9, 1, 2, 3, 4)
