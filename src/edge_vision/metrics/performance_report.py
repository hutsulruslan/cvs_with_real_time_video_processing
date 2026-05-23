from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from edge_vision.core.result import FrameResult


@dataclass(frozen=True, slots=True)
class PerformanceSummary:
    """Aggregated frame-level metrics for one application run."""

    processed_frames: int
    processed_inference_frames: int
    total_detections: int
    average_fps: float | None
    min_fps: float | None
    max_fps: float | None
    average_inference_ms: float | None
    min_inference_ms: float | None
    max_inference_ms: float | None
    average_total_frame_ms: float | None
    min_total_frame_ms: float | None
    max_total_frame_ms: float | None
    average_result_age_ms: float | None
    min_result_age_ms: float | None
    max_result_age_ms: float | None
    average_end_to_end_latency_ms: float | None
    min_end_to_end_latency_ms: float | None
    max_end_to_end_latency_ms: float | None


class PerformanceReportBuilder:
    """Collect FrameResult metrics and build a compact run summary."""

    def __init__(self) -> None:
        self._processed_frames = 0
        self._processed_inference_frames = 0
        self._total_detections = 0
        self._fps = _MetricAccumulator()
        self._inference_ms = _MetricAccumulator()
        self._total_frame_ms = _MetricAccumulator()
        self._result_age_ms = _MetricAccumulator()
        self._end_to_end_latency_ms = _MetricAccumulator()

    def add_result(self, result: FrameResult) -> None:
        """Add one processed frame result to the report."""
        self._processed_frames += 1
        if result.inference_ran:
            self._processed_inference_frames += 1
            self._inference_ms.add(result.inference_ms)
        self._total_detections += len(result.detections)
        self._fps.add(result.fps)
        self._total_frame_ms.add(result.total_frame_ms)
        self._result_age_ms.add_optional(result.result_age_ms)
        self._end_to_end_latency_ms.add_optional(result.end_to_end_latency_ms)

    def build(self) -> PerformanceSummary:
        """Return the current immutable summary."""
        return PerformanceSummary(
            processed_frames=self._processed_frames,
            processed_inference_frames=self._processed_inference_frames,
            total_detections=self._total_detections,
            average_fps=self._fps.average,
            min_fps=self._fps.minimum,
            max_fps=self._fps.maximum,
            average_inference_ms=self._inference_ms.average,
            min_inference_ms=self._inference_ms.minimum,
            max_inference_ms=self._inference_ms.maximum,
            average_total_frame_ms=self._total_frame_ms.average,
            min_total_frame_ms=self._total_frame_ms.minimum,
            max_total_frame_ms=self._total_frame_ms.maximum,
            average_result_age_ms=self._result_age_ms.average,
            min_result_age_ms=self._result_age_ms.minimum,
            max_result_age_ms=self._result_age_ms.maximum,
            average_end_to_end_latency_ms=self._end_to_end_latency_ms.average,
            min_end_to_end_latency_ms=self._end_to_end_latency_ms.minimum,
            max_end_to_end_latency_ms=self._end_to_end_latency_ms.maximum,
        )


def format_performance_report(
    summary: PerformanceSummary,
    *,
    runtime_metrics: Mapping[str, int | float | None] | None = None,
    dropped_frames: int | None = None,
    replaced_results: int | None = None,
) -> str:
    """Format a performance summary for CLI output."""
    lines = [
        "Performance report:",
        f"- processed_frames: {summary.processed_frames}",
        f"- processed_inference_frames: {summary.processed_inference_frames}",
        f"- total_detections: {summary.total_detections}",
        f"- average_fps: {_format_metric(summary.average_fps)}",
        f"- min_fps: {_format_metric(summary.min_fps)}",
        f"- max_fps: {_format_metric(summary.max_fps)}",
        f"- average_inference_ms: {_format_metric(summary.average_inference_ms)}",
        f"- min_inference_ms: {_format_metric(summary.min_inference_ms)}",
        f"- max_inference_ms: {_format_metric(summary.max_inference_ms)}",
        f"- average_total_frame_ms: {_format_metric(summary.average_total_frame_ms)}",
        f"- min_total_frame_ms: {_format_metric(summary.min_total_frame_ms)}",
        f"- max_total_frame_ms: {_format_metric(summary.max_total_frame_ms)}",
        f"- average_result_age_ms: {_format_metric(summary.average_result_age_ms)}",
        f"- min_result_age_ms: {_format_metric(summary.min_result_age_ms)}",
        f"- max_result_age_ms: {_format_metric(summary.max_result_age_ms)}",
        (
            "- average_end_to_end_latency_ms: "
            f"{_format_metric(summary.average_end_to_end_latency_ms)}"
        ),
        (
            "- min_end_to_end_latency_ms: "
            f"{_format_metric(summary.min_end_to_end_latency_ms)}"
        ),
        (
            "- max_end_to_end_latency_ms: "
            f"{_format_metric(summary.max_end_to_end_latency_ms)}"
        ),
    ]
    extra_metrics = dict(runtime_metrics or {})
    if dropped_frames is not None and "dropped_frames" not in extra_metrics:
        extra_metrics["dropped_frames"] = dropped_frames
    if replaced_results is not None and "replaced_results" not in extra_metrics:
        extra_metrics["replaced_results"] = replaced_results
    for name, value in extra_metrics.items():
        lines.append(f"- {name}: {_format_report_value(value)}")
    return "\n".join(lines)


class _MetricAccumulator:
    def __init__(self) -> None:
        self._count = 0
        self._total = 0.0
        self._minimum: float | None = None
        self._maximum: float | None = None

    def add(self, value: float) -> None:
        self._count += 1
        self._total += value
        self._minimum = value if self._minimum is None else min(self._minimum, value)
        self._maximum = value if self._maximum is None else max(self._maximum, value)

    def add_optional(self, value: float | None) -> None:
        if value is not None:
            self.add(value)

    @property
    def average(self) -> float | None:
        if self._count == 0:
            return None
        return self._total / self._count

    @property
    def minimum(self) -> float | None:
        return self._minimum

    @property
    def maximum(self) -> float | None:
        return self._maximum


def _format_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def _format_report_value(value: int | float | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return _format_metric(value)
