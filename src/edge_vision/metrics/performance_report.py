from __future__ import annotations

from dataclasses import dataclass

from edge_vision.core.result import FrameResult


@dataclass(frozen=True, slots=True)
class PerformanceSummary:
    """Aggregated frame-level metrics for one application run."""

    processed_frames: int
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


class PerformanceReportBuilder:
    """Collect FrameResult metrics and build a compact run summary."""

    def __init__(self) -> None:
        self._processed_frames = 0
        self._total_detections = 0
        self._fps = _MetricAccumulator()
        self._inference_ms = _MetricAccumulator()
        self._total_frame_ms = _MetricAccumulator()

    def add_result(self, result: FrameResult) -> None:
        """Add one processed frame result to the report."""
        self._processed_frames += 1
        self._total_detections += len(result.detections)
        self._fps.add(result.fps)
        self._inference_ms.add(result.inference_ms)
        self._total_frame_ms.add(result.total_frame_ms)

    def build(self) -> PerformanceSummary:
        """Return the current immutable summary."""
        return PerformanceSummary(
            processed_frames=self._processed_frames,
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
        )


def format_performance_report(summary: PerformanceSummary) -> str:
    """Format a performance summary for CLI output."""
    return "\n".join(
        [
            "Performance report:",
            f"- processed_frames: {summary.processed_frames}",
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
        ]
    )


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
