from __future__ import annotations

from dataclasses import dataclass

from edge_vision.core.detection import Detection


@dataclass(frozen=True, slots=True)
class FrameResult:
    """Detection and performance data produced for one processed frame."""

    frame_id: int
    timestamp_ms: float
    detections: list[Detection]
    fps: float
    inference_ms: float
    total_frame_ms: float
    timestamp_ns: int | None = None
    source_frame_id: int | None = None
    source_timestamp_ms: float | None = None
    source_timestamp_ns: int | None = None
    completed_timestamp_ns: int | None = None
    result_age_ms: float | None = None
    end_to_end_latency_ms: float | None = None
    inference_ran: bool = True

    def __post_init__(self) -> None:
        timestamp_ns = self.timestamp_ns
        if timestamp_ns is None:
            timestamp_ns = round(self.timestamp_ms * 1_000_000)
            object.__setattr__(self, "timestamp_ns", timestamp_ns)

        source_frame_id = self.source_frame_id
        if source_frame_id is None:
            source_frame_id = self.frame_id
            object.__setattr__(self, "source_frame_id", source_frame_id)

        source_timestamp_ns = self.source_timestamp_ns
        source_timestamp_ms = self.source_timestamp_ms
        if source_timestamp_ms is None:
            if source_timestamp_ns is None:
                source_timestamp_ms = self.timestamp_ms
            else:
                source_timestamp_ms = source_timestamp_ns / 1_000_000.0
            object.__setattr__(self, "source_timestamp_ms", source_timestamp_ms)

        if source_timestamp_ns is None:
            source_timestamp_ns = round(source_timestamp_ms * 1_000_000)
            object.__setattr__(self, "source_timestamp_ns", source_timestamp_ns)

        completed_timestamp_ns = self.completed_timestamp_ns
        if completed_timestamp_ns is None:
            completed_timestamp_ns = source_timestamp_ns + round(
                self.total_frame_ms * 1_000_000
            )
            object.__setattr__(
                self,
                "completed_timestamp_ns",
                completed_timestamp_ns,
            )

        if self.result_age_ms is None:
            object.__setattr__(
                self,
                "result_age_ms",
                (completed_timestamp_ns - source_timestamp_ns) / 1_000_000,
            )

        if self.end_to_end_latency_ms is None:
            object.__setattr__(
                self,
                "end_to_end_latency_ms",
                (completed_timestamp_ns - timestamp_ns) / 1_000_000,
            )
