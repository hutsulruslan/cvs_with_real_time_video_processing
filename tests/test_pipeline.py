from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from edge_vision.app.pipeline import ProcessingPipeline, VideoProcessingPipeline
from edge_vision.core.detection import Detection
from edge_vision.core.frame import FramePacket
from edge_vision.inference.mock_detector import MockObjectDetector
from edge_vision.metrics.fps_counter import FPSCounter
from edge_vision.metrics.profiler import Profiler
from edge_vision.postprocessing.postprocessor import DetectionPostProcessor
from edge_vision.preprocessing.preprocessor import FramePreprocessor
from edge_vision.video.video_source import VideoSource


def test_processing_pipeline_returns_postprocessed_frame_result() -> None:
    original_frame = np.zeros((4, 4, 3), dtype=np.uint8)
    original_copy = original_frame.copy()
    pipeline = _pipeline(
        detections=[
            _detection("low", confidence=0.2, x_min=0, y_min=0, x_max=1, y_max=1),
            _detection("high", confidence=0.9, x_min=1, y_min=1, x_max=2, y_max=2),
        ],
        fps_clock=SequenceClock([0.0, 0.5]),
        profiler_clock=SequenceClock([0.0, 0.01, 0.02, 0.03, 0.08, 0.09, 0.10, 0.12]),
        completion_clock_ns=SequenceClockNs([150_000_000]),
    )

    result = pipeline.process_frame(
        FramePacket(frame_id=3, timestamp_ms=30.0, original_frame=original_frame)
    )

    assert result.frame_id == 3
    assert result.timestamp_ms == 30.0
    assert result.timestamp_ns == 30_000_000
    assert result.source_frame_id == 3
    assert result.source_timestamp_ns == 30_000_000
    assert result.completed_timestamp_ns == 150_000_000
    assert result.result_age_ms == pytest.approx(120.0)
    assert result.end_to_end_latency_ms == pytest.approx(120.0)
    assert result.inference_ran is True
    assert result.fps == 2.0
    assert result.inference_ms == pytest.approx(50.0)
    assert result.total_frame_ms == pytest.approx(120.0)
    assert [detection.class_name for detection in result.detections] == ["high"]
    assert (result.detections[0].x_min, result.detections[0].y_min) == (2, 2)
    assert (result.detections[0].x_max, result.detections[0].y_max) == (3, 3)
    np.testing.assert_array_equal(original_frame, original_copy)


def test_processing_pipeline_limits_detections_by_confidence() -> None:
    pipeline = _pipeline(
        detections=[
            _detection("middle", confidence=0.8, x_min=0),
            _detection("top", confidence=0.95, x_min=1),
            _detection("bottom", confidence=0.7, x_min=2),
        ],
        max_detections=2,
    )

    result = pipeline.process_frame(_packet())

    assert [detection.class_name for detection in result.detections] == ["top", "middle"]


def test_processing_pipeline_reuses_latest_detections_for_skipped_frames() -> None:
    pipeline = _pipeline(
        detections=[_detection("object", confidence=0.9)],
        profiler_clock=SequenceClock(
            [
                0.00,
                0.01,
                0.02,
                0.03,
                0.04,
                0.05,
                0.06,
                0.07,
                0.08,
                0.09,
                0.10,
                0.11,
                0.12,
                0.13,
                0.14,
                0.15,
                0.16,
                0.17,
            ]
        ),
        completion_clock_ns=SequenceClockNs(
            [80_000_000, 95_000_000, 170_000_000]
        ),
        frame_skip=1,
    )

    first_result = pipeline.process_frame(_packet(frame_id=1))
    second_result = pipeline.process_frame(_packet(frame_id=2))
    third_result = pipeline.process_frame(_packet(frame_id=3))

    assert first_result.inference_ms > 0.0
    assert first_result.source_frame_id == 1
    assert second_result.inference_ms == 0.0
    assert second_result.frame_id == 2
    assert second_result.source_frame_id == 1
    assert second_result.result_age_ms == pytest.approx(85.0)
    assert second_result.inference_ran is False
    assert third_result.inference_ms > 0.0
    assert third_result.source_frame_id == 3
    assert [d.class_name for d in second_result.detections] == ["object"]


def test_video_processing_pipeline_reads_controlled_sequence() -> None:
    source = FakeVideoSource([_packet(frame_id=1)])
    video_pipeline = VideoProcessingPipeline(
        video_source=source,
        pipeline=_pipeline(detections=[_detection("object", confidence=0.9)]),
    )

    first_result = video_pipeline.process_next_frame()
    second_result = video_pipeline.process_next_frame()

    assert first_result is not None
    assert first_result.frame_id == 1
    assert second_result is None


class FakeVideoSource(VideoSource):
    def __init__(self, packets: list[FramePacket]) -> None:
        self._packets = list(packets)

    def open(self) -> None:
        return None

    def read(self) -> FramePacket | None:
        if not self._packets:
            return None
        return self._packets.pop(0)

    def release(self) -> None:
        return None


class SequenceClock:
    def __init__(self, values: list[float] | None = None) -> None:
        self._values = list(values or [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06])

    def now(self) -> float:
        if not self._values:
            return 0.0
        return self._values.pop(0)


class SequenceClockNs:
    def __init__(self, values: list[int] | None = None) -> None:
        self._values = list(values or [0])

    def now(self) -> int:
        if not self._values:
            return 0
        return self._values.pop(0)


def _pipeline(
    detections: list[Detection],
    confidence_threshold: float = 0.5,
    max_detections: int | None = None,
    fps_clock: SequenceClock | None = None,
    profiler_clock: SequenceClock | None = None,
    completion_clock_ns: SequenceClockNs | None = None,
    frame_skip: int = 0,
) -> ProcessingPipeline:
    kwargs = {}
    if completion_clock_ns is not None:
        kwargs["time_provider_ns"] = completion_clock_ns.now

    return ProcessingPipeline(
        preprocessor=FramePreprocessor(input_width=2, input_height=2),
        detector=MockObjectDetector(detections),
        postprocessor=DetectionPostProcessor(
            confidence_threshold=confidence_threshold,
            max_detections=max_detections,
        ),
        fps_counter=FPSCounter(time_provider=(fps_clock or SequenceClock()).now),
        profiler=Profiler(time_provider=(profiler_clock or SequenceClock()).now),
        frame_skip=frame_skip,
        **kwargs,
    )


def _packet(frame_id: int = 1) -> FramePacket:
    return FramePacket(
        frame_id=frame_id,
        timestamp_ms=float(frame_id * 10),
        original_frame=np.zeros((4, 4, 3), dtype=np.uint8),
    )


def _detection(
    class_name: str,
    confidence: float,
    x_min: int = 1,
    y_min: int = 1,
    x_max: int = 2,
    y_max: int = 2,
) -> Detection:
    return Detection(
        class_id=1,
        class_name=class_name,
        confidence=confidence,
        x_min=x_min,
        y_min=y_min,
        x_max=x_max,
        y_max=y_max,
    )
