from __future__ import annotations

from time import perf_counter_ns
from typing import Callable

from edge_vision.core.detection import Detection
from edge_vision.core.frame import FramePacket
from edge_vision.core.result import FrameResult
from edge_vision.inference.detector import ObjectDetector
from edge_vision.metrics.fps_counter import FPSCounter
from edge_vision.metrics.profiler import Profiler
from edge_vision.postprocessing.postprocessor import DetectionPostProcessor
from edge_vision.preprocessing.preprocessor import FramePreprocessor
from edge_vision.video.video_source import VideoSource


class ProcessingPipeline:
    """Coordinate one frame through preprocessing, detection, and postprocessing."""

    def __init__(
        self,
        preprocessor: FramePreprocessor,
        detector: ObjectDetector,
        postprocessor: DetectionPostProcessor,
        fps_counter: FPSCounter,
        profiler: Profiler,
        frame_skip: int = 0,
        time_provider_ns: Callable[[], int] = perf_counter_ns,
    ) -> None:
        if frame_skip < 0:
            raise ValueError("frame_skip must be non-negative.")
        self._preprocessor = preprocessor
        self._detector = detector
        self._postprocessor = postprocessor
        self._fps_counter = fps_counter
        self._profiler = profiler
        self._frame_skip = frame_skip
        self._skip_countdown = 0
        self._last_detections: list[Detection] = []
        self._last_source_frame_id: int | None = None
        self._last_source_timestamp_ms: float | None = None
        self._last_source_timestamp_ns: int | None = None
        self._time_provider_ns = time_provider_ns

    def process_frame(self, frame_packet: FramePacket) -> FrameResult:
        """Process one frame packet and return detections with timing data."""
        self._profiler.reset()

        self._profiler.start("total_frame")
        if self._should_reuse_last_detections():
            self._skip_countdown -= 1
            total_frame_ms = self._profiler.stop("total_frame")
            fps = self._fps_counter.update()
            completed_timestamp_ns = self._time_provider_ns()
            return FrameResult(
                frame_id=frame_packet.frame_id,
                timestamp_ms=frame_packet.timestamp_ms,
                detections=list(self._last_detections),
                fps=fps,
                inference_ms=0.0,
                total_frame_ms=total_frame_ms,
                timestamp_ns=frame_packet.timestamp_ns,
                source_frame_id=self._last_source_frame_id,
                source_timestamp_ms=self._last_source_timestamp_ms,
                source_timestamp_ns=self._last_source_timestamp_ns,
                completed_timestamp_ns=completed_timestamp_ns,
                inference_ran=False,
            )

        self._profiler.start("preprocess")
        preprocessed_frame = self._preprocessor.preprocess(frame_packet)
        self._profiler.stop("preprocess")

        self._profiler.start("inference")
        raw_detections = self._detector.detect(preprocessed_frame)
        inference_ms = self._profiler.stop("inference")

        self._profiler.start("postprocess")
        detections = self._postprocessor.process(raw_detections, preprocessed_frame)
        self._profiler.stop("postprocess")

        total_frame_ms = self._profiler.stop("total_frame")
        fps = self._fps_counter.update()
        completed_timestamp_ns = self._time_provider_ns()
        self._last_detections = list(detections)
        self._last_source_frame_id = frame_packet.frame_id
        self._last_source_timestamp_ms = frame_packet.timestamp_ms
        self._last_source_timestamp_ns = frame_packet.timestamp_ns
        self._skip_countdown = self._frame_skip

        return FrameResult(
            frame_id=frame_packet.frame_id,
            timestamp_ms=frame_packet.timestamp_ms,
            detections=detections,
            fps=fps,
            inference_ms=inference_ms,
            total_frame_ms=total_frame_ms,
            timestamp_ns=frame_packet.timestamp_ns,
            source_frame_id=frame_packet.frame_id,
            source_timestamp_ms=frame_packet.timestamp_ms,
            source_timestamp_ns=frame_packet.timestamp_ns,
            completed_timestamp_ns=completed_timestamp_ns,
            inference_ran=True,
        )

    def _should_reuse_last_detections(self) -> bool:
        return self._skip_countdown > 0


class VideoProcessingPipeline:
    """Read frames from a video source and pass them to a processing pipeline."""

    def __init__(
        self,
        video_source: VideoSource,
        pipeline: ProcessingPipeline,
    ) -> None:
        self._video_source = video_source
        self._pipeline = pipeline

    def process_next_frame(self) -> FrameResult | None:
        """Process one frame, or return None when the source is exhausted."""
        packet = self._video_source.read()
        if packet is None:
            return None

        return self._pipeline.process_frame(packet)
