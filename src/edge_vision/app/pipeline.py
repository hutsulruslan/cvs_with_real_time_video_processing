from __future__ import annotations

from edge_vision.core.result import FrameResult
from edge_vision.inference.detector import ObjectDetector
from edge_vision.preprocessing.preprocessor import FramePreprocessor
from edge_vision.video.video_source import VideoSource


class VideoProcessingPipeline:
    """Coordinate one frame through source, preprocessing, and detection."""

    def __init__(
        self,
        video_source: VideoSource,
        preprocessor: FramePreprocessor,
        detector: ObjectDetector,
    ) -> None:
        self._video_source = video_source
        self._preprocessor = preprocessor
        self._detector = detector

    def process_next_frame(self) -> FrameResult | None:
        """Process one frame, or return None when the source is exhausted."""
        packet = self._video_source.read()
        if packet is None:
            return None

        preprocessed_frame = self._preprocessor.preprocess(packet)
        detections = self._detector.detect(preprocessed_frame)

        return FrameResult(
            frame_id=packet.frame_id,
            timestamp_ms=packet.timestamp_ms,
            detections=detections,
            fps=0.0,
            inference_ms=0.0,
            total_frame_ms=0.0,
        )
