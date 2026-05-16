from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from edge_vision.app.pipeline import VideoProcessingPipeline
from edge_vision.core.detection import Detection
from edge_vision.core.frame import FramePacket, PreprocessedFrame
from edge_vision.inference.mock_detector import MockObjectDetector
from edge_vision.preprocessing.preprocessor import FramePreprocessor
from edge_vision.video.video_source import VideoSource


def test_pipeline_processes_one_frame_into_frame_result() -> None:
    detection = Detection(
        class_id=1,
        class_name="test-object",
        confidence=0.8,
        x_min=1,
        y_min=2,
        x_max=30,
        y_max=40,
    )
    pipeline = VideoProcessingPipeline(
        video_source=FakeVideoSource([_packet(frame_id=3)]),
        preprocessor=FramePreprocessor(input_width=2, input_height=2),
        detector=MockObjectDetector([detection]),
    )

    result = pipeline.process_next_frame()

    assert result is not None
    assert result.frame_id == 3
    assert result.timestamp_ms == 30.0
    assert result.detections == [detection]
    assert result.fps == 0.0
    assert result.inference_ms == 0.0
    assert result.total_frame_ms == 0.0


def test_pipeline_returns_none_when_source_has_no_frame() -> None:
    pipeline = VideoProcessingPipeline(
        video_source=FakeVideoSource([]),
        preprocessor=FramePreprocessor(input_width=2, input_height=2),
        detector=MockObjectDetector(),
    )

    assert pipeline.process_next_frame() is None


def test_pipeline_passes_preprocessed_frame_to_detector() -> None:
    detector = RecordingDetector()
    pipeline = VideoProcessingPipeline(
        video_source=FakeVideoSource([_packet(frame_id=1)]),
        preprocessor=FramePreprocessor(input_width=2, input_height=2),
        detector=detector,
    )

    pipeline.process_next_frame()

    assert detector.received_frame is not None
    assert detector.received_frame.input_tensor.shape == (1, 2, 2, 3)
    assert detector.received_frame.scale_x == 2.0
    assert detector.received_frame.scale_y == 2.0


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


class RecordingDetector:
    def __init__(self) -> None:
        self.received_frame: PreprocessedFrame | None = None

    def detect(self, preprocessed_frame: PreprocessedFrame) -> list[Detection]:
        self.received_frame = preprocessed_frame
        return []


def _packet(frame_id: int) -> FramePacket:
    return FramePacket(
        frame_id=frame_id,
        timestamp_ms=float(frame_id * 10),
        original_frame=np.zeros((4, 4, 3), dtype=np.uint8),
    )
