from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from edge_vision.app.application import EdgeVisionApplication
from edge_vision.app.pipeline import ProcessingPipeline
from edge_vision.core.detection import Detection
from edge_vision.core.frame import FramePacket
from edge_vision.core.result import FrameResult
from edge_vision.inference.mock_detector import MockObjectDetector
from edge_vision.metrics.fps_counter import FPSCounter
from edge_vision.metrics.profiler import Profiler
from edge_vision.postprocessing.postprocessor import DetectionPostProcessor
from edge_vision.preprocessing.preprocessor import FramePreprocessor
from edge_vision.visualization.renderer import Renderer


def test_application_runs_visual_flow_with_mock_detector() -> None:
    source = FakeVideoSource([_packet(1), _packet(2), _packet(3)])
    display = FakeDisplay()

    processed_count = _application(
        source,
        pipeline=_mock_processing_pipeline(),
        renderer=Renderer(show_fps=True),
        display=display,
        max_frames=2,
    ).run()

    assert processed_count == 2
    assert len(display.shown_frames) == 2
    assert all(frame.sum() > 0 for frame in display.shown_frames)


def test_application_writes_results_when_writer_is_present() -> None:
    writer = FakeResultWriter()
    pipeline = FakePipeline()

    processed_count = _application(
        FakeVideoSource([_packet(1), _packet(2)]),
        pipeline=pipeline,
        display=FakeDisplay(),
        result_writer=writer,
    ).run()

    assert processed_count == 2
    assert [result.frame_id for result in writer.results] == [1, 2]
    assert writer.closed is True


def test_application_stops_when_source_ends() -> None:
    source = FakeVideoSource([_packet(1)])
    display = FakeDisplay()

    processed_count = _application(source, display=display).run()

    assert processed_count == 1
    assert source.read_count == 2
    assert source.opened is True
    assert source.released is True
    assert display.closed is True


def test_application_stops_when_display_requests_quit() -> None:
    pipeline = FakePipeline()
    writer = FakeResultWriter()

    processed_count = _application(
        FakeVideoSource([_packet(1), _packet(2), _packet(3)]),
        pipeline=pipeline,
        display=FakeDisplay(quit_after=1),
        result_writer=writer,
    ).run()

    assert processed_count == 1
    assert [packet.frame_id for packet in pipeline.received_packets] == [1]
    assert [result.frame_id for result in writer.results] == [1]
    assert writer.closed is True


def test_application_with_zero_max_frames_only_opens_and_closes() -> None:
    source = FakeVideoSource([_packet(1)])
    pipeline = FakePipeline()
    display = FakeDisplay()
    writer = FakeResultWriter()

    processed_count = _application(
        source,
        pipeline=pipeline,
        display=display,
        max_frames=0,
        result_writer=writer,
    ).run()

    assert processed_count == 0
    assert pipeline.received_packets == []
    assert source.released is True
    assert display.closed is True
    assert writer.results == []
    assert writer.closed is True


def test_application_can_run_without_display() -> None:
    source = FakeVideoSource([_packet(1), _packet(2)])
    pipeline = FakePipeline()
    writer = FakeResultWriter()

    processed_count = EdgeVisionApplication(
        video_source=source,
        processing_pipeline=pipeline,
        renderer=None,
        display=None,
        max_frames=2,
        result_writer=writer,
    ).run()

    assert processed_count == 2
    assert [packet.frame_id for packet in pipeline.received_packets] == [1, 2]
    assert [result.frame_id for result in writer.results] == [1, 2]
    assert source.released is True
    assert writer.closed is True


def test_application_reports_results_through_callback() -> None:
    reported_results: list[FrameResult] = []

    processed_count = _application(
        FakeVideoSource([_packet(1), _packet(2)]),
        pipeline=FakePipeline(),
        display=FakeDisplay(),
        result_callback=reported_results.append,
    ).run()

    assert processed_count == 2
    assert [result.frame_id for result in reported_results] == [1, 2]


class FakeVideoSource:
    def __init__(self, packets: list[FramePacket]) -> None:
        self._packets = list(packets)
        self.opened = False
        self.released = False
        self.read_count = 0

    def open(self) -> None:
        self.opened = True

    def read(self) -> FramePacket | None:
        self.read_count += 1
        return self._packets.pop(0) if self._packets else None

    def release(self) -> None:
        self.released = True


class FakePipeline:
    def __init__(self) -> None:
        self.received_packets: list[FramePacket] = []

    def process_frame(self, frame_packet: FramePacket) -> FrameResult:
        self.received_packets.append(frame_packet)
        return FrameResult(frame_packet.frame_id, frame_packet.timestamp_ms, [_detection()], 12.5, 5.0, 20.0)


class FakeRenderer:
    def __init__(self) -> None:
        self.render_calls: list[tuple[np.ndarray, list[Detection], float | None, bool]] = []

    def render(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        fps: float | None = None,
        inference_ran: bool = True,
        result_age_ms: float | None = None,
    ) -> np.ndarray:
        self.render_calls.append((frame, detections, fps, inference_ran))
        return frame.copy()


class FakeDisplay:
    def __init__(self, quit_after: int | None = None) -> None:
        self._quit_after = quit_after
        self.shown_frames: list[np.ndarray] = []
        self.closed = False

    def show(self, frame: np.ndarray) -> bool:
        self.shown_frames.append(frame)
        return self._quit_after is not None and len(self.shown_frames) >= self._quit_after

    def close(self) -> None:
        self.closed = True


class FakeResultWriter:
    def __init__(self) -> None:
        self.results: list[FrameResult] = []
        self.closed = False

    def write(self, result: FrameResult) -> None:
        self.results.append(result)

    def close(self) -> None:
        self.closed = True


def _packet(frame_id: int) -> FramePacket:
    return FramePacket(frame_id, float(frame_id * 10), np.zeros((4, 4, 3), dtype=np.uint8))


def _application(
    source,
    pipeline=None,
    renderer=None,
    display=None,
    max_frames=None,
    result_writer=None,
    result_callback=None,
) -> EdgeVisionApplication:
    return EdgeVisionApplication(
        source,
        pipeline or FakePipeline(),
        renderer or FakeRenderer(),
        display or FakeDisplay(),
        max_frames,
        result_writer,
        result_callback,
    )

def _mock_processing_pipeline() -> ProcessingPipeline:
    return ProcessingPipeline(
        preprocessor=FramePreprocessor(input_width=2, input_height=2),
        detector=MockObjectDetector([_detection()]),
        postprocessor=DetectionPostProcessor(confidence_threshold=0.5),
        fps_counter=FPSCounter(),
        profiler=Profiler(),
    )


def _detection() -> Detection:
    return Detection(1, "mock-object", 0.9, 0, 0, 3, 3)
