from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from edge_vision.app.low_latency_application import LowLatencyStreamingApplication
from edge_vision.core.detection import Detection
from edge_vision.core.frame import FramePacket
from edge_vision.core.result import FrameResult


def test_low_latency_application_runs_bounded_headless_flow() -> None:
    source = FakeVideoSource([_packet(1), _packet(2)])
    pipeline = FakePipeline()
    writer = FakeResultWriter()
    reported_results: list[FrameResult] = []

    processed_frames = LowLatencyStreamingApplication(
        video_source=source,
        processing_pipeline=pipeline,
        renderer=None,
        display=None,
        max_frames=2,
        result_writer=writer,
        result_callback=reported_results.append,
    ).run()

    assert processed_frames == 2
    assert [packet.frame_id for packet in pipeline.received_packets] == [1, 2]
    assert [result.frame_id for result in writer.results] == [1, 2]
    assert [result.frame_id for result in reported_results] == [1, 2]
    assert source.released is True
    assert writer.closed is True


def test_low_latency_application_uses_latest_frame_without_queueing_old_frames() -> None:
    source = FakeVideoSource([_packet(1), _packet(2), _packet(3)])
    pipeline = FakePipeline()
    application = LowLatencyStreamingApplication(
        video_source=source,
        processing_pipeline=pipeline,
        renderer=None,
        display=None,
        max_frames=1,
        capture_batch_size=2,
    )

    processed_frames = application.run()

    assert processed_frames == 1
    assert [packet.frame_id for packet in pipeline.received_packets] == [2]
    assert application.dropped_frames == 1
    assert source.read_count == 2
    assert source.released is True


def test_low_latency_application_renders_latest_result() -> None:
    source = FakeVideoSource([_packet(1), _packet(2)])
    pipeline = FakePipeline()
    renderer = FakeRenderer()
    display = FakeDisplay(quit_after=1)

    processed_frames = LowLatencyStreamingApplication(
        video_source=source,
        processing_pipeline=pipeline,
        renderer=renderer,
        display=display,
        max_frames=2,
    ).run()

    assert processed_frames == 1
    assert [call[0] for call in renderer.render_calls] == [1]
    assert [call[1][0].class_name for call in renderer.render_calls] == ["object-1"]
    assert display.closed is True


def test_low_latency_application_closes_resources_when_source_is_empty() -> None:
    source = FakeVideoSource([])
    writer = FakeResultWriter()
    display = FakeDisplay()

    processed_frames = LowLatencyStreamingApplication(
        video_source=source,
        processing_pipeline=FakePipeline(),
        renderer=FakeRenderer(),
        display=display,
        result_writer=writer,
    ).run()

    assert processed_frames == 0
    assert source.opened is True
    assert source.released is True
    assert display.closed is True
    assert writer.closed is True


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
        return FrameResult(
            frame_packet.frame_id,
            frame_packet.timestamp_ms,
            [_detection(frame_packet.frame_id)],
            30.0,
            4.0,
            8.0,
        )


class FakeRenderer:
    def __init__(self) -> None:
        self.render_calls: list[tuple[int, list[Detection], float | None]] = []

    def render(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        fps: float | None = None,
    ) -> np.ndarray:
        self.render_calls.append((int(frame[0, 0, 0]), detections, fps))
        return frame.copy()


class FakeDisplay:
    def __init__(self, quit_after: int | None = None) -> None:
        self._quit_after = quit_after
        self.shown_frames: list[np.ndarray] = []
        self.closed = False

    def show(self, frame: np.ndarray) -> bool:
        self.shown_frames.append(frame)
        return (
            self._quit_after is not None
            and len(self.shown_frames) >= self._quit_after
        )

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
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    frame[:, :, 0] = frame_id
    return FramePacket(
        frame_id=frame_id,
        timestamp_ms=float(frame_id * 10),
        original_frame=frame,
    )


def _detection(frame_id: int) -> Detection:
    return Detection(1, f"object-{frame_id}", 0.9, 0, 0, 2, 2)
