from __future__ import annotations

import sys
from threading import Event, Thread
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from edge_vision.app.low_latency_application import LowLatencyStreamingApplication
from edge_vision.core.detection import Detection
from edge_vision.core.frame import FramePacket
from edge_vision.core.result import FrameResult


def test_low_latency_application_runs_bounded_headless_flow() -> None:
    allow_next_read = Event()
    source = PacedVideoSource([_packet(1), _packet(2)], allow_next_read)
    pipeline = FakePipeline(after_process_event=allow_next_read)
    writer = FakeResultWriter()
    reported_results: list[FrameResult] = []

    application = LowLatencyStreamingApplication(
        video_source=source,
        processing_pipeline=pipeline,
        renderer=None,
        display=None,
        max_frames=2,
        result_writer=writer,
        result_callback=reported_results.append,
    )

    processed_frames = application.run()

    assert processed_frames == 2
    assert [packet.frame_id for packet in pipeline.received_packets] == [1, 2]
    assert [result.frame_id for result in writer.results] == [1, 2]
    assert [result.frame_id for result in reported_results] == [1, 2]
    assert application.runtime_metrics["captured_frames"] == 2
    assert application.runtime_metrics["latest_captured_frame_id"] == 2
    assert application.runtime_metrics["latest_processed_frame_id"] == 2
    assert source.released is True
    assert writer.closed is True


def test_low_latency_application_drops_old_frames_when_capture_outpaces_inference() -> None:
    allow_remaining_reads = Event()
    release_inference = Event()
    source = ControlledFastVideoSource(
        [_packet(1), _packet(2), _packet(3), _packet(4), _packet(5), _packet(6)],
        allow_remaining_reads,
    )
    pipeline = BlockingPipeline(allow_remaining_reads, release_inference)
    application = LowLatencyStreamingApplication(
        video_source=source,
        processing_pipeline=pipeline,
        renderer=None,
        display=None,
        max_frames=2,
    )
    processed_frames: list[int] = []
    errors: list[Exception] = []

    def run_application() -> None:
        try:
            processed_frames.append(application.run())
        except Exception as error:
            errors.append(error)

    run_thread = Thread(target=run_application)
    run_thread.start()
    try:
        assert pipeline.started.wait(timeout=1.0)
        assert source.finished.wait(timeout=1.0)
        assert application.dropped_frames == 4
    finally:
        release_inference.set()
        run_thread.join(timeout=1.0)

    assert run_thread.is_alive() is False
    assert errors == []
    assert processed_frames == [2]
    assert [packet.frame_id for packet in pipeline.received_packets] == [1, 6]
    assert application.runtime_metrics["captured_frames"] == 6
    assert application.runtime_metrics["dropped_frames"] == 4
    assert application.runtime_metrics["dropped_frame_ratio"] == pytest.approx(4 / 6)
    assert application.runtime_metrics["latest_captured_frame_id"] == 6
    assert application.runtime_metrics["latest_processed_frame_id"] == 6
    assert application.runtime_metrics["capture_fps"] is not None
    assert application.runtime_metrics["inference_fps"] is not None
    assert source.released is True


def test_low_latency_application_renders_latest_result() -> None:
    source = FakeVideoSource([_packet(1)])
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


def test_low_latency_application_surfaces_capture_worker_errors() -> None:
    source = RaisingVideoSource(RuntimeError("capture failed"))
    writer = FakeResultWriter()

    with pytest.raises(RuntimeError, match="capture failed"):
        LowLatencyStreamingApplication(
            video_source=source,
            processing_pipeline=FakePipeline(),
            renderer=None,
            display=None,
            result_writer=writer,
        ).run()

    assert source.released is True
    assert writer.closed is True


def test_low_latency_application_surfaces_inference_worker_errors() -> None:
    source = FakeVideoSource([_packet(1)])
    writer = FakeResultWriter()

    with pytest.raises(RuntimeError, match="inference failed"):
        LowLatencyStreamingApplication(
            video_source=source,
            processing_pipeline=RaisingPipeline(RuntimeError("inference failed")),
            renderer=None,
            display=None,
            result_writer=writer,
        ).run()

    assert source.released is True
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


class PacedVideoSource(FakeVideoSource):
    def __init__(
        self,
        packets: list[FramePacket],
        allow_next_read: Event,
    ) -> None:
        super().__init__(packets)
        self._allow_next_read = allow_next_read

    def read(self) -> FramePacket | None:
        if self.read_count > 0:
            self._allow_next_read.wait(timeout=1.0)
        return super().read()


class ControlledFastVideoSource(FakeVideoSource):
    def __init__(
        self,
        packets: list[FramePacket],
        allow_remaining_reads: Event,
    ) -> None:
        super().__init__(packets)
        self._allow_remaining_reads = allow_remaining_reads
        self.finished = Event()

    def read(self) -> FramePacket | None:
        if self.read_count > 0:
            self._allow_remaining_reads.wait(timeout=1.0)
        packet = super().read()
        if packet is None:
            self.finished.set()
        return packet


class RaisingVideoSource(FakeVideoSource):
    def __init__(self, error: Exception) -> None:
        super().__init__([])
        self._error = error

    def read(self) -> FramePacket | None:
        self.read_count += 1
        raise self._error


class FakePipeline:
    def __init__(self, after_process_event: Event | None = None) -> None:
        self.received_packets: list[FramePacket] = []
        self._after_process_event = after_process_event

    def process_frame(self, frame_packet: FramePacket) -> FrameResult:
        self.received_packets.append(frame_packet)
        if self._after_process_event is not None:
            self._after_process_event.set()
        return FrameResult(
            frame_packet.frame_id,
            frame_packet.timestamp_ms,
            [_detection(frame_packet.frame_id)],
            30.0,
            4.0,
            8.0,
        )


class BlockingPipeline(FakePipeline):
    def __init__(
        self,
        allow_remaining_reads: Event,
        release_inference: Event,
    ) -> None:
        super().__init__()
        self.started = Event()
        self._allow_remaining_reads = allow_remaining_reads
        self._release_inference = release_inference

    def process_frame(self, frame_packet: FramePacket) -> FrameResult:
        self.received_packets.append(frame_packet)
        if len(self.received_packets) == 1:
            self.started.set()
            self._allow_remaining_reads.set()
            self._release_inference.wait(timeout=1.0)
        return FrameResult(
            frame_packet.frame_id,
            frame_packet.timestamp_ms,
            [_detection(frame_packet.frame_id)],
            30.0,
            4.0,
            8.0,
        )


class RaisingPipeline(FakePipeline):
    def __init__(self, error: Exception) -> None:
        super().__init__()
        self._error = error

    def process_frame(self, frame_packet: FramePacket) -> FrameResult:
        self.received_packets.append(frame_packet)
        raise self._error


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
