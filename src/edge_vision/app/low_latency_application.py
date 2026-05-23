from __future__ import annotations

from typing import Callable

from edge_vision.app.pipeline import ProcessingPipeline
from edge_vision.app.realtime_state import LatestFrameBuffer, LatestResultStore
from edge_vision.core.frame import FramePacket
from edge_vision.core.result import FrameResult
from edge_vision.storage.result_writer import ResultWriter
from edge_vision.video.video_source import VideoSource
from edge_vision.visualization.renderer import Renderer
from edge_vision.visualization.window_display import WindowDisplay


class LowLatencyStreamingApplication:
    """Run an opt-in latest-frame processing loop without accumulating old frames."""

    def __init__(
        self,
        video_source: VideoSource,
        processing_pipeline: ProcessingPipeline,
        renderer: Renderer | None,
        display: WindowDisplay | None,
        max_frames: int | None = None,
        result_writer: ResultWriter | None = None,
        result_callback: Callable[[FrameResult], None] | None = None,
        frame_buffer: LatestFrameBuffer | None = None,
        result_store: LatestResultStore | None = None,
        capture_batch_size: int = 1,
    ) -> None:
        if max_frames is not None and max_frames < 0:
            raise ValueError("max_frames must be non-negative or None.")
        if capture_batch_size <= 0:
            raise ValueError("capture_batch_size must be positive.")
        if (renderer is None) != (display is None):
            raise ValueError(
                "renderer and display must either both be set or both be None."
            )
        self._video_source = video_source
        self._processing_pipeline = processing_pipeline
        self._renderer = renderer
        self._display = display
        self._max_frames = max_frames
        self._result_writer = result_writer
        self._result_callback = result_callback
        self._frame_buffer = frame_buffer or LatestFrameBuffer()
        self._result_store = result_store or LatestResultStore()
        self._capture_batch_size = capture_batch_size

    def run(self) -> int:
        """Run a bounded latest-frame loop and return processed frames."""
        processed_frames = 0
        try:
            self._video_source.open()
            while self._can_process_more(processed_frames):
                if not self._capture_latest_frames():
                    break

                frame_packet = self._frame_buffer.pop_latest()
                if frame_packet is None:
                    continue

                result = self._processing_pipeline.process_frame(frame_packet)
                self._result_store.put(result)
                if self._result_writer is not None:
                    self._result_writer.write(result)
                if self._result_callback is not None:
                    self._result_callback(result)
                processed_frames += 1
                if self._should_stop_for_display(frame_packet):
                    break
        finally:
            self._video_source.release()
            if self._display is not None:
                self._display.close()
            if self._result_writer is not None:
                self._result_writer.close()

        return processed_frames

    @property
    def dropped_frames(self) -> int:
        """Frames replaced in the latest-frame buffer before processing."""
        return self._frame_buffer.dropped_frames

    @property
    def replaced_results(self) -> int:
        """Results replaced in the latest-result store."""
        return self._result_store.replaced_results

    def _capture_latest_frames(self) -> bool:
        captured_any = False
        for _ in range(self._capture_batch_size):
            frame_packet = self._video_source.read()
            if frame_packet is None:
                break
            self._frame_buffer.put(frame_packet)
            captured_any = True
        return captured_any

    def _can_process_more(self, processed_frames: int) -> bool:
        if self._max_frames is None:
            return True
        return processed_frames < self._max_frames

    def _should_stop_for_display(self, frame_packet: FramePacket) -> bool:
        if self._renderer is None or self._display is None:
            return False

        latest_result = self._result_store.get_latest()
        if latest_result is None:
            return False

        rendered_frame = self._renderer.render(
            frame_packet.original_frame,
            latest_result.detections,
            fps=latest_result.fps,
        )
        return self._display.show(rendered_frame)
