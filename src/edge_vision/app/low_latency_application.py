from __future__ import annotations

from threading import Event, Lock, Thread
from typing import Any, Callable

from edge_vision.app.pipeline import ProcessingPipeline
from edge_vision.app.realtime_state import (
    LatestFrameBuffer,
    LatestResultStore,
    LowLatencyRuntimeStats,
)
from edge_vision.core.frame import FramePacket
from edge_vision.core.result import FrameResult
from edge_vision.storage.result_writer import ResultWriter
from edge_vision.video.video_source import VideoSource
from edge_vision.visualization.renderer import Renderer
from edge_vision.visualization.window_display import WindowDisplay


class LowLatencyStreamingApplication:
    """Run opt-in latest-frame processing without accumulating old frames."""

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
        runtime_stats: LowLatencyRuntimeStats | None = None,
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
        self._runtime_stats = runtime_stats or LowLatencyRuntimeStats()
        self._capture_batch_size = capture_batch_size
        self._stop_event = Event()
        self._capture_finished = Event()
        self._inference_finished = Event()
        self._worker_error_lock = Lock()
        self._render_lock = Lock()
        self._worker_error: Exception | None = None
        self._latest_rendered_frame: Any | None = None
        self._latest_rendered_frame_id: int | None = None
        self._last_displayed_frame_id: int | None = None
        self._control_poll_interval_s = 0.01
        self._frame_wait_timeout_s = 0.01

    def run(self) -> int:
        """Run separated capture/inference workers and return processed frames."""
        workers: list[Thread] = []
        try:
            self._prepare_run_state()
            self._video_source.open()
            if self._can_process_more():
                workers = self._start_workers()
                self._wait_for_workers()
                self._raise_worker_error()
        finally:
            self._stop_event.set()
            for worker in workers:
                worker.join()
            self._video_source.release()
            if self._display is not None:
                self._display.close()
            if self._result_writer is not None:
                self._result_writer.close()

        self._raise_worker_error()
        return self._processed_frame_count()

    @property
    def dropped_frames(self) -> int:
        """Frames replaced in the latest-frame buffer before processing."""
        return self._frame_buffer.dropped_frames

    @property
    def replaced_results(self) -> int:
        """Results replaced in the latest-result store."""
        return self._result_store.replaced_results

    @property
    def runtime_metrics(self) -> dict[str, int | float | None]:
        """Low-latency counters for CLI reporting."""
        return self._runtime_stats.report_fields(
            dropped_frames=self.dropped_frames,
            replaced_results=self.replaced_results,
        )

    def _prepare_run_state(self) -> None:
        self._stop_event.clear()
        self._capture_finished.clear()
        self._inference_finished.clear()
        self._frame_buffer.reset()
        self._result_store.reset()
        self._runtime_stats.reset()
        with self._worker_error_lock:
            self._worker_error = None
        with self._render_lock:
            self._latest_rendered_frame = None
            self._latest_rendered_frame_id = None
            self._last_displayed_frame_id = None

    def _start_workers(self) -> list[Thread]:
        capture_worker = Thread(
            target=self._capture_frames,
            name="edge-vision-low-latency-capture",
        )
        inference_worker = Thread(
            target=self._process_latest_frames,
            name="edge-vision-low-latency-inference",
        )
        capture_worker.start()
        inference_worker.start()
        return [capture_worker, inference_worker]

    def _capture_frames(self) -> None:
        try:
            self._runtime_stats.start_capture()
            while not self._stop_event.is_set():
                for _ in range(self._capture_batch_size):
                    if self._stop_event.is_set():
                        break
                    frame_packet = self._video_source.read()
                    if frame_packet is None:
                        return
                    self._runtime_stats.record_captured_frame(frame_packet)
                    self._frame_buffer.put(frame_packet)
        except Exception as error:
            self._record_worker_error(error)
        finally:
            self._runtime_stats.finish_capture()
            self._capture_finished.set()

    def _process_latest_frames(self) -> None:
        try:
            self._runtime_stats.start_inference()
            while not self._stop_event.is_set():
                if not self._can_process_more():
                    self._stop_event.set()
                    break

                frame_packet = self._frame_buffer.wait_pop_latest(
                    timeout=self._frame_wait_timeout_s
                )
                if frame_packet is None:
                    if self._capture_finished.is_set():
                        break
                    continue

                result = self._processing_pipeline.process_frame(frame_packet)
                self._result_store.put(result)
                if self._result_writer is not None:
                    self._result_writer.write(result)
                if self._result_callback is not None:
                    self._result_callback(result)
                self._publish_rendered_frame(frame_packet, result)
                self._record_processed_result(result)
        except Exception as error:
            self._record_worker_error(error)
        finally:
            self._runtime_stats.finish_inference()
            self._inference_finished.set()

    def _wait_for_workers(self) -> None:
        while not self._inference_finished.is_set():
            self._raise_worker_error()
            self._show_latest_rendered_frame()
            self._inference_finished.wait(self._control_poll_interval_s)
        self._show_latest_rendered_frame()

    def _can_process_more(self) -> bool:
        if self._max_frames is None:
            return True
        return self._processed_frame_count() < self._max_frames

    def _processed_frame_count(self) -> int:
        return self._runtime_stats.snapshot().processed_frames

    def _record_processed_result(self, result: FrameResult) -> None:
        processed_frames = self._runtime_stats.record_processed_result(result)
        should_stop = (
            self._max_frames is not None and processed_frames >= self._max_frames
        )
        if should_stop:
            self._stop_event.set()

    def _publish_rendered_frame(
        self,
        frame_packet: FramePacket,
        result: FrameResult,
    ) -> None:
        if self._renderer is None:
            return

        rendered_frame = self._renderer.render(
            frame_packet.original_frame,
            result.detections,
            fps=result.fps,
            inference_ran=result.inference_ran,
            result_age_ms=result.result_age_ms,
        )
        with self._render_lock:
            self._latest_rendered_frame = rendered_frame
            self._latest_rendered_frame_id = result.frame_id

    def _show_latest_rendered_frame(self) -> bool:
        if self._display is None:
            return False

        with self._render_lock:
            frame = self._latest_rendered_frame
            frame_id = self._latest_rendered_frame_id
            if frame is None or frame_id == self._last_displayed_frame_id:
                return False
            self._last_displayed_frame_id = frame_id

        should_stop = self._display.show(frame)
        if should_stop:
            self._stop_event.set()
        return should_stop

    def _record_worker_error(self, error: Exception) -> None:
        with self._worker_error_lock:
            if self._worker_error is None:
                self._worker_error = error
        self._stop_event.set()

    def _raise_worker_error(self) -> None:
        with self._worker_error_lock:
            error = self._worker_error
        if error is not None:
            raise error
