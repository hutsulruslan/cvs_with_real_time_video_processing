from __future__ import annotations

from edge_vision.app.pipeline import ProcessingPipeline
from edge_vision.storage.result_writer import ResultWriter
from edge_vision.video.video_source import VideoSource
from edge_vision.visualization.renderer import Renderer
from edge_vision.visualization.window_display import WindowDisplay


class EdgeVisionApplication:
    """Coordinate source, processing pipeline, renderer, and display."""

    def __init__(
        self,
        video_source: VideoSource,
        processing_pipeline: ProcessingPipeline,
        renderer: Renderer,
        display: WindowDisplay,
        max_frames: int | None = None,
        result_writer: ResultWriter | None = None,
    ) -> None:
        if max_frames is not None and max_frames < 0:
            raise ValueError("max_frames must be non-negative or None.")
        self._video_source = video_source
        self._processing_pipeline = processing_pipeline
        self._renderer = renderer
        self._display = display
        self._max_frames = max_frames
        self._result_writer = result_writer

    def run(self) -> int:
        """Run a controlled visual processing loop and return processed frames."""
        processed_frames = 0
        try:
            self._video_source.open()
            while self._can_process_more(processed_frames):
                frame_packet = self._video_source.read()
                if frame_packet is None:
                    break

                result = self._processing_pipeline.process_frame(frame_packet)
                if self._result_writer is not None:
                    self._result_writer.write(result)
                rendered_frame = self._renderer.render(
                    frame_packet.original_frame,
                    result.detections,
                    fps=result.fps,
                )
                processed_frames += 1
                if self._display.show(rendered_frame):
                    break
        finally:
            self._video_source.release()
            self._display.close()
            if self._result_writer is not None:
                self._result_writer.close()

        return processed_frames

    def _can_process_more(self, processed_frames: int) -> bool:
        if self._max_frames is None:
            return True
        return processed_frames < self._max_frames
