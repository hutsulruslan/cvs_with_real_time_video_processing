from __future__ import annotations

from typing import Callable

from edge_vision.app.application import EdgeVisionApplication
from edge_vision.app.pipeline import ProcessingPipeline
from edge_vision.config.settings import AppSettings
from edge_vision.core.errors import ApplicationError
from edge_vision.core.result import FrameResult
from edge_vision.inference.detector import ObjectDetector
from edge_vision.inference.mock_detector import MockObjectDetector
from edge_vision.inference.tflite_detector import TFLiteObjectDetector
from edge_vision.metrics.fps_counter import FPSCounter
from edge_vision.metrics.profiler import Profiler
from edge_vision.postprocessing.postprocessor import DetectionPostProcessor
from edge_vision.preprocessing.preprocessor import FramePreprocessor
from edge_vision.storage.writer_factory import create_result_writer
from edge_vision.video.source_factory import create_video_source
from edge_vision.video.video_source import VideoSource
from edge_vision.visualization.renderer import Renderer
from edge_vision.visualization.window_display import WindowDisplay


def create_application(
    settings: AppSettings,
    *,
    video_source: VideoSource | None = None,
    display: WindowDisplay | None = None,
    max_frames: int | None = None,
    no_display: bool = False,
    result_callback: Callable[[FrameResult], None] | None = None,
) -> EdgeVisionApplication:
    """Create the current visual MVP application from typed settings."""
    if not no_display and not settings.display.show_window:
        raise ApplicationError("Visual mode requires display.show_window to be true.")

    return EdgeVisionApplication(
        video_source=video_source or create_video_source(settings.video),
        processing_pipeline=create_processing_pipeline(settings),
        renderer=None if no_display else Renderer(show_fps=settings.display.show_fps),
        display=(
            None
            if no_display
            else display or WindowDisplay(window_name=settings.display.window_name)
        ),
        max_frames=max_frames,
        result_writer=create_result_writer(settings.storage),
        result_callback=result_callback,
    )


def create_processing_pipeline(settings: AppSettings) -> ProcessingPipeline:
    """Create the processing pipeline for the configured detector runtime."""
    return ProcessingPipeline(
        preprocessor=FramePreprocessor.from_model_settings(
            settings.model,
            low_light_settings=settings.low_light,
        ),
        detector=_create_detector(settings),
        postprocessor=DetectionPostProcessor(
            confidence_threshold=settings.model.confidence_threshold,
            max_detections=settings.processing.max_detections,
        ),
        fps_counter=FPSCounter(),
        profiler=Profiler(),
    )


def _create_detector(settings: AppSettings) -> ObjectDetector:
    if settings.model.runtime == "mock":
        return MockObjectDetector()
    if settings.model.runtime == "tflite":
        return TFLiteObjectDetector(
            model_path=settings.model.model_path,
            labels_path=settings.model.labels_path,
        )
    raise ApplicationError(f"Unsupported model runtime: {settings.model.runtime}")
