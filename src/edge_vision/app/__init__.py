"""Application-level orchestration components."""

from edge_vision.app.application import EdgeVisionApplication
from edge_vision.app.application_factory import create_application, create_processing_pipeline
from edge_vision.app.pipeline import ProcessingPipeline, VideoProcessingPipeline

__all__ = [
    "EdgeVisionApplication",
    "ProcessingPipeline",
    "VideoProcessingPipeline",
    "create_application",
    "create_processing_pipeline",
]
