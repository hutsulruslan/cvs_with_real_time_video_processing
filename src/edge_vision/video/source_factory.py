from __future__ import annotations

from edge_vision.config.settings import VideoSettings
from edge_vision.core.errors import VideoSourceError
from edge_vision.video.opencv_camera_source import OpenCVCameraSource
from edge_vision.video.video_file_source import VideoFileSource
from edge_vision.video.video_source import VideoSource


def create_video_source(settings: VideoSettings) -> VideoSource:
    """Create one configured video source for the current application run."""
    if settings.source_type == "camera":
        return OpenCVCameraSource(
            camera_index=settings.camera_index,
            width=settings.width,
            height=settings.height,
        )
    if settings.source_type == "file":
        return VideoFileSource(settings.file_path)
    if settings.source_type == "picamera2":
        raise VideoSourceError("Picamera2 source is planned but not implemented yet.")

    raise VideoSourceError(f"Unsupported video source type: {settings.source_type}")
