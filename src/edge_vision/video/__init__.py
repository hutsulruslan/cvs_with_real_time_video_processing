"""Video source abstractions and OpenCV-backed implementations."""

from edge_vision.video.opencv_camera_source import OpenCVCameraSource
from edge_vision.video.source_factory import create_video_source
from edge_vision.video.video_file_source import VideoFileSource
from edge_vision.video.video_source import VideoSource

__all__ = [
    "OpenCVCameraSource",
    "VideoFileSource",
    "VideoSource",
    "create_video_source",
]
