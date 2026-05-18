"""Video source abstractions and OpenCV-backed implementations."""

from edge_vision.video.opencv_camera_source import OpenCVCameraSource
from edge_vision.video.source_factory import create_video_source
from edge_vision.video.video_file_source import VideoFileSource
from edge_vision.video.video_source import VideoSource
from edge_vision.video.video_stream_source import VideoStreamSource

__all__ = [
    "OpenCVCameraSource",
    "VideoFileSource",
    "VideoSource",
    "VideoStreamSource",
    "create_video_source",
]
