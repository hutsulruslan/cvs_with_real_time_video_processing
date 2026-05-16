"""Shared data structures and application errors."""

from edge_vision.core.detection import Detection
from edge_vision.core.errors import ApplicationError, ConfigurationError, EdgeVisionError
from edge_vision.core.frame import FramePacket, PreprocessedFrame
from edge_vision.core.result import FrameResult

__all__ = [
    "ApplicationError",
    "ConfigurationError",
    "Detection",
    "EdgeVisionError",
    "FramePacket",
    "FrameResult",
    "PreprocessedFrame",
]
