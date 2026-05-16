from __future__ import annotations

from abc import ABC, abstractmethod

from edge_vision.core.frame import FramePacket


class VideoSource(ABC):
    """Common interface for camera and file-based video sources."""

    @abstractmethod
    def open(self) -> None:
        """Open the underlying video source."""

    @abstractmethod
    def read(self) -> FramePacket | None:
        """Read the next frame, or return None when no frame is available."""

    @abstractmethod
    def release(self) -> None:
        """Release source resources."""

    def __enter__(self) -> VideoSource:
        self.open()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.release()
