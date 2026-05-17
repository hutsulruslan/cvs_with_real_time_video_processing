from __future__ import annotations

from typing import Protocol

from edge_vision.core.result import FrameResult


class ResultWriter(Protocol):
    """Interface for optional frame result persistence."""

    def write(self, result: FrameResult) -> None:
        """Persist one processed frame result."""

    def close(self) -> None:
        """Release writer resources."""
