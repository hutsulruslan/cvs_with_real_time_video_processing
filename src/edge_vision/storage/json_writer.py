from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from edge_vision.core.result import FrameResult
from edge_vision.storage.serialization import frame_result_to_dict


class JSONResultWriter:
    """Write frame results as a JSON list when the writer is closed."""

    def __init__(self, output_path: str | Path) -> None:
        self._path = Path(output_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._results: list[dict[str, Any]] = []
        self._closed = False

    def write(self, result: FrameResult) -> None:
        """Store one frame result for later JSON serialization."""
        if self._closed:
            raise ValueError("JSONResultWriter is closed.")
        self._results.append(frame_result_to_dict(result))

    def close(self) -> None:
        """Write the JSON file. Safe to call more than once."""
        if self._closed:
            return
        with self._path.open("w", encoding="utf-8") as output_file:
            json.dump(self._results, output_file, ensure_ascii=False, indent=2)
        self._closed = True
