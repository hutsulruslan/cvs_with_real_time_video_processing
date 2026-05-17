from __future__ import annotations

import csv
from pathlib import Path
from typing import TextIO

from edge_vision.core.result import FrameResult
from edge_vision.storage.serialization import CSV_FIELD_NAMES, frame_result_to_csv_rows


class CSVResultWriter:
    """Write frame results as one CSV row per detection."""

    def __init__(self, output_path: str | Path) -> None:
        self._path = Path(output_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file: TextIO | None = self._path.open(
            "w", encoding="utf-8", newline=""
        )
        self._writer = csv.DictWriter(self._file, fieldnames=CSV_FIELD_NAMES)
        self._writer.writeheader()

    def write(self, result: FrameResult) -> None:
        """Write one frame result."""
        if self._file is None:
            raise ValueError("CSVResultWriter is closed.")
        self._writer.writerows(frame_result_to_csv_rows(result))

    def close(self) -> None:
        """Close the CSV file. Safe to call more than once."""
        if self._file is None:
            return
        self._file.close()
        self._file = None
