from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from edge_vision.core.errors import EdgeVisionError


class LabelsError(EdgeVisionError):
    """Raised when a labels file cannot be loaded."""


@dataclass(frozen=True, slots=True)
class LabelMap:
    """Class id to label lookup loaded from labels.txt."""

    labels: tuple[str, ...]

    def get_label(self, class_id: int) -> str:
        if 0 <= class_id < len(self.labels):
            return self.labels[class_id]
        return f"class_{class_id}"


def load_labels(labels_path: str | Path) -> LabelMap:
    """Load one non-empty class label per line from a text file."""
    path = Path(labels_path)
    if not path.exists():
        raise LabelsError(f"Labels file does not exist: {path}")

    with path.open("r", encoding="utf-8") as labels_file:
        labels = tuple(line.strip() for line in labels_file if line.strip())

    return LabelMap(labels)
