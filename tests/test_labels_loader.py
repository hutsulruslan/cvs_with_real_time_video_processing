from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from edge_vision.inference.labels_loader import LabelsError, load_labels


def test_load_labels_reads_one_non_empty_label_per_line(tmp_path: Path) -> None:
    labels_path = tmp_path / "labels.txt"
    labels_path.write_text("person\n\ncar\n  bicycle  \n", encoding="utf-8")

    labels = load_labels(labels_path)

    assert labels.labels == ("person", "car", "bicycle")


def test_label_map_returns_fallback_for_unknown_class_id(tmp_path: Path) -> None:
    labels_path = tmp_path / "labels.txt"
    labels_path.write_text("person\ncar\n", encoding="utf-8")

    labels = load_labels(labels_path)

    assert labels.get_label(0) == "person"
    assert labels.get_label(8) == "class_8"


def test_load_labels_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(LabelsError, match="Labels file"):
        load_labels(tmp_path / "missing-labels.txt")
