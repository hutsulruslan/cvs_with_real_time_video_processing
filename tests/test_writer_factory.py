from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from edge_vision.config.settings import StorageSettings
from edge_vision.core.errors import ConfigurationError
from edge_vision.storage.csv_writer import CSVResultWriter
from edge_vision.storage.json_writer import JSONResultWriter
from edge_vision.storage.writer_factory import create_result_writer


def test_writer_factory_returns_none_when_saving_is_disabled(tmp_path: Path) -> None:
    writer = create_result_writer(_settings(tmp_path, save_detections=False))

    assert writer is None
    assert not (tmp_path / "detections").exists()


def test_writer_factory_creates_csv_writer(tmp_path: Path) -> None:
    writer = create_result_writer(_settings(tmp_path, format="csv"))

    try:
        assert isinstance(writer, CSVResultWriter)
        assert (tmp_path / "detections" / "results.csv").exists()
    finally:
        assert writer is not None
        writer.close()


def test_writer_factory_creates_json_writer(tmp_path: Path) -> None:
    writer = create_result_writer(_settings(tmp_path, format="json"))

    try:
        assert isinstance(writer, JSONResultWriter)
        assert not (tmp_path / "detections" / "results.json").exists()
    finally:
        assert writer is not None
        writer.close()


def test_writer_factory_rejects_unsupported_format(tmp_path: Path) -> None:
    settings = _settings(tmp_path, format="xml")

    with pytest.raises(ConfigurationError, match="storage.format"):
        create_result_writer(settings)


def _settings(
    output_dir: Path,
    *,
    save_detections: bool = True,
    format: str = "csv",
) -> StorageSettings:
    return StorageSettings(
        save_detections=save_detections,
        save_frames=False,
        output_dir=str(output_dir),
        format=format,  # type: ignore[arg-type]
    )
