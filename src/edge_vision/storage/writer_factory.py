from __future__ import annotations

from pathlib import Path

from edge_vision.config.settings import StorageSettings
from edge_vision.core.errors import ConfigurationError
from edge_vision.storage.csv_writer import CSVResultWriter
from edge_vision.storage.json_writer import JSONResultWriter
from edge_vision.storage.result_writer import ResultWriter


def create_result_writer(settings: StorageSettings) -> ResultWriter | None:
    """Create an optional result writer from storage settings."""
    if not settings.save_detections:
        return None

    output_path = _output_path(settings)
    if settings.format == "csv":
        return CSVResultWriter(output_path)
    if settings.format == "json":
        return JSONResultWriter(output_path)
    raise ConfigurationError("storage.format must be csv or json.")


def _output_path(settings: StorageSettings) -> Path:
    return Path(settings.output_dir) / "detections" / f"results.{settings.format}"
