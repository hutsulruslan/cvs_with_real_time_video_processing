from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from edge_vision.config.settings import (
    AppSettings,
    DisplaySettings,
    ModelSettings,
    ProcessingSettings,
    StorageSettings,
    VideoSettings,
)
from edge_vision.core.errors import ConfigurationError


REQUIRED_FIELDS = {
    "video": ("source_type", "camera_index", "file_path", "width", "height"),
    "model": (
        "runtime",
        "model_path",
        "labels_path",
        "input_width",
        "input_height",
        "confidence_threshold",
        "nms_threshold",
    ),
    "processing": ("frame_skip", "enable_tracking", "max_detections"),
    "display": ("show_window", "show_fps", "window_name"),
    "storage": ("save_detections", "save_frames", "output_dir"),
}


def load_config(config_path: str | Path) -> AppSettings:
    """Load and validate application settings from a YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise ConfigurationError(f"Configuration file does not exist: {path}")

    with path.open("r", encoding="utf-8") as config_file:
        raw_data = yaml.safe_load(config_file)

    return parse_config_data(raw_data)


def parse_config_data(raw_data: Any) -> AppSettings:
    """Convert raw YAML data into validated application settings."""
    if not isinstance(raw_data, Mapping):
        raise ConfigurationError("Configuration root must be a mapping.")

    sections = {
        name: _required_section(raw_data, name) for name in REQUIRED_FIELDS
    }
    for name, required_fields in REQUIRED_FIELDS.items():
        _require_fields(sections[name], name, required_fields)

    settings = AppSettings(
        video=VideoSettings(
            source_type=sections["video"]["source_type"],
            camera_index=sections["video"]["camera_index"],
            file_path=sections["video"]["file_path"],
            width=sections["video"]["width"],
            height=sections["video"]["height"],
        ),
        model=ModelSettings(
            runtime=sections["model"]["runtime"],
            model_path=sections["model"]["model_path"],
            labels_path=sections["model"]["labels_path"],
            input_width=sections["model"]["input_width"],
            input_height=sections["model"]["input_height"],
            confidence_threshold=sections["model"]["confidence_threshold"],
            nms_threshold=sections["model"]["nms_threshold"],
            normalize=sections["model"].get("normalize", False),
        ),
        processing=ProcessingSettings(
            frame_skip=sections["processing"]["frame_skip"],
            enable_tracking=sections["processing"]["enable_tracking"],
            max_detections=sections["processing"]["max_detections"],
        ),
        display=DisplaySettings(
            show_window=sections["display"]["show_window"],
            show_fps=sections["display"]["show_fps"],
            window_name=sections["display"]["window_name"],
        ),
        storage=StorageSettings(
            save_detections=sections["storage"]["save_detections"],
            save_frames=sections["storage"]["save_frames"],
            output_dir=sections["storage"]["output_dir"],
        ),
    )
    _validate_settings(settings)
    return settings


def _required_section(data: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    section = data.get(name)
    if not isinstance(section, Mapping):
        raise ConfigurationError(f"Missing or invalid '{name}' section.")
    return section


def _require_fields(
    section: Mapping[str, Any], section_name: str, field_names: tuple[str, ...]
) -> None:
    missing = [field for field in field_names if field not in section]
    if missing:
        joined = ", ".join(missing)
        raise ConfigurationError(f"Missing fields in '{section_name}': {joined}")


def _validate_settings(settings: AppSettings) -> None:
    if settings.video.source_type not in {"camera", "file", "picamera2"}:
        raise ConfigurationError("video.source_type must be camera, file, or picamera2.")
    _require_non_negative_int(settings.video.camera_index, "video.camera_index")
    _require_positive_int(settings.video.width, "video.width")
    _require_positive_int(settings.video.height, "video.height")
    _require_text(settings.video.file_path, "video.file_path")

    if settings.model.runtime not in {"mock", "tflite"}:
        raise ConfigurationError("model.runtime must be mock or tflite.")
    _require_text(settings.model.model_path, "model.model_path")
    _require_text(settings.model.labels_path, "model.labels_path")
    _require_positive_int(settings.model.input_width, "model.input_width")
    _require_positive_int(settings.model.input_height, "model.input_height")
    _require_probability(settings.model.confidence_threshold, "model.confidence_threshold")
    _require_probability(settings.model.nms_threshold, "model.nms_threshold")
    _require_bool(settings.model.normalize, "model.normalize")

    _require_non_negative_int(settings.processing.frame_skip, "processing.frame_skip")
    _require_bool(settings.processing.enable_tracking, "processing.enable_tracking")
    _require_positive_int(settings.processing.max_detections, "processing.max_detections")

    _require_bool(settings.display.show_window, "display.show_window")
    _require_bool(settings.display.show_fps, "display.show_fps")
    _require_text(settings.display.window_name, "display.window_name")

    _require_bool(settings.storage.save_detections, "storage.save_detections")
    _require_bool(settings.storage.save_frames, "storage.save_frames")
    _require_text(settings.storage.output_dir, "storage.output_dir")


def _require_positive_int(value: Any, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigurationError(f"{field_name} must be a positive integer.")


def _require_non_negative_int(value: Any, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ConfigurationError(f"{field_name} must be a non-negative integer.")


def _require_probability(value: Any, field_name: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ConfigurationError(f"{field_name} must be a number from 0.0 to 1.0.")
    if value < 0.0 or value > 1.0:
        raise ConfigurationError(f"{field_name} must be between 0.0 and 1.0.")


def _require_bool(value: Any, field_name: str) -> None:
    if not isinstance(value, bool):
        raise ConfigurationError(f"{field_name} must be true or false.")


def _require_text(value: Any, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{field_name} must be a non-empty string.")
