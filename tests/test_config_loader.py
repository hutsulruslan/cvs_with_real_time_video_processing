from __future__ import annotations

import sys
from pathlib import Path
from textwrap import dedent

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from edge_vision.config.config_loader import load_config, parse_config_data
from edge_vision.core.errors import ConfigurationError


def test_project_default_config_uses_mock_runtime() -> None:
    settings = load_config(PROJECT_ROOT / "config.yaml")

    assert settings.model.runtime == "mock"


def test_load_config_returns_typed_settings(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(_valid_config_yaml(), encoding="utf-8")

    settings = load_config(config_path)

    assert settings.video.source_type == "camera"
    assert settings.video.width == 640
    assert settings.model.runtime == "mock"
    assert settings.model.confidence_threshold == 0.4
    assert settings.model.normalize is False
    assert settings.processing.max_detections == 20
    assert settings.display.window_name == "Edge Vision System"
    assert settings.storage.output_dir == "output"
    assert settings.storage.format == "csv"


def test_parse_config_rejects_missing_section() -> None:
    raw_config = _valid_config_dict()
    raw_config.pop("model")

    with pytest.raises(ConfigurationError, match="model"):
        parse_config_data(raw_config)


def test_parse_config_rejects_invalid_source_type() -> None:
    raw_config = _valid_config_dict()
    raw_config["video"]["source_type"] = "stream"

    with pytest.raises(ConfigurationError, match="source_type"):
        parse_config_data(raw_config)


def test_parse_config_rejects_invalid_threshold() -> None:
    raw_config = _valid_config_dict()
    raw_config["model"]["confidence_threshold"] = 1.5

    with pytest.raises(ConfigurationError, match="confidence_threshold"):
        parse_config_data(raw_config)


def test_parse_config_accepts_tflite_runtime_for_future_detector() -> None:
    raw_config = _valid_config_dict()
    raw_config["model"]["runtime"] = "tflite"

    settings = parse_config_data(raw_config)

    assert settings.model.runtime == "tflite"


def test_parse_config_accepts_model_normalization_flag() -> None:
    raw_config = _valid_config_dict()
    raw_config["model"]["normalize"] = True

    settings = parse_config_data(raw_config)

    assert settings.model.normalize is True


def test_parse_config_rejects_invalid_model_normalization_flag() -> None:
    raw_config = _valid_config_dict()
    raw_config["model"]["normalize"] = "false"

    with pytest.raises(ConfigurationError, match="model.normalize"):
        parse_config_data(raw_config)


def test_parse_config_accepts_storage_json_format() -> None:
    raw_config = _valid_config_dict()
    raw_config["storage"]["format"] = "json"

    settings = parse_config_data(raw_config)

    assert settings.storage.format == "json"


def test_parse_config_rejects_invalid_storage_format() -> None:
    raw_config = _valid_config_dict()
    raw_config["storage"]["format"] = "xml"

    with pytest.raises(ConfigurationError, match="storage.format"):
        parse_config_data(raw_config)


def _valid_config_yaml() -> str:
    return dedent(
        """
        video:
          source_type: "camera"
          camera_index: 0
          file_path: "assets/samples/sample_video.mp4"
          width: 640
          height: 480

        model:
          runtime: "mock"
          model_path: "assets/models/model.tflite"
          labels_path: "assets/models/labels.txt"
          input_width: 320
          input_height: 320
          confidence_threshold: 0.4
          nms_threshold: 0.5
          normalize: false

        processing:
          frame_skip: 0
          enable_tracking: false
          max_detections: 20

        display:
          show_window: true
          show_fps: true
          window_name: "Edge Vision System"

        storage:
          save_detections: false
          save_frames: false
          output_dir: "output"
          format: "csv"
        """
    )


def _valid_config_dict() -> dict:
    return {
        "video": {
            "source_type": "camera",
            "camera_index": 0,
            "file_path": "assets/samples/sample_video.mp4",
            "width": 640,
            "height": 480,
        },
        "model": {
            "runtime": "mock",
            "model_path": "assets/models/model.tflite",
            "labels_path": "assets/models/labels.txt",
            "input_width": 320,
            "input_height": 320,
            "confidence_threshold": 0.4,
            "nms_threshold": 0.5,
        },
        "processing": {
            "frame_skip": 0,
            "enable_tracking": False,
            "max_detections": 20,
        },
        "display": {
            "show_window": True,
            "show_fps": True,
            "window_name": "Edge Vision System",
        },
        "storage": {
            "save_detections": False,
            "save_frames": False,
            "output_dir": "output",
            "format": "csv",
        },
    }
