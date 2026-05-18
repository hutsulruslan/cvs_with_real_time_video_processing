from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from edge_vision.app.runner import run_cli


def test_run_cli_check_config_applies_profile_without_opening_app(
    tmp_path: Path,
    capsys,
) -> None:
    config_path = _write_config(tmp_path)

    exit_code = run_cli(
        ["--config", str(config_path), "--check-config", "--profile", "mock-file"]
    )

    assert exit_code == 0
    assert "source: file" in capsys.readouterr().out


def test_run_cli_rejects_unbounded_headless_camera_run(tmp_path: Path, capsys) -> None:
    config_path = _write_config(tmp_path)

    exit_code = run_cli(
        ["--config", str(config_path), "--profile", "mock-camera", "--no-display"]
    )

    assert exit_code == 1
    assert "requires --max-frames" in capsys.readouterr().err


def test_run_cli_rejects_unbounded_headless_stream_run(tmp_path: Path, capsys) -> None:
    config_path = _write_config(tmp_path)

    exit_code = run_cli(
        [
            "--config",
            str(config_path),
            "--profile",
            "mock-stream",
            "--stream-url",
            "http://example.local:8080/video",
            "--no-display",
        ]
    )

    assert exit_code == 1
    assert "requires --max-frames" in capsys.readouterr().err


def test_run_cli_check_config_rejects_negative_max_frames(
    tmp_path: Path,
    capsys,
) -> None:
    config_path = _write_config(tmp_path)

    exit_code = run_cli(
        ["--config", str(config_path), "--check-config", "--max-frames", "-1"]
    )

    assert exit_code == 1
    assert "--max-frames must be non-negative" in capsys.readouterr().err


def test_run_cli_check_config_applies_stream_profile_and_url(
    tmp_path: Path,
    capsys,
) -> None:
    config_path = _write_config(tmp_path)

    exit_code = run_cli(
        [
            "--config",
            str(config_path),
            "--check-config",
            "--profile",
            "mock-stream",
            "--stream-url",
            "http://example.local:8080/video",
        ]
    )

    assert exit_code == 0
    assert "source: stream" in capsys.readouterr().out


def _write_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
video:
  source_type: "camera"
  camera_index: 0
  file_path: "assets/samples/sample_video.mp4"
  stream_url: ""
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
""",
        encoding="utf-8",
    )
    return config_path
