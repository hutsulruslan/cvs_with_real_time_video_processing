from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from edge_vision.app.runner import build_arg_parser, run_cli
from edge_vision.app.preflight import PreflightCheckResult, PreflightReport
from edge_vision.core.detection import Detection
from edge_vision.core.result import FrameResult


def test_run_cli_parser_accepts_report_flag() -> None:
    args = build_arg_parser("config.yaml").parse_args(["--report"])

    assert args.report is True


def test_run_cli_parser_accepts_preflight_flag() -> None:
    args = build_arg_parser("config.yaml").parse_args(["--preflight"])

    assert args.preflight is True


def test_run_cli_parser_accepts_frame_skip_override() -> None:
    args = build_arg_parser("config.yaml").parse_args(["--frame-skip", "2"])

    assert args.frame_skip == 2


def test_run_cli_parser_accepts_low_light_and_confidence_overrides() -> None:
    args = build_arg_parser("config.yaml").parse_args(
        [
            "--low-light",
            "gamma_clahe",
            "--gamma",
            "2.0",
            "--brightness-threshold",
            "90",
            "--clahe-clip-limit",
            "3.0",
            "--clahe-tile-grid-size",
            "8",
            "--confidence-threshold",
            "0.25",
        ]
    )

    assert args.low_light == "gamma_clahe"
    assert args.gamma == 2.0
    assert args.brightness_threshold == 90
    assert args.clahe_clip_limit == 3.0
    assert args.clahe_tile_grid_size == 8
    assert args.confidence_threshold == 0.25


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


def test_run_cli_check_config_accepts_low_light_and_confidence_overrides(
    tmp_path: Path,
    capsys,
) -> None:
    config_path = _write_config(tmp_path)
    original_config = config_path.read_text(encoding="utf-8")

    exit_code = run_cli(
        [
            "--config",
            str(config_path),
            "--check-config",
            "--low-light",
            "auto",
            "--gamma",
            "1.8",
            "--brightness-threshold",
            "90",
            "--confidence-threshold",
            "0.25",
        ]
    )

    assert exit_code == 0
    assert "source: camera" in capsys.readouterr().out
    assert config_path.read_text(encoding="utf-8") == original_config


def test_run_cli_preflight_prints_report_without_opening_app(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    config_path = _write_config(tmp_path)

    monkeypatch.setattr(
        "edge_vision.app.runner.create_application",
        _raise_if_application_is_created,
    )
    monkeypatch.setattr("edge_vision.app.runner.run_preflight", _fake_preflight)

    exit_code = run_cli(
        [
            "--config",
            str(config_path),
            "--preflight",
            "--profile",
            "mock-camera",
            "--max-frames",
            "1",
            "--no-display",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Preflight report:" in output
    assert "Result: OK" in output


def test_run_cli_report_mode_prints_summary_without_display(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    config_path = _write_config(tmp_path)

    monkeypatch.setattr(
        "edge_vision.app.runner.create_application",
        _fake_application_factory,
    )

    exit_code = run_cli(
        [
            "--config",
            str(config_path),
            "--profile",
            "mock-file",
            "--max-frames",
            "2",
            "--no-display",
            "--report",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Processed frames: 2" in output
    assert "Performance report:" in output
    assert "processed_frames: 2" in output
    assert "total_detections: 3" in output
    assert "average_fps: 15.00" in output


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


def test_run_cli_check_config_rejects_invalid_confidence_threshold(
    tmp_path: Path,
    capsys,
) -> None:
    config_path = _write_config(tmp_path)

    exit_code = run_cli(
        ["--config", str(config_path), "--check-config", "--confidence-threshold", "1.1"]
    )

    assert exit_code == 1
    assert "--confidence-threshold" in capsys.readouterr().err


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


def _fake_application_factory(*args, **kwargs) -> "FakeApplication":
    assert kwargs["max_frames"] == 2
    assert kwargs["no_display"] is True
    return FakeApplication(kwargs["result_callback"])


def _raise_if_application_is_created(*args, **kwargs) -> None:
    raise AssertionError("preflight must not create the application")


def _fake_preflight(*args, **kwargs) -> PreflightReport:
    return PreflightReport([PreflightCheckResult("config", True, "configuration loaded")])


class FakeApplication:
    def __init__(self, result_callback) -> None:
        self._result_callback = result_callback

    def run(self) -> int:
        self._result_callback(_result(1, 1, 10.0, 5.0, 30.0))
        self._result_callback(_result(2, 2, 20.0, 15.0, 50.0))
        return 2


def _result(
    frame_id: int,
    detections: int,
    fps: float,
    inference_ms: float,
    total_frame_ms: float,
) -> FrameResult:
    return FrameResult(
        frame_id=frame_id,
        timestamp_ms=float(frame_id),
        detections=[Detection(index, "object", 0.9, 1, 2, 3, 4) for index in range(detections)],
        fps=fps,
        inference_ms=inference_ms,
        total_frame_ms=total_frame_ms,
    )
