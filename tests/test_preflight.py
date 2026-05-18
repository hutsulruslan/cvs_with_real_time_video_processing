from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from edge_vision.app.preflight import (
    PreflightCheckResult,
    PreflightReport,
    format_preflight_report,
    run_preflight,
)
from edge_vision.app.run_overrides import RunOverrides
from edge_vision.config.settings import (
    AppSettings,
    DisplaySettings,
    ModelSettings,
    ProcessingSettings,
    StorageSettings,
    VideoSettings,
)
from edge_vision.core.frame import FramePacket
from edge_vision.video.video_source import VideoSource


def test_preflight_report_is_ok_when_all_fake_checks_pass(tmp_path: Path) -> None:
    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"fake")
    settings = _settings(video=VideoSettings("file", 0, str(video_path), 640, 480))

    report = run_preflight(settings, RunOverrides(profile="mock-file"), video_source_factory=_source_factory())

    assert report.is_ok is True
    assert "Result: OK" in format_preflight_report(report)


def test_preflight_report_is_failed_when_one_check_fails(tmp_path: Path) -> None:
    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"fake")
    settings = _settings(video=VideoSettings("file", 0, str(video_path), 640, 480))

    report = run_preflight(
        settings,
        RunOverrides(profile="mock-file"),
        video_source_factory=_source_factory_without_frame(),
    )

    assert report.is_ok is False
    assert "video source: FAILED" in format_preflight_report(report)


def test_mock_runtime_does_not_require_model_files(tmp_path: Path) -> None:
    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"fake")
    settings = _settings(
        video=VideoSettings("file", 0, str(video_path), 640, 480),
        model=ModelSettings("mock", "", "", 320, 320, 0.4, 0.5, False),
    )

    report = run_preflight(settings, RunOverrides(profile="mock-file"), video_source_factory=_source_factory())

    assert report.is_ok is True
    assert "model" not in {check.name for check in report.checks}
    assert _message_for(report, "runtime").startswith("mock runtime selected")


def test_tflite_runtime_requires_model_and_labels(tmp_path: Path) -> None:
    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"fake")
    settings = _settings(
        video=VideoSettings("file", 0, str(video_path), 640, 480),
        model=ModelSettings(
            "tflite",
            str(tmp_path / "missing.tflite"),
            str(tmp_path / "missing_labels.txt"),
            320,
            320,
            0.4,
            0.5,
            False,
        ),
    )

    report = run_preflight(
        settings,
        RunOverrides(profile="tflite-file"),
        video_source_factory=_source_factory(),
        runtime_checker=lambda: "fake-litert",
    )

    assert report.is_ok is False
    assert _check(report, "model").ok is False
    assert _check(report, "labels").ok is False


def test_storage_disabled_is_reported(tmp_path: Path) -> None:
    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"fake")
    settings = _settings(video=VideoSettings("file", 0, str(video_path), 640, 480))

    report = run_preflight(settings, RunOverrides(profile="mock-file"), video_source_factory=_source_factory())

    assert _check(report, "storage").ok is True
    assert _message_for(report, "storage") == "disabled"


def test_unsupported_storage_format_fails(tmp_path: Path) -> None:
    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"fake")
    settings = _settings(
        video=VideoSettings("file", 0, str(video_path), 640, 480),
        storage=StorageSettings(True, False, str(tmp_path / "output"), "xml"),
    )

    report = run_preflight(settings, RunOverrides(profile="mock-file"), video_source_factory=_source_factory())

    assert report.is_ok is False
    assert "storage.format" in _message_for(report, "storage")


def test_no_display_live_camera_without_max_frames_fails() -> None:
    settings = _settings(video=VideoSettings("camera", 0, "unused.mp4", 640, 480))

    report = run_preflight(
        settings,
        RunOverrides(profile="mock-camera", no_display=True),
        video_source_factory=_source_factory(),
    )

    assert report.is_ok is False
    assert "requires --max-frames" in _message_for(report, "runtime overrides")
    assert _message_for(report, "display") == "headless mode enabled"


def test_format_preflight_report_contains_useful_messages() -> None:
    report = PreflightReport(
        [
            PreflightCheckResult("config", True, "configuration loaded"),
            PreflightCheckResult("video source", False, "camera index 2 failed"),
        ]
    )

    text = format_preflight_report(report)

    assert "Preflight report:" in text
    assert "config: OK configuration loaded" in text
    assert "video source: FAILED camera index 2 failed" in text
    assert "Result: FAILED" in text


class FakeVideoSource(VideoSource):
    def __init__(self, frame: np.ndarray | None = None) -> None:
        self._frame = frame
        self.released = False

    def open(self) -> None:
        return None

    def read(self) -> FramePacket | None:
        if self._frame is None:
            return None
        return FramePacket(1, 10.0, self._frame)

    def release(self) -> None:
        self.released = True


def _source_factory():
    def create_source(video_settings: VideoSettings) -> FakeVideoSource:
        return FakeVideoSource(np.zeros((480, 640, 3), dtype=np.uint8))

    return create_source


def _source_factory_without_frame():
    def create_source(video_settings: VideoSettings) -> FakeVideoSource:
        return FakeVideoSource(None)

    return create_source


def _settings(
    *,
    video: VideoSettings | None = None,
    model: ModelSettings | None = None,
    storage: StorageSettings | None = None,
) -> AppSettings:
    return AppSettings(
        video=video or VideoSettings("camera", 0, "assets/samples/sample_video.mp4", 640, 480),
        model=model or ModelSettings("mock", "missing.tflite", "missing.txt", 320, 320, 0.4, 0.5, False),
        processing=ProcessingSettings(0, False, 20),
        display=DisplaySettings(True, True, "Edge Vision System"),
        storage=storage or StorageSettings(False, False, "output", "csv"),
    )


def _check(report: PreflightReport, name: str) -> PreflightCheckResult:
    return next(check for check in report.checks if check.name == name)


def _message_for(report: PreflightReport, name: str) -> str:
    return _check(report, name).message
