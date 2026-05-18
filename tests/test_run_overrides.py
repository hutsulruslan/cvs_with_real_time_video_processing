from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from edge_vision.app.run_overrides import (
    RunOverrides,
    apply_run_overrides,
    validate_run_overrides,
)
from edge_vision.config.settings import (
    AppSettings,
    DisplaySettings,
    ModelSettings,
    ProcessingSettings,
    StorageSettings,
    VideoSettings,
)
from edge_vision.core.errors import ApplicationError


@pytest.mark.parametrize(
    ("profile", "runtime", "source_type"),
    [
        ("mock-file", "mock", "file"),
        ("mock-camera", "mock", "camera"),
        ("mock-stream", "mock", "stream"),
        ("tflite-file", "tflite", "file"),
        ("tflite-camera", "tflite", "camera"),
        ("tflite-stream", "tflite", "stream"),
    ],
)
def test_profile_applies_runtime_and_source_type(
    profile: str,
    runtime: str,
    source_type: str,
) -> None:
    updated = apply_run_overrides(_settings(), RunOverrides(profile=profile))

    assert updated.model.runtime == runtime
    assert updated.video.source_type == source_type


def test_runtime_field_overrides_are_applied_without_mutating_settings() -> None:
    settings = _settings()

    updated = apply_run_overrides(
        settings,
        RunOverrides(
            camera_index=2,
            file_path="assets/samples/demo.mp4",
            stream_url="http://example.local:8080/video",
        ),
    )

    assert updated.video.camera_index == 2
    assert updated.video.file_path == "assets/samples/demo.mp4"
    assert updated.video.stream_url == "http://example.local:8080/video"
    assert settings.video.camera_index == 0
    assert settings.video.file_path == "assets/samples/sample_video.mp4"
    assert settings.video.stream_url == ""


def test_validate_run_overrides_allows_bounded_headless_camera_run() -> None:
    settings = apply_run_overrides(_settings(), RunOverrides(profile="mock-camera"))

    validate_run_overrides(settings, RunOverrides(no_display=True, max_frames=5))


def test_validate_run_overrides_rejects_unbounded_headless_camera_run() -> None:
    settings = apply_run_overrides(_settings(), RunOverrides(profile="mock-camera"))

    with pytest.raises(ApplicationError, match="requires --max-frames"):
        validate_run_overrides(settings, RunOverrides(no_display=True))


def test_validate_run_overrides_rejects_unbounded_headless_stream_run() -> None:
    settings = apply_run_overrides(_settings(), RunOverrides(profile="mock-stream"))

    with pytest.raises(ApplicationError, match="requires --max-frames"):
        validate_run_overrides(settings, RunOverrides(no_display=True))


def test_validate_run_overrides_allows_bounded_headless_stream_run() -> None:
    settings = apply_run_overrides(_settings(), RunOverrides(profile="mock-stream"))

    validate_run_overrides(settings, RunOverrides(no_display=True, max_frames=5))


def test_validate_run_overrides_rejects_negative_values() -> None:
    settings = _settings()

    with pytest.raises(ApplicationError, match="--camera-index"):
        validate_run_overrides(settings, RunOverrides(camera_index=-1))
    with pytest.raises(ApplicationError, match="--max-frames"):
        validate_run_overrides(settings, RunOverrides(max_frames=-1))
    with pytest.raises(ApplicationError, match="--stream-url"):
        validate_run_overrides(settings, RunOverrides(stream_url=""))


def _settings() -> AppSettings:
    return AppSettings(
        video=VideoSettings("camera", 0, "assets/samples/sample_video.mp4", 640, 480),
        model=ModelSettings(
            "mock",
            "assets/models/model.tflite",
            "assets/models/labels.txt",
            320,
            320,
            0.4,
            0.5,
            False,
        ),
        processing=ProcessingSettings(0, False, 20),
        display=DisplaySettings(True, True, "Edge Vision System"),
        storage=StorageSettings(False, False, "output", "csv"),
    )
