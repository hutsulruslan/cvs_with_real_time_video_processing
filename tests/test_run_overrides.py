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


@pytest.mark.parametrize(
    ("mode", "enabled"),
    [
        ("off", False),
        ("gamma", True),
        ("clahe", True),
        ("gamma_clahe", True),
        ("auto", True),
    ],
)
def test_low_light_mode_override_sets_mode_and_enabled_state(
    mode: str,
    enabled: bool,
) -> None:
    updated = apply_run_overrides(_settings(), RunOverrides(low_light=mode))

    assert updated.low_light.mode == mode
    assert updated.low_light.enabled is enabled


def test_low_light_numeric_overrides_are_applied() -> None:
    settings = _settings()

    updated = apply_run_overrides(
        settings,
        RunOverrides(
            gamma=1.8,
            brightness_threshold=90,
            clahe_clip_limit=3.0,
            clahe_tile_grid_size=6,
        ),
    )

    assert updated.low_light.gamma == 1.8
    assert updated.low_light.brightness_threshold == 90
    assert updated.low_light.clahe_clip_limit == 3.0
    assert updated.low_light.clahe_tile_grid_size == 6
    assert settings.low_light.gamma == 1.5


def test_confidence_threshold_override_is_applied_without_mutating_settings() -> None:
    settings = _settings()

    updated = apply_run_overrides(settings, RunOverrides(confidence_threshold=0.25))

    assert updated.model.confidence_threshold == 0.25
    assert settings.model.confidence_threshold == 0.4


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


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (RunOverrides(gamma=0), "--gamma"),
        (RunOverrides(brightness_threshold=-1), "--brightness-threshold"),
        (RunOverrides(brightness_threshold=256), "--brightness-threshold"),
        (RunOverrides(clahe_clip_limit=0), "--clahe-clip-limit"),
        (RunOverrides(clahe_tile_grid_size=0), "--clahe-tile-grid-size"),
        (RunOverrides(confidence_threshold=-0.1), "--confidence-threshold"),
        (RunOverrides(confidence_threshold=1.1), "--confidence-threshold"),
    ],
)
def test_validate_run_overrides_rejects_invalid_image_and_threshold_values(
    overrides: RunOverrides,
    message: str,
) -> None:
    with pytest.raises(ApplicationError, match=message):
        validate_run_overrides(_settings(), overrides)


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
