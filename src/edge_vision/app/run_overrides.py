from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from edge_vision.config.settings import AppSettings, ModelRuntime, VideoSourceType
from edge_vision.core.errors import ApplicationError


RunProfile = Literal[
    "mock-file",
    "mock-camera",
    "mock-stream",
    "tflite-file",
    "tflite-camera",
    "tflite-stream",
]


@dataclass(frozen=True, slots=True)
class RunOverrides:
    """Runtime-only overrides applied after loading config.yaml."""

    profile: RunProfile | None = None
    camera_index: int | None = None
    file_path: str | None = None
    stream_url: str | None = None
    max_frames: int | None = None
    no_display: bool = False


PROFILE_SETTINGS: dict[RunProfile, tuple[ModelRuntime, VideoSourceType]] = {
    "mock-file": ("mock", "file"),
    "mock-camera": ("mock", "camera"),
    "mock-stream": ("mock", "stream"),
    "tflite-file": ("tflite", "file"),
    "tflite-camera": ("tflite", "camera"),
    "tflite-stream": ("tflite", "stream"),
}


def apply_run_overrides(
    settings: AppSettings,
    overrides: RunOverrides,
) -> AppSettings:
    """Apply runtime profile and field overrides without modifying config.yaml."""
    video = settings.video
    model = settings.model

    if overrides.profile is not None:
        runtime, source_type = _profile_values(overrides.profile)
        model = replace(model, runtime=runtime)
        video = replace(video, source_type=source_type)

    if overrides.camera_index is not None:
        video = replace(video, camera_index=overrides.camera_index)
    if overrides.file_path is not None:
        video = replace(video, file_path=overrides.file_path)
    if overrides.stream_url is not None:
        video = replace(video, stream_url=overrides.stream_url)

    return replace(settings, video=video, model=model)


def validate_run_overrides(settings: AppSettings, overrides: RunOverrides) -> None:
    """Validate runtime-only settings that are unsafe for actual execution."""
    validate_override_values(overrides)
    if (
        overrides.no_display
        and settings.video.source_type in {"camera", "stream"}
        and overrides.max_frames is None
    ):
        raise ApplicationError(
            "--no-display with a live camera or stream requires --max-frames "
            "to avoid an endless headless run."
        )


def validate_override_values(overrides: RunOverrides) -> None:
    """Validate runtime override values independent of source type."""
    if overrides.camera_index is not None and overrides.camera_index < 0:
        raise ApplicationError("--camera-index must be non-negative.")
    if overrides.max_frames is not None and overrides.max_frames < 0:
        raise ApplicationError("--max-frames must be non-negative.")
    if overrides.stream_url is not None and not overrides.stream_url.strip():
        raise ApplicationError("--stream-url must be a non-empty string.")


def _profile_values(profile: RunProfile) -> tuple[ModelRuntime, VideoSourceType]:
    try:
        return PROFILE_SETTINGS[profile]
    except KeyError as error:
        raise ApplicationError(f"Unsupported run profile: {profile}") from error
