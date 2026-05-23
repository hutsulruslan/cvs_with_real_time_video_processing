from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from edge_vision.config.settings import (
    AppSettings,
    LowLightMode,
    ModelRuntime,
    PipelineMode,
    VideoSourceType,
)
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
    file_source_fps: float | None = None
    stream_url: str | None = None
    max_frames: int | None = None
    frame_skip: int | None = None
    pipeline_mode: PipelineMode | None = None
    no_display: bool = False
    low_light: LowLightMode | None = None
    gamma: float | None = None
    brightness_threshold: float | None = None
    clahe_clip_limit: float | None = None
    clahe_tile_grid_size: int | None = None
    confidence_threshold: float | None = None


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
    low_light = settings.low_light

    if overrides.profile is not None:
        runtime, source_type = _profile_values(overrides.profile)
        model = replace(model, runtime=runtime)
        video = replace(video, source_type=source_type)

    if overrides.camera_index is not None:
        video = replace(video, camera_index=overrides.camera_index)
    if overrides.file_path is not None:
        video = replace(video, file_path=overrides.file_path)
    if overrides.file_source_fps is not None:
        video = replace(video, file_source_fps=overrides.file_source_fps)
    if overrides.stream_url is not None:
        video = replace(video, stream_url=overrides.stream_url)
    if overrides.confidence_threshold is not None:
        model = replace(model, confidence_threshold=overrides.confidence_threshold)
    if overrides.frame_skip is not None:
        settings = replace(
            settings,
            processing=replace(settings.processing, frame_skip=overrides.frame_skip),
        )
    if overrides.pipeline_mode is not None:
        settings = replace(
            settings,
            processing=replace(
                settings.processing,
                pipeline_mode=overrides.pipeline_mode,
            ),
        )

    if overrides.low_light is not None:
        low_light = replace(
            low_light,
            enabled=overrides.low_light != "off",
            mode=overrides.low_light,
        )
    if overrides.gamma is not None:
        low_light = replace(low_light, gamma=overrides.gamma)
    if overrides.brightness_threshold is not None:
        low_light = replace(
            low_light,
            brightness_threshold=overrides.brightness_threshold,
        )
    if overrides.clahe_clip_limit is not None:
        low_light = replace(low_light, clahe_clip_limit=overrides.clahe_clip_limit)
    if overrides.clahe_tile_grid_size is not None:
        low_light = replace(low_light, clahe_tile_grid_size=overrides.clahe_tile_grid_size)

    return replace(settings, video=video, model=model, low_light=low_light)


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
    if overrides.frame_skip is not None and overrides.frame_skip < 0:
        raise ApplicationError("--frame-skip must be non-negative.")
    if overrides.file_source_fps is not None and (
        isinstance(overrides.file_source_fps, bool)
        or overrides.file_source_fps <= 0
    ):
        raise ApplicationError("--file-source-fps must be a positive number.")
    if overrides.pipeline_mode is not None and overrides.pipeline_mode not in {
        "sequential",
        "low_latency",
    }:
        raise ApplicationError("--pipeline-mode must be sequential or low_latency.")
    if overrides.stream_url is not None and not overrides.stream_url.strip():
        raise ApplicationError("--stream-url must be a non-empty string.")
    if overrides.low_light is not None and overrides.low_light not in {
        "off",
        "gamma",
        "clahe",
        "gamma_clahe",
        "auto",
    }:
        raise ApplicationError("--low-light must be off, gamma, clahe, gamma_clahe, or auto.")
    if overrides.gamma is not None and (
        isinstance(overrides.gamma, bool) or overrides.gamma <= 0
    ):
        raise ApplicationError("--gamma must be a positive number.")
    if (
        overrides.brightness_threshold is not None
        and (
            isinstance(overrides.brightness_threshold, bool)
            or overrides.brightness_threshold < 0
            or overrides.brightness_threshold > 255
        )
    ):
        raise ApplicationError("--brightness-threshold must be between 0 and 255.")
    if overrides.clahe_clip_limit is not None and (
        isinstance(overrides.clahe_clip_limit, bool) or overrides.clahe_clip_limit <= 0
    ):
        raise ApplicationError("--clahe-clip-limit must be a positive number.")
    if overrides.clahe_tile_grid_size is not None and (
        isinstance(overrides.clahe_tile_grid_size, bool)
        or overrides.clahe_tile_grid_size <= 0
    ):
        raise ApplicationError("--clahe-tile-grid-size must be a positive integer.")
    if (
        overrides.confidence_threshold is not None
        and (
            isinstance(overrides.confidence_threshold, bool)
            or overrides.confidence_threshold < 0.0
            or overrides.confidence_threshold > 1.0
        )
    ):
        raise ApplicationError("--confidence-threshold must be between 0.0 and 1.0.")


def _profile_values(profile: RunProfile) -> tuple[ModelRuntime, VideoSourceType]:
    try:
        return PROFILE_SETTINGS[profile]
    except KeyError as error:
        raise ApplicationError(f"Unsupported run profile: {profile}") from error
