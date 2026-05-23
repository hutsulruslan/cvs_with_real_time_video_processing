from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


VideoSourceType = Literal["camera", "file", "stream", "picamera2"]
ModelRuntime = Literal["mock", "tflite"]
StorageFormat = Literal["csv", "json"]
LowLightMode = Literal["off", "gamma", "clahe", "gamma_clahe", "auto"]
PipelineMode = Literal["sequential", "low_latency"]


@dataclass(frozen=True, slots=True)
class VideoSettings:
    source_type: VideoSourceType
    camera_index: int
    file_path: str
    width: int
    height: int
    stream_url: str = ""
    file_source_fps: float | None = None


@dataclass(frozen=True, slots=True)
class ModelSettings:
    runtime: ModelRuntime
    model_path: str
    labels_path: str
    input_width: int
    input_height: int
    confidence_threshold: float
    nms_threshold: float
    normalize: bool = False


@dataclass(frozen=True, slots=True)
class LowLightSettings:
    enabled: bool = False
    mode: LowLightMode = "off"
    gamma: float = 1.5
    brightness_threshold: float = 65.0
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: int = 8


@dataclass(frozen=True, slots=True)
class BoxSmoothingSettings:
    enabled: bool = False
    alpha: float = 0.6
    iou_threshold: float = 0.3


@dataclass(frozen=True, slots=True)
class ProcessingSettings:
    frame_skip: int
    enable_tracking: bool
    max_detections: int
    pipeline_mode: PipelineMode = "sequential"


@dataclass(frozen=True, slots=True)
class DisplaySettings:
    show_window: bool
    show_fps: bool
    window_name: str
    show_debug_overlay: bool = False


@dataclass(frozen=True, slots=True)
class StorageSettings:
    save_detections: bool
    save_frames: bool
    output_dir: str
    format: StorageFormat = "csv"


@dataclass(frozen=True, slots=True)
class AppSettings:
    video: VideoSettings
    model: ModelSettings
    processing: ProcessingSettings
    display: DisplaySettings
    storage: StorageSettings
    low_light: LowLightSettings = field(default_factory=LowLightSettings)
    box_smoothing: BoxSmoothingSettings = field(default_factory=BoxSmoothingSettings)
