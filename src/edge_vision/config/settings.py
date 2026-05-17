from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


VideoSourceType = Literal["camera", "file", "picamera2"]
ModelRuntime = Literal["mock", "tflite"]
StorageFormat = Literal["csv", "json"]


@dataclass(frozen=True, slots=True)
class VideoSettings:
    source_type: VideoSourceType
    camera_index: int
    file_path: str
    width: int
    height: int


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
class ProcessingSettings:
    frame_skip: int
    enable_tracking: bool
    max_detections: int


@dataclass(frozen=True, slots=True)
class DisplaySettings:
    show_window: bool
    show_fps: bool
    window_name: str


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
