from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from edge_vision.app.application_factory import create_application
from edge_vision.config.settings import (
    AppSettings,
    DisplaySettings,
    ModelSettings,
    ProcessingSettings,
    StorageSettings,
    VideoSettings,
)
from edge_vision.core.errors import ApplicationError
from edge_vision.core.frame import FramePacket
from edge_vision.inference.model_loader import TFLiteRuntimeError
from edge_vision.video.video_source import VideoSource


def test_create_application_runs_mock_visual_mode_with_overrides() -> None:
    source = FakeVideoSource([_packet()])
    display = FakeDisplay()

    processed_frames = create_application(
        _settings(),
        video_source=source,
        display=display,
        max_frames=1,
    ).run()

    assert processed_frames == 1
    assert source.released is True
    assert display.closed is True
    assert len(display.shown_frames) == 1
    assert display.shown_frames[0].sum() > 0


def test_create_application_reports_missing_tflite_model_file(tmp_path: Path) -> None:
    with pytest.raises(TFLiteRuntimeError, match="model file"):
        create_application(
            _settings(runtime="tflite", model_path=str(tmp_path / "missing_model.tflite")),
            video_source=FakeVideoSource([]),
            display=FakeDisplay(),
        )


def test_create_application_rejects_disabled_window_for_visual_mode() -> None:
    with pytest.raises(ApplicationError, match="show_window"):
        create_application(_settings(show_window=False), video_source=FakeVideoSource([]), display=FakeDisplay())


def test_create_application_saves_results_when_storage_is_enabled(tmp_path: Path) -> None:
    source = FakeVideoSource([_packet()])
    display = FakeDisplay()

    processed_frames = create_application(
        _settings(save_detections=True, output_dir=str(tmp_path)),
        video_source=source,
        display=display,
        max_frames=1,
    ).run()

    assert processed_frames == 1
    assert (tmp_path / "detections" / "results.csv").exists()


class FakeVideoSource(VideoSource):
    def __init__(self, packets: list[FramePacket]) -> None:
        self._packets = list(packets)
        self.released = False

    def open(self) -> None:
        return None

    def read(self) -> FramePacket | None:
        return self._packets.pop(0) if self._packets else None

    def release(self) -> None:
        self.released = True


class FakeDisplay:
    def __init__(self) -> None:
        self.shown_frames: list[np.ndarray] = []
        self.closed = False

    def show(self, frame: np.ndarray) -> bool:
        self.shown_frames.append(frame)
        return False

    def close(self) -> None:
        self.closed = True


def _settings(
    runtime: str = "mock",
    show_window: bool = True,
    model_path: str = "assets/models/model.tflite",
    labels_path: str = "assets/models/labels.txt",
    save_detections: bool = False,
    output_dir: str = "output",
) -> AppSettings:
    return AppSettings(
        video=VideoSettings("camera", 0, "assets/samples/sample_video.mp4", 640, 480),
        model=ModelSettings(
            runtime,
            model_path,
            labels_path,
            320,
            320,
            0.4,
            0.5,
        ),
        processing=ProcessingSettings(0, False, 20),
        display=DisplaySettings(show_window, True, "Edge Vision System"),
        storage=StorageSettings(save_detections, False, output_dir),
    )


def _packet() -> FramePacket:
    return FramePacket(1, 10.0, np.zeros((480, 640, 3), dtype=np.uint8))
