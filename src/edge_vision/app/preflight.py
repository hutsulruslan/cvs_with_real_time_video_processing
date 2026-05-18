from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from edge_vision.app.run_overrides import RunOverrides, validate_run_overrides
from edge_vision.config.settings import AppSettings, StorageSettings, VideoSettings
from edge_vision.core.errors import EdgeVisionError
from edge_vision.video.source_factory import create_video_source
from edge_vision.video.video_source import VideoSource


@dataclass(frozen=True, slots=True)
class PreflightCheckResult:
    """One diagnostic check result."""

    name: str
    ok: bool
    message: str


@dataclass(frozen=True, slots=True)
class PreflightReport:
    """Preflight diagnostics for one selected runtime profile."""

    checks: list[PreflightCheckResult]

    @property
    def is_ok(self) -> bool:
        return all(check.ok for check in self.checks)


VideoSourceFactory = Callable[[VideoSettings], VideoSource]
RuntimeChecker = Callable[[], str]


def run_preflight(
    settings: AppSettings,
    overrides: RunOverrides,
    *,
    video_source_factory: VideoSourceFactory = create_video_source,
    runtime_checker: RuntimeChecker | None = None,
) -> PreflightReport:
    """Run lightweight diagnostics without starting the full application loop."""
    runtime_checker = runtime_checker or _check_tflite_runtime
    checks = [
        _ok("config", "configuration loaded"),
        _ok("profile", overrides.profile or "config defaults"),
        _check_runtime_overrides(settings, overrides),
        _check_video_source(settings.video, video_source_factory),
        *_check_model_runtime(settings, runtime_checker),
        _check_storage(settings.storage),
        _check_display(settings, overrides),
    ]
    return PreflightReport(checks)


def format_preflight_report(report: PreflightReport) -> str:
    """Format a preflight report for CLI output."""
    lines = ["Preflight report:"]
    lines.extend(
        f"- {check.name}: {'OK' if check.ok else 'FAILED'} {check.message}"
        for check in report.checks
    )
    lines.append(f"Result: {'OK' if report.is_ok else 'FAILED'}")
    return "\n".join(lines)


def _check_runtime_overrides(
    settings: AppSettings,
    overrides: RunOverrides,
) -> PreflightCheckResult:
    try:
        validate_run_overrides(settings, overrides)
    except EdgeVisionError as error:
        return _failed("runtime overrides", str(error))
    return _ok("runtime overrides", "valid")


def _check_video_source(
    settings: VideoSettings,
    video_source_factory: VideoSourceFactory,
) -> PreflightCheckResult:
    initial_check = _check_video_settings(settings)
    if initial_check is not None:
        return initial_check

    source: VideoSource | None = None
    try:
        source = video_source_factory(settings)
        source.open()
        packet = source.read()
        if packet is None:
            return _failed("video source", f"{settings.source_type} opened but no frame was read")
        return _ok("video source", f"{settings.source_type} {_frame_size(packet.original_frame)}")
    except EdgeVisionError as error:
        return _failed("video source", str(error))
    except Exception as error:
        return _failed("video source", f"unexpected error: {error}")
    finally:
        if source is not None:
            source.release()


def _check_video_settings(settings: VideoSettings) -> PreflightCheckResult | None:
    if settings.source_type == "file":
        if not settings.file_path:
            return _failed("video source", "video.file_path is missing")
        if not Path(settings.file_path).exists():
            return _failed("video source", f"video file does not exist: {settings.file_path}")
    if settings.source_type == "camera" and settings.camera_index < 0:
        return _failed("video source", "video.camera_index must be non-negative")
    if settings.source_type == "stream" and not settings.stream_url.strip():
        return _failed("video source", "video.stream_url is missing")
    return None


def _check_model_runtime(
    settings: AppSettings,
    runtime_checker: RuntimeChecker,
) -> list[PreflightCheckResult]:
    if settings.model.runtime == "mock":
        return [_ok("runtime", "mock runtime selected; model files are not required")]
    if settings.model.runtime != "tflite":
        return [_failed("runtime", f"unsupported runtime: {settings.model.runtime}")]

    checks = [
        _check_required_file("model", settings.model.model_path),
        _check_required_file("labels", settings.model.labels_path),
    ]
    try:
        runtime_name = runtime_checker()
        checks.append(_ok("runtime", f"{runtime_name} available"))
    except EdgeVisionError as error:
        checks.append(_failed("runtime", str(error)))
    except Exception as error:
        checks.append(_failed("runtime", f"unexpected error: {error}"))
    return checks


def _check_required_file(name: str, path_value: str) -> PreflightCheckResult:
    if not path_value:
        return _failed(name, f"{name}_path is missing")
    path = Path(path_value)
    if not path.exists():
        return _failed(name, f"file does not exist: {path}")
    return _ok(name, str(path))


def _check_storage(settings: StorageSettings) -> PreflightCheckResult:
    if not settings.save_detections:
        return _ok("storage", "disabled")
    if settings.format not in {"csv", "json"}:
        return _failed("storage", "storage.format must be csv or json")
    if not settings.output_dir:
        return _failed("storage", "storage.output_dir is missing")

    output_dir = Path(settings.output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        test_file = output_dir / ".preflight_write_test"
        test_file.write_text("ok", encoding="utf-8")
        test_file.unlink(missing_ok=True)
    except OSError as error:
        return _failed("storage", f"output directory is not writable: {error}")
    return _ok("storage", f"enabled {settings.format} in {output_dir}")


def _check_display(
    settings: AppSettings,
    overrides: RunOverrides,
) -> PreflightCheckResult:
    if overrides.no_display:
        return _ok("display", "headless mode enabled")
    if settings.display.show_window:
        return _ok("display", "display requested; window not opened during preflight")
    return _failed("display", "display.show_window is false; use --no-display")


def _check_tflite_runtime() -> str:
    try:
        from tflite_runtime.interpreter import Interpreter as _

        return "tflite-runtime"
    except ImportError:
        return _check_ai_edge_litert_runtime()


def _check_ai_edge_litert_runtime() -> str:
    try:
        from ai_edge_litert.interpreter import Interpreter as _

        return "ai-edge-litert"
    except ImportError:
        return _check_tensorflow_lite_runtime()


def _check_tensorflow_lite_runtime() -> str:
    try:
        import tensorflow as _
    except ImportError as error:
        raise EdgeVisionError(
            "TFLite runtime is not installed. Install tflite-runtime, "
            "ai-edge-litert, TensorFlow, or another LiteRT-compatible runtime."
        ) from error
    return "tensorflow-lite"


def _frame_size(frame: object) -> str:
    shape = getattr(frame, "shape", None)
    if shape is None or len(shape) < 2:
        return "frame read"
    return f"frame {shape[1]}x{shape[0]}"


def _ok(name: str, message: str) -> PreflightCheckResult:
    return PreflightCheckResult(name, True, message)


def _failed(name: str, message: str) -> PreflightCheckResult:
    return PreflightCheckResult(name, False, message)
