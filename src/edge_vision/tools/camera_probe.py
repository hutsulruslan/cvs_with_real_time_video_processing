from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from edge_vision.core.errors import VideoSourceError


CaptureFactory = Callable[[int], Any]
CAP_PROP_FRAME_WIDTH = 3
CAP_PROP_FRAME_HEIGHT = 4


@dataclass(frozen=True, slots=True)
class CameraProbeResult:
    """Result of probing one OpenCV camera index."""

    index: int
    opened: bool
    frame_read: bool
    width: int | None
    height: int | None
    error: str | None = None


def probe_camera_indexes(
    max_index: int = 5,
    capture_factory: CaptureFactory | None = None,
) -> list[CameraProbeResult]:
    """Probe OpenCV camera indexes from 0 through max_index."""
    if max_index < 0:
        return []

    factory = capture_factory or _load_capture_factory()
    return [_probe_camera_index(index, factory) for index in range(max_index + 1)]


def format_probe_results(results: list[CameraProbeResult]) -> str:
    """Format camera probe results for manual terminal use."""
    lines = ["Camera probe results:"]
    for result in results:
        size = _format_size(result.width, result.height)
        line = (
            f"- index {result.index}: opened={_yes_no(result.opened)}, "
            f"frame_read={_yes_no(result.frame_read)}, size={size}"
        )
        if result.error:
            line = f"{line}, error={result.error}"
        lines.append(line)
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Probe OpenCV camera indexes.")
    parser.add_argument(
        "--max-index", type=int, default=5, help="Highest camera index to check, inclusive."
    )
    args = parser.parse_args(argv)

    try:
        results = probe_camera_indexes(max_index=args.max_index)
    except VideoSourceError as error:
        print(error)
        return 1

    print(format_probe_results(results))
    return 0


def _probe_camera_index(index: int, factory: CaptureFactory) -> CameraProbeResult:
    capture: Any | None = None
    opened = False
    try:
        capture = factory(index)
        opened = bool(capture.isOpened())
        if not opened:
            return CameraProbeResult(index, False, False, None, None)

        try:
            frame_read, frame = capture.read()
        except Exception as error:
            width, height = _capture_size(capture)
            return CameraProbeResult(index, True, False, width, height, str(error))

        width, height = _frame_size(frame) if frame_read else (None, None)
        if width is None or height is None:
            width, height = _capture_size(capture)
        return CameraProbeResult(index, True, bool(frame_read), width, height)
    except Exception as error:
        return CameraProbeResult(index, opened, False, None, None, str(error))
    finally:
        _release_capture(capture)


def _load_capture_factory() -> CaptureFactory:
    try:
        import cv2
    except ImportError as error:
        raise VideoSourceError("OpenCV is required to probe camera indexes.") from error
    return cv2.VideoCapture


def _frame_size(frame: Any) -> tuple[int | None, int | None]:
    shape = getattr(frame, "shape", None)
    if shape is None or len(shape) < 2:
        return None, None
    return int(shape[1]), int(shape[0])


def _capture_size(capture: Any) -> tuple[int | None, int | None]:
    get_property = getattr(capture, "get", None)
    if get_property is None:
        return None, None

    try:
        width = int(get_property(CAP_PROP_FRAME_WIDTH) or 0)
        height = int(get_property(CAP_PROP_FRAME_HEIGHT) or 0)
    except Exception:
        return None, None

    if width <= 0 or height <= 0:
        return None, None
    return width, height


def _release_capture(capture: Any | None) -> None:
    if capture is None:
        return
    release = getattr(capture, "release", None)
    if release is not None:
        try:
            release()
        except Exception:
            pass


def _format_size(width: int | None, height: int | None) -> str:
    if width is None or height is None:
        return "unknown"
    return f"{width}x{height}"


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


if __name__ == "__main__":
    raise SystemExit(main())
