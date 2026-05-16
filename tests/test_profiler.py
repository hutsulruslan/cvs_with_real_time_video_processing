from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from edge_vision.metrics.profiler import Profiler


def test_profiler_measures_one_named_section_in_milliseconds() -> None:
    clock = FakeClock()
    profiler = Profiler(time_provider=clock.now)

    profiler.start("inference")
    clock.set(0.025)
    elapsed_ms = profiler.stop("inference")

    assert elapsed_ms == 25.0
    assert profiler.get_ms("inference") == 25.0


def test_profiler_measures_multiple_sections_independently() -> None:
    clock = FakeClock()
    profiler = Profiler(time_provider=clock.now)

    profiler.start("preprocess")
    clock.set(0.010)
    profiler.stop("preprocess")
    profiler.start("inference")
    clock.set(0.050)
    profiler.stop("inference")

    assert profiler.get_ms("preprocess") == 10.0
    assert profiler.get_ms("inference") == 40.0


def test_as_dict_returns_measured_values_copy() -> None:
    clock = FakeClock()
    profiler = Profiler(time_provider=clock.now)

    profiler.start("frame")
    clock.set(0.033)
    profiler.stop("frame")
    measurements = profiler.as_dict()
    measurements["frame"] = 999.0

    assert profiler.get_ms("frame") == 33.0


def test_reset_clears_measured_sections() -> None:
    clock = FakeClock()
    profiler = Profiler(time_provider=clock.now)

    profiler.start("frame")
    clock.set(0.010)
    profiler.stop("frame")
    profiler.reset()

    assert profiler.as_dict() == {}
    assert profiler.get_ms("frame") == 0.0


def test_stopping_unknown_section_raises_clear_error() -> None:
    profiler = Profiler(time_provider=FakeClock().now)

    with pytest.raises(ValueError, match="not started"):
        profiler.stop("missing")


def test_profiler_rejects_empty_section_names() -> None:
    profiler = Profiler(time_provider=FakeClock().now)

    with pytest.raises(ValueError, match="non-empty"):
        profiler.start("")


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self._time = start

    def now(self) -> float:
        return self._time

    def set(self, value: float) -> None:
        self._time = value
