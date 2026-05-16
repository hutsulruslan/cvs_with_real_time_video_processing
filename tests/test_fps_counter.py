from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from edge_vision.metrics.fps_counter import FPSCounter


def test_initial_fps_is_zero() -> None:
    counter = FPSCounter(time_provider=FakeClock().now)

    assert counter.get_fps() == 0.0


def test_update_increments_frame_count_and_returns_fps() -> None:
    clock = FakeClock(start=10.0)
    counter = FPSCounter(time_provider=clock.now)

    clock.set(12.0)
    fps = counter.update()

    assert fps == 0.5
    assert counter.get_fps() == 0.5


def test_multiple_updates_calculate_fps_from_total_frames() -> None:
    clock = FakeClock()
    counter = FPSCounter(time_provider=clock.now)

    clock.set(1.0)
    counter.update()
    clock.set(2.0)
    fps = counter.update(frames=3)

    assert fps == 2.0


def test_zero_elapsed_time_does_not_crash() -> None:
    clock = FakeClock(start=5.0)
    counter = FPSCounter(time_provider=clock.now)

    assert counter.update() == 0.0


def test_reset_clears_state() -> None:
    clock = FakeClock()
    counter = FPSCounter(time_provider=clock.now)

    clock.set(1.0)
    counter.update(frames=2)
    counter.reset()

    assert counter.get_fps() == 0.0
    clock.set(3.0)
    assert counter.update() == 0.5


def test_update_rejects_non_positive_frame_increment() -> None:
    counter = FPSCounter(time_provider=FakeClock().now)

    with pytest.raises(ValueError, match="positive"):
        counter.update(frames=0)


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self._time = start

    def now(self) -> float:
        return self._time

    def set(self, value: float) -> None:
        self._time = value
