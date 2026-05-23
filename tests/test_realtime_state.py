from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from edge_vision.app.realtime_state import LatestFrameBuffer, LatestResultStore
from edge_vision.core.frame import FramePacket
from edge_vision.core.result import FrameResult


def test_latest_frame_buffer_starts_empty() -> None:
    buffer = LatestFrameBuffer()

    assert buffer.get_latest() is None
    assert buffer.pop_latest() is None
    assert buffer.dropped_frames == 0


def test_latest_frame_buffer_returns_stored_frame_without_removing_it() -> None:
    buffer = LatestFrameBuffer()
    frame = _frame(1)

    buffer.put(frame)

    assert buffer.get_latest() is frame
    assert buffer.get_latest() is frame
    assert buffer.dropped_frames == 0


def test_latest_frame_buffer_replaces_unconsumed_frame_and_counts_drop() -> None:
    buffer = LatestFrameBuffer()
    first = _frame(1)
    second = _frame(2)

    buffer.put(first)
    buffer.put(second)

    assert buffer.get_latest() is second
    assert buffer.dropped_frames == 1


def test_latest_frame_buffer_pop_clears_without_counting_next_put_as_drop() -> None:
    buffer = LatestFrameBuffer()
    first = _frame(1)
    second = _frame(2)

    buffer.put(first)

    assert buffer.pop_latest() is first
    assert buffer.get_latest() is None

    buffer.put(second)

    assert buffer.get_latest() is second
    assert buffer.dropped_frames == 0


def test_latest_frame_buffer_wait_pop_latest_returns_none_when_empty() -> None:
    buffer = LatestFrameBuffer()

    assert buffer.wait_pop_latest(timeout=0.0) is None


def test_latest_frame_buffer_clear_preserves_count_and_reset_clears_count() -> None:
    buffer = LatestFrameBuffer()

    buffer.put(_frame(1))
    buffer.put(_frame(2))
    buffer.clear()

    assert buffer.get_latest() is None
    assert buffer.dropped_frames == 1

    buffer.put(_frame(3))
    buffer.reset()

    assert buffer.get_latest() is None
    assert buffer.dropped_frames == 0


def test_latest_result_store_starts_empty() -> None:
    store = LatestResultStore()

    assert store.get_latest() is None
    assert store.replaced_results == 0


def test_latest_result_store_returns_stored_result() -> None:
    store = LatestResultStore()
    result = _result(1)

    store.put(result)

    assert store.get_latest() is result
    assert store.replaced_results == 0


def test_latest_result_store_replaces_previous_result() -> None:
    store = LatestResultStore()
    first = _result(1)
    second = _result(2)

    store.put(first)
    store.put(second)

    assert store.get_latest() is second
    assert store.replaced_results == 1


def test_latest_result_store_clear_preserves_count_and_reset_clears_count() -> None:
    store = LatestResultStore()

    store.put(_result(1))
    store.put(_result(2))
    store.clear()

    assert store.get_latest() is None
    assert store.replaced_results == 1

    store.put(_result(3))
    store.reset()

    assert store.get_latest() is None
    assert store.replaced_results == 0


def _frame(frame_id: int) -> FramePacket:
    return FramePacket(
        frame_id=frame_id,
        timestamp_ms=float(frame_id * 10),
        timestamp_ns=frame_id * 10_000_000,
        original_frame=np.zeros((2, 2, 3), dtype=np.uint8),
    )


def _result(frame_id: int) -> FrameResult:
    return FrameResult(
        frame_id=frame_id,
        timestamp_ms=float(frame_id * 10),
        timestamp_ns=frame_id * 10_000_000,
        detections=[],
        fps=30.0,
        inference_ms=4.0,
        total_frame_ms=8.0,
    )
