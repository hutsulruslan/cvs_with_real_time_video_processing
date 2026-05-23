from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from edge_vision.core.frame import FramePacket
from edge_vision.core.result import FrameResult


def test_frame_packet_keeps_timestamp_ms_and_derives_timestamp_ns() -> None:
    packet = FramePacket(
        frame_id=7,
        timestamp_ms=123.456,
        original_frame=np.zeros((2, 2, 3), dtype=np.uint8),
    )

    assert packet.frame_id == 7
    assert packet.timestamp_ms == 123.456
    assert packet.timestamp_ns == 123_456_000


def test_frame_result_derives_source_and_latency_metadata() -> None:
    result = FrameResult(
        frame_id=3,
        timestamp_ms=30.0,
        detections=[],
        fps=10.0,
        inference_ms=12.0,
        total_frame_ms=40.0,
    )

    assert result.source_frame_id == 3
    assert result.source_timestamp_ms == 30.0
    assert result.source_timestamp_ns == 30_000_000
    assert result.completed_timestamp_ns == 70_000_000
    assert result.result_age_ms == 40.0
    assert result.end_to_end_latency_ms == 40.0
    assert result.inference_ran is True


def test_frame_result_keeps_explicit_source_metadata_for_reused_detections() -> None:
    result = FrameResult(
        frame_id=5,
        timestamp_ms=90.0,
        detections=[],
        fps=30.0,
        inference_ms=0.0,
        total_frame_ms=2.0,
        timestamp_ns=90_000_000,
        source_frame_id=2,
        source_timestamp_ms=40.0,
        source_timestamp_ns=40_000_000,
        completed_timestamp_ns=95_000_000,
        inference_ran=False,
    )

    assert result.source_frame_id == 2
    assert result.result_age_ms == 55.0
    assert result.end_to_end_latency_ms == 5.0
    assert result.inference_ran is False
