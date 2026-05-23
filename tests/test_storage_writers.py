from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from edge_vision.core.detection import Detection
from edge_vision.core.result import FrameResult
from edge_vision.storage.csv_writer import CSVResultWriter
from edge_vision.storage.json_writer import JSONResultWriter


def test_csv_writer_creates_file_with_header_and_detection_rows(tmp_path: Path) -> None:
    output_path = tmp_path / "detections" / "results.csv"
    writer = CSVResultWriter(output_path)

    writer.write(_result_with_detection())
    writer.close()

    rows = _read_csv_rows(output_path)
    assert output_path.exists()
    assert rows[0]["frame_id"] == "1"
    assert rows[0]["timestamp_ms"] == "123.4"
    assert rows[0]["timestamp_ns"] == "123400000"
    assert rows[0]["source_frame_id"] == "1"
    assert rows[0]["source_timestamp_ms"] == "123.4"
    assert rows[0]["source_timestamp_ns"] == "123400000"
    assert rows[0]["completed_timestamp_ns"] == "163400000"
    assert rows[0]["result_age_ms"] == "40.0"
    assert rows[0]["end_to_end_latency_ms"] == "40.0"
    assert rows[0]["inference_ran"] == "True"
    assert rows[0]["class_id"] == "0"
    assert rows[0]["class_name"] == "person"
    assert rows[0]["confidence"] == "0.85"
    assert rows[0]["x_min"] == "10"
    assert rows[0]["y_max"] == "160"


def test_csv_writer_writes_frame_row_without_detections(tmp_path: Path) -> None:
    output_path = tmp_path / "results.csv"
    writer = CSVResultWriter(output_path)

    writer.write(_result_without_detections())
    writer.close()

    rows = _read_csv_rows(output_path)
    assert rows[0]["frame_id"] == "2"
    assert rows[0]["fps"] == "30.0"
    assert rows[0]["class_id"] == ""
    assert rows[0]["class_name"] == ""


def test_csv_writer_close_is_safe_to_call_multiple_times(tmp_path: Path) -> None:
    writer = CSVResultWriter(tmp_path / "results.csv")

    writer.close()
    writer.close()


def test_json_writer_creates_structured_result_file(tmp_path: Path) -> None:
    output_path = tmp_path / "detections" / "results.json"
    writer = JSONResultWriter(output_path)

    writer.write(_result_with_detection())
    writer.close()

    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert output_path.exists()
    assert data[0]["frame_id"] == 1
    assert data[0]["timestamp_ms"] == 123.4
    assert data[0]["timestamp_ns"] == 123400000
    assert data[0]["source_frame_id"] == 1
    assert data[0]["source_timestamp_ms"] == 123.4
    assert data[0]["source_timestamp_ns"] == 123400000
    assert data[0]["completed_timestamp_ns"] == 163400000
    assert data[0]["result_age_ms"] == 40.0
    assert data[0]["end_to_end_latency_ms"] == 40.0
    assert data[0]["inference_ran"] is True
    assert data[0]["fps"] == 25.0
    assert data[0]["inference_ms"] == 12.3
    assert data[0]["total_frame_ms"] == 40.0
    assert data[0]["detections"][0]["class_id"] == 0
    assert data[0]["detections"][0]["class_name"] == "person"
    assert data[0]["detections"][0]["confidence"] == 0.85


def test_json_writer_supports_multiple_frame_results(tmp_path: Path) -> None:
    output_path = tmp_path / "results.json"
    writer = JSONResultWriter(output_path)

    writer.write(_result_with_detection())
    writer.write(_result_without_detections())
    writer.close()

    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert [item["frame_id"] for item in data] == [1, 2]
    assert data[1]["detections"] == []


def test_json_writer_close_is_safe_to_call_multiple_times(tmp_path: Path) -> None:
    writer = JSONResultWriter(tmp_path / "results.json")

    writer.close()
    writer.close()


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def _result_with_detection() -> FrameResult:
    return FrameResult(
        frame_id=1,
        timestamp_ms=123.4,
        detections=[
            Detection(
                class_id=0,
                class_name="person",
                confidence=0.85,
                x_min=10,
                y_min=20,
                x_max=100,
                y_max=160,
            )
        ],
        fps=25.0,
        inference_ms=12.3,
        total_frame_ms=40.0,
    )


def _result_without_detections() -> FrameResult:
    return FrameResult(
        frame_id=2,
        timestamp_ms=200.0,
        detections=[],
        fps=30.0,
        inference_ms=8.0,
        total_frame_ms=33.0,
    )
