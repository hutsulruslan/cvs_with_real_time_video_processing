# Edge Vision System

Local edge computer vision system for video stream processing on a Linux-based
peripheral device such as Raspberry Pi or a similar single-board computer.

The project is designed as a bachelor thesis implementation. The codebase aims
to stay small, modular, testable, and easy to explain in documentation.

## Current Status

Phase 1 is implemented:

- typed configuration dataclasses;
- YAML configuration loading;
- basic configuration validation;
- core pipeline dataclasses;
- custom application exceptions;
- pytest tests for configuration loading.

Phase 2 is implemented:

- base video source interface;
- OpenCV camera source;
- OpenCV video file source;
- video source factory;
- tests using fake capture objects.

Phase 3 is implemented:

- frame resizing for model input size;
- BGR to RGB color conversion;
- batched tensor preparation;
- optional float32 normalization;
- preprocessing tests with NumPy frames.

Phase 4A is implemented:

- object detector interface;
- deterministic mock detector;
- inference tests that do not require a model file.

Pipeline foundation is implemented:

- minimal single-frame pipeline coordinator with postprocessing and metrics;
- pipeline tests with fake video source and mock detector;
- no real model, display, or storage integration yet.

Postprocessing foundation is implemented:

- confidence threshold filtering;
- max detections limiting by confidence;
- coordinate scaling back to original frame size;
- focused tests for postprocessing logic.

Metrics foundation is implemented:

- FPS counter with injectable time provider;
- lightweight named-section profiler;
- deterministic tests without real sleeps.

Renderer foundation is implemented:

- detection bounding box drawing;
- class label and confidence drawing;
- optional FPS overlay;
- renderer tests without display windows.

Window display foundation is implemented:

- OpenCV window display wrapper;
- configurable quit key;
- tests with fake OpenCV object and no real GUI window.

Minimal visual application wrapper is implemented:

- controlled finite run loop;
- dependency-injected source, pipeline, renderer, and display;
- tests with fake source/display and no real camera or GUI window.

Runnable mock visual mode is implemented:

- `main.py` builds the application from `config.yaml`;
- `model.runtime: "mock"` uses deterministic mock detections;
- the OpenCV window can be closed with `q`.

No model files are included in the repository yet.

TFLite detector foundation is implemented:

- lazy TFLite runtime loading;
- `labels.txt` loading;
- SSD/EfficientDet-style output conversion to `Detection` objects;
- model tensor inspection helper;
- tests with fake interpreter, without a real model file.

## Environment

Recommended:

- Python 3.10+
- a virtual environment

Install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Configuration

Runtime settings are stored in `config.yaml`.

The first version keeps all important values configurable, including video
source type, model paths, input size, thresholds, display settings, and optional
storage settings.

## Run

From the project root:

```powershell
python main.py
```

Or use a custom configuration file:

```powershell
python main.py --config config.yaml
```

To validate configuration without opening a camera or window:

```powershell
python main.py --check-config
```

The current runnable mode uses `MockObjectDetector`. It is useful for checking
the video, preprocessing, postprocessing, metrics, renderer, and display path
before the real TFLite detector is connected.

## TFLite Model Files

The project runs in mock mode by default. Real TFLite mode is optional and
requires local model files that are not committed to git.

Expected local paths:

```text
assets/models/model.tflite
assets/models/labels.txt
```

Model binaries are ignored by `.gitignore` to avoid accidental commits.

To enable TFLite mode later, place a compatible model and labels file under the
paths above, install a TFLite/LiteRT-compatible runtime, then change:

```yaml
model:
  runtime: "tflite"
  model_path: "assets/models/model.tflite"
  labels_path: "assets/models/labels.txt"
```

Validate configuration without opening a camera:

```powershell
python main.py --check-config
```

Inspect model tensors before running the visual application:

```powershell
$env:PYTHONPATH = "src"
python -m edge_vision.inference.model_inspector assets/models/model.tflite
```

The current detector supports only SSD/EfficientDet-style TFLite object
detection outputs:

- boxes;
- classes;
- scores;
- num_detections.

The expected box format is normalized `[y_min, x_min, y_max, x_max]`.
YOLO-style outputs are not supported by the current detector.

Phone camera testing can be done later by exposing the phone as a virtual camera
or by adding a dedicated stream source. The Phase 2 camera source already supports
virtual camera devices available through OpenCV camera indexes.

## Optional Result Saving

Detection result saving is optional and disabled by default. It is intended for
local experiments and thesis demos where frame-level metrics and detections need
to be reviewed after a run.

To save CSV results, set:

```yaml
storage:
  save_detections: true
  format: "csv"
  output_dir: "output"
```

The CSV file is written to:

```text
output/detections/results.csv
```

To save JSON results, set:

```yaml
storage:
  save_detections: true
  format: "json"
  output_dir: "output"
```

The JSON file is written to:

```text
output/detections/results.json
```

Saved result data includes `frame_id`, `timestamp_ms`, `fps`, `inference_ms`,
`total_frame_ms`, and detections with `class_id`, `class_name`, `confidence`,
`x_min`, `y_min`, `x_max`, and `y_max`.

The `output/` directory is local runtime output and should not be committed.
After experiments, restore storage to the usual default:

```yaml
storage:
  save_detections: false
```

## Tests

```powershell
python -m pytest
```

## Planned Pipeline

```text
VideoSource
  -> FramePreprocessor
  -> ObjectDetector
  -> DetectionPostProcessor
  -> Renderer
  -> Display / Storage / Metrics
```
