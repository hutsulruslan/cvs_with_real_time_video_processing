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

The full video processing pipeline is intentionally not implemented yet.

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

At this stage the command only validates and loads configuration. Later phases
will connect the video source, preprocessing, inference, visualization, metrics,
and optional storage modules.

Phone camera testing can be done later by exposing the phone as a virtual camera
or by adding a dedicated stream source. The Phase 2 camera source already supports
virtual camera devices available through OpenCV camera indexes.

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
