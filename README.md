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

The video processing pipeline is intentionally not implemented yet.

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
