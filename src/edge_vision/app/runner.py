from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from edge_vision.app.application_factory import create_application
from edge_vision.app.run_overrides import (
    PROFILE_SETTINGS,
    RunOverrides,
    apply_run_overrides,
    validate_override_values,
    validate_run_overrides,
)
from edge_vision.config.config_loader import load_config
from edge_vision.core.errors import EdgeVisionError
from edge_vision.metrics.performance_report import (
    PerformanceReportBuilder,
    format_performance_report,
)


def run_cli(
    argv: Sequence[str] | None = None,
    *,
    default_config_path: str | Path = "config.yaml",
) -> int:
    """Run the command-line entrypoint for the edge vision application."""
    parser = build_arg_parser(default_config_path)
    args = parser.parse_args(argv)
    overrides = _overrides_from_args(args)

    try:
        validate_override_values(overrides)
        settings = apply_run_overrides(load_config(args.config), overrides)
        if args.check_config:
            print(f"Configuration loaded for source: {settings.video.source_type}")
            return 0

        validate_run_overrides(settings, overrides)
        report_builder = PerformanceReportBuilder() if args.report else None
        application = create_application(
            settings,
            max_frames=overrides.max_frames,
            no_display=overrides.no_display,
            result_callback=None if report_builder is None else report_builder.add_result,
        )
        processed_frames = application.run()
    except EdgeVisionError as error:
        print(f"Application error: {error}", file=sys.stderr)
        return 1

    print(f"Processed frames: {processed_frames}")
    if report_builder is not None:
        print(format_performance_report(report_builder.build()))
    return 0


def build_arg_parser(default_config_path: str | Path) -> argparse.ArgumentParser:
    """Create the small runtime CLI parser."""
    parser = argparse.ArgumentParser(description="Edge Vision System")
    parser.add_argument(
        "--config",
        default=str(default_config_path),
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Only validate configuration and exit.",
    )
    parser.add_argument(
        "--profile",
        choices=tuple(PROFILE_SETTINGS),
        help="Runtime profile, for example mock-file or tflite-stream.",
    )
    parser.add_argument("--camera-index", type=int, help="Override video.camera_index.")
    parser.add_argument("--file-path", help="Override video.file_path.")
    parser.add_argument("--stream-url", help="Override video.stream_url.")
    parser.add_argument("--max-frames", type=int, help="Stop after N processed frames.")
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Process frames without opening an OpenCV window.",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Print a performance summary after the run.",
    )
    return parser


def _overrides_from_args(args: argparse.Namespace) -> RunOverrides:
    return RunOverrides(
        profile=args.profile,
        camera_index=args.camera_index,
        file_path=args.file_path,
        stream_url=args.stream_url,
        max_frames=args.max_frames,
        no_display=args.no_display,
    )
