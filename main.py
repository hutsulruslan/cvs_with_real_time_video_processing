from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from edge_vision.app.application_factory import create_application
from edge_vision.config.config_loader import load_config
from edge_vision.core.errors import EdgeVisionError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Edge Vision System")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config.yaml"),
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Only validate configuration and exit.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional frame limit for controlled manual runs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.max_frames is not None and args.max_frames < 0:
        print("Application error: --max-frames must be non-negative.", file=sys.stderr)
        return 1

    try:
        settings = load_config(args.config)
        if args.check_config:
            print(f"Configuration loaded for source: {settings.video.source_type}")
            return 0

        application = create_application(settings, max_frames=args.max_frames)
        processed_frames = application.run()
    except EdgeVisionError as error:
        print(f"Application error: {error}", file=sys.stderr)
        return 1

    print(f"Processed frames: {processed_frames}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
