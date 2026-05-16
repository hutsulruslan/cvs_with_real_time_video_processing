from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from edge_vision.config.config_loader import load_config
from edge_vision.core.errors import ConfigurationError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Edge Vision System")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config.yaml"),
        help="Path to YAML configuration file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        settings = load_config(args.config)
    except ConfigurationError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 1

    print(f"Configuration loaded for source: {settings.video.source_type}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
