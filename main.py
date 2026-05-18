from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from edge_vision.app.runner import run_cli


def main() -> int:
    return run_cli(default_config_path=PROJECT_ROOT / "config.yaml")


if __name__ == "__main__":
    raise SystemExit(main())
