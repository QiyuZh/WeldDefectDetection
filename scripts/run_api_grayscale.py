from __future__ import annotations

import sys

from run_api import main as run_api_main


def ensure_default_arg(flag: str, value: str) -> None:
    if flag not in sys.argv:
        sys.argv.extend([flag, value])


if __name__ == "__main__":
    ensure_default_arg("--config", "configs/app_grayscale.yaml")
    raise SystemExit(run_api_main())
