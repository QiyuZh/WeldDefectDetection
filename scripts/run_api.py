from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from weld_inspector.bootstrap import bootstrap_windows_runtime

bootstrap_windows_runtime()

import uvicorn

from weld_inspector.api import create_app
from weld_inspector.config import load_app_config


def main() -> int:
    parser = argparse.ArgumentParser(description="启动焊缝质检 HTTP 服务")
    parser.add_argument("--config", default="configs/app.yaml", help="配置文件路径")
    args = parser.parse_args()

    config = load_app_config(args.config)
    app = create_app(args.config)
    uvicorn.run(app, host=config.api.host, port=config.api.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
