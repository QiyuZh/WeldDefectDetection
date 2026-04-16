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

# Windows + conda 环境下，先导入 cv2 再导入 PySide6 更稳定，
# 否则可能触发 Qt/OpenCV 相关 DLL 搜索顺序问题。
import cv2  # noqa: F401
from PySide6.QtWidgets import QApplication

from weld_inspector.ui.main_window import MainWindow


def main() -> int:
    parser = argparse.ArgumentParser(description="启动 YOLOv8 焊缝质检桌面端")
    parser.add_argument("--config", default="configs/app.yaml", help="配置文件路径")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    window = MainWindow(config_path=args.config)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
