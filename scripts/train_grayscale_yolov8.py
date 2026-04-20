from __future__ import annotations

import sys
from pathlib import Path

from train_yolov8 import main as train_main


DEFAULT_CONFIG = "configs/app_grayscale.yaml"
DEFAULT_DATA_YAML = Path("data/datasets_grayscale/data.yaml")


def ensure_default_arg(flag: str, value: str) -> None:
    if flag not in sys.argv:
        sys.argv.extend([flag, value])


if __name__ == "__main__":
    had_config_arg = "--config" in sys.argv
    had_data_arg = "--data" in sys.argv
    requesting_help = any(flag in sys.argv for flag in ("-h", "--help"))
    ensure_default_arg("--config", DEFAULT_CONFIG)
    using_default_data = not had_config_arg and not had_data_arg
    if not requesting_help and using_default_data and not DEFAULT_DATA_YAML.exists():
        raise SystemExit(
            "默认灰度数据集未找到: data/datasets_grayscale/data.yaml\n"
            "请先执行: python scripts\\prepare_grayscale_dataset.py --dataset-dir data\\datasets"
        )
    raise SystemExit(train_main())
