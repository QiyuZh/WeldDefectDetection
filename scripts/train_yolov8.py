from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from weld_inspector.config import load_app_config, resolve_project_path


def parse_bool_or_str(value: str | None) -> bool | str | None:
    if value is None:
        return None

    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    if normalized in {"ram", "disk"}:
        return normalized
    return value


def get_config_value(obj: Any, name: str, default: Any = None) -> Any:
    return getattr(obj, name, default)


def choose(cli_value: Any, config_value: Any) -> Any:
    return cli_value if cli_value is not None else config_value


def remove_none_values(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练焊缝缺陷 YOLOv8 模型")
    parser.add_argument("--config", default="configs/app.yaml", help="配置文件路径")
    parser.add_argument("--weights", default=None, help="预训练权重")
    parser.add_argument("--data", default=None, help="数据集 yaml")
    parser.add_argument("--epochs", type=int, default=None, help="训练轮数")
    parser.add_argument("--imgsz", type=int, default=None, help="输入尺寸")
    parser.add_argument("--batch", type=int, default=None, help="batch size")
    parser.add_argument("--device", default=None, help="训练设备，如 0 / cpu")
    parser.add_argument("--name", default=None, help="训练任务名称")

    # 新增：可覆盖 YAML 的训练参数
    parser.add_argument("--workers", type=int, default=None, help="数据加载线程数")
    parser.add_argument("--patience", type=int, default=None, help="早停轮数")
    parser.add_argument("--close-mosaic", dest="close_mosaic", type=int, default=None, help="最后 N 轮关闭 mosaic")
    parser.add_argument(
        "--cache",
        default=None,
        help="数据缓存方式：True / False / ram / disk",
    )
    parser.add_argument("--lr0", type=float, default=None, help="初始学习率")
    parser.add_argument("--lrf", type=float, default=None, help="最终学习率比例")
    parser.add_argument(
        "--cos-lr",
        dest="cos_lr",
        action="store_true",
        help="启用 cosine learning rate",
    )
    parser.add_argument(
        "--no-cos-lr",
        dest="cos_lr",
        action="store_false",
        help="关闭 cosine learning rate",
    )
    parser.set_defaults(cos_lr=None)
    parser.add_argument("--label-smoothing", dest="label_smoothing", type=float, default=None, help="标签平滑")
    parser.add_argument("--dropout", type=float, default=None, help="dropout 比例")

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_app_config(args.config)

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("未安装 ultralytics，请先 `pip install -r requirements/train.txt`。") from exc

    training_cfg = config.training

    weights = choose(args.weights, training_cfg.weights)
    data_yaml = resolve_project_path(choose(args.data, training_cfg.data_yaml))
    project_dir = resolve_project_path(training_cfg.project)
    run_name = choose(args.name, training_cfg.name)
    project_dir.mkdir(parents=True, exist_ok=True)

    if not data_yaml.exists():
        raise FileNotFoundError(f"训练数据配置不存在: {data_yaml}")

    cache_cli_value = parse_bool_or_str(args.cache)
    cache_value = choose(cache_cli_value, get_config_value(training_cfg, "cache", False))

    # 允许在 YAML 的 training 下透传 Ultralytics 支持的附加训练参数，
    # 同时确保脚本显式管理的核心参数仍然由配置或 CLI 覆盖。
    train_kwargs = dict(get_config_value(training_cfg, "extra_args", {}))
    train_kwargs.update(
        remove_none_values(
            {
        "data": str(data_yaml),
        "epochs": choose(args.epochs, training_cfg.epochs),
        "imgsz": choose(args.imgsz, training_cfg.imgsz),
        "batch": choose(args.batch, training_cfg.batch),
        "workers": choose(args.workers, training_cfg.workers),
        "patience": choose(args.patience, training_cfg.patience),
        "project": str(project_dir),
        "name": run_name,
        "device": choose(args.device, training_cfg.device),
        "exist_ok": True,
        "close_mosaic": choose(args.close_mosaic, get_config_value(training_cfg, "close_mosaic", 10)),
        "cache": cache_value,
        "lr0": choose(args.lr0, get_config_value(training_cfg, "lr0", 0.01)),
        "lrf": choose(args.lrf, get_config_value(training_cfg, "lrf", 0.01)),
        "cos_lr": choose(args.cos_lr, get_config_value(training_cfg, "cos_lr", False)),
        "label_smoothing": choose(args.label_smoothing, get_config_value(training_cfg, "label_smoothing", 0.0)),
        "dropout": choose(args.dropout, get_config_value(training_cfg, "dropout", 0.0)),
            }
        )
    )

    model = YOLO(weights)
    model.train(**train_kwargs)

    weights_dir = project_dir / run_name / "weights"
    model_dir = project_dir
    model_dir.mkdir(parents=True, exist_ok=True)

    for filename in ("best.pt", "last.pt"):
        source = weights_dir / filename
        if source.exists():
            target = model_dir / filename
            shutil.copy2(source, target)
            print(f"copied: {source} -> {target}")

    print(f"训练完成，权重目录: {weights_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
