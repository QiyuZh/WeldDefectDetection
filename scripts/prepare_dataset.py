from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from weld_inspector.config import ImagePreprocessSettings, load_app_config
from weld_inspector.dataset import build_yolo_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="整理焊缝缺陷 YOLOv8 数据集")
    parser.add_argument("--config", default=None, help="可选 app 配置文件，用于读取预处理默认参数")
    parser.add_argument("--image-dir", required=True, help="原始图片目录")
    parser.add_argument("--label-dir", required=True, help="YOLO 标签目录")
    parser.add_argument("--output-dir", default="data/dataset", help="输出数据集目录")
    parser.add_argument("--dataset-config", default="configs/dataset/weld.yaml", help="同步写入的 dataset yaml 路径")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="训练集比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--class-names",
        nargs="+",
        default=None,
        help="类别名称列表",
    )
    parser.add_argument("--preprocess-enabled", action="store_true", help="启用图像预处理后再生成数据集")
    parser.add_argument("--preprocess-mode", choices=["color", "grayscale_weld"], default=None, help="预处理模式")
    parser.add_argument("--clahe-clip-limit", type=float, default=None, help="CLAHE clip limit")
    parser.add_argument("--clahe-tile-grid-size", type=int, default=None, help="CLAHE tile grid size")
    parser.add_argument("--blur-kernel-size", type=int, default=None, help="锐化前高斯模糊核大小")
    parser.add_argument("--unsharp-amount", type=float, default=None, help="unsharp mask 强度")
    return parser.parse_args()


def choose(cli_value, config_value):
    return cli_value if cli_value is not None else config_value


def build_preprocess_settings(args: argparse.Namespace) -> ImagePreprocessSettings:
    config_settings = ImagePreprocessSettings()
    if args.config:
        config_settings = load_app_config(args.config).dataset_preprocess

    return ImagePreprocessSettings(
        enabled=args.preprocess_enabled or config_settings.enabled,
        mode=choose(args.preprocess_mode, config_settings.mode),
        clahe_clip_limit=choose(args.clahe_clip_limit, config_settings.clahe_clip_limit),
        clahe_tile_grid_size=choose(args.clahe_tile_grid_size, config_settings.clahe_tile_grid_size),
        blur_kernel_size=choose(args.blur_kernel_size, config_settings.blur_kernel_size),
        unsharp_amount=choose(args.unsharp_amount, config_settings.unsharp_amount),
    )


def resolve_class_names(args: argparse.Namespace) -> list[str]:
    if args.class_names:
        return list(args.class_names)

    if args.config:
        config = load_app_config(args.config)
        if config.model.class_names:
            return list(config.model.class_names)

    return ["crack", "hole", "unwelded", "offset_weld"]


def main() -> int:
    args = parse_args()
    preprocess_settings = build_preprocess_settings(args)
    stats = build_yolo_dataset(
        image_dir=args.image_dir,
        label_dir=args.label_dir,
        output_dir=args.output_dir,
        class_names=resolve_class_names(args),
        train_ratio=args.train_ratio,
        seed=args.seed,
        preprocess_settings=preprocess_settings,
    )

    generated_yaml = Path(args.output_dir) / "weld.yaml"
    target_yaml = PROJECT_ROOT / args.dataset_config
    target_yaml.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(generated_yaml, target_yaml)

    print("数据集整理完成。")
    for key, value in stats.to_dict().items():
        print(f"{key}: {value}")
    print(f"dataset_yaml: {generated_yaml}")
    print(f"config_yaml:  {target_yaml}")
    print(f"preprocess_enabled: {preprocess_settings.enabled}")
    print(f"preprocess_mode: {preprocess_settings.mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
