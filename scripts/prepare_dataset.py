from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from weld_inspector.dataset import build_yolo_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="整理焊缝缺陷 YOLOv8 数据集")
    parser.add_argument("--image-dir", required=True, help="原始图片目录")
    parser.add_argument("--label-dir", required=True, help="YOLO 标签目录")
    parser.add_argument("--output-dir", default="data/dataset", help="输出数据集目录")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="训练集比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--class-names",
        nargs="+",
        default=["crack", "hole", "unwelded", "offset_weld"],
        help="类别名称列表",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stats = build_yolo_dataset(
        image_dir=args.image_dir,
        label_dir=args.label_dir,
        output_dir=args.output_dir,
        class_names=args.class_names,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )

    generated_yaml = Path(args.output_dir) / "weld.yaml"
    target_yaml = PROJECT_ROOT / "configs" / "dataset" / "weld.yaml"
    target_yaml.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(generated_yaml, target_yaml)

    print("数据集整理完成。")
    for key, value in stats.to_dict().items():
        print(f"{key}: {value}")
    print(f"dataset_yaml: {generated_yaml}")
    print(f"config_yaml:  {target_yaml}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

