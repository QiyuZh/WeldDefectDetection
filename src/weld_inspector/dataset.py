from __future__ import annotations

import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2

from .config import ImagePreprocessSettings
from .preprocess import apply_image_preprocess
from .utils.io import ensure_dir, iter_image_files, write_text


@dataclass(slots=True)
class DatasetBuildStats:
    total_images: int
    train_images: int
    val_images: int
    negative_images: int
    missing_labels: int

    def to_dict(self) -> dict[str, int]:
        return {
            "total_images": self.total_images,
            "train_images": self.train_images,
            "val_images": self.val_images,
            "negative_images": self.negative_images,
            "missing_labels": self.missing_labels,
        }


def label_path_for_image(image_path: Path, label_dir: Path) -> Path:
    return label_dir / f"{image_path.stem}.txt"


def split_items(items: list[Path], train_ratio: float, seed: int) -> tuple[list[Path], list[Path]]:
    if not items:
        raise ValueError("未发现任何图片，无法划分数据集。")
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio 必须位于 (0, 1) 区间。")
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    split_index = max(1, min(len(shuffled) - 1, int(len(shuffled) * train_ratio)))
    return shuffled[:split_index], shuffled[split_index:]


def build_yolo_dataset(
    image_dir: str | Path,
    label_dir: str | Path,
    output_dir: str | Path,
    class_names: list[str],
    train_ratio: float = 0.8,
    seed: int = 42,
    preprocess_settings: ImagePreprocessSettings | None = None,
) -> DatasetBuildStats:
    image_root = Path(image_dir)
    label_root = Path(label_dir)
    output_root = Path(output_dir)
    images = sorted(iter_image_files(image_root))
    train_items, val_items = split_items(images, train_ratio=train_ratio, seed=seed)

    directories = [
        output_root / "images" / "train",
        output_root / "images" / "val",
        output_root / "labels" / "train",
        output_root / "labels" / "val",
    ]
    for directory in directories:
        ensure_dir(directory)

    negative_images = 0
    missing_labels = 0
    active_preprocess = preprocess_settings or ImagePreprocessSettings()
    for split_name, split_items_list in (("train", train_items), ("val", val_items)):
        for image_path in split_items_list:
            destination_image = output_root / "images" / split_name / image_path.name
            if active_preprocess.is_active:
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f"Failed to read image for preprocessing: {image_path}")
                processed = apply_image_preprocess(image, active_preprocess)
                cv2.imwrite(str(destination_image), processed)
            else:
                shutil.copy2(image_path, destination_image)

            source_label = label_path_for_image(image_path, label_root)
            destination_label = output_root / "labels" / split_name / f"{image_path.stem}.txt"
            if source_label.exists():
                shutil.copy2(source_label, destination_label)
            else:
                negative_images += 1
                missing_labels += 1
                destination_label.write_text("", encoding="utf-8")

    names_block = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(class_names))
    dataset_yaml = (
        f"path: {output_root.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n{names_block}\n"
    )
    write_text(output_root / "weld.yaml", dataset_yaml)

    return DatasetBuildStats(
        total_images=len(images),
        train_images=len(train_items),
        val_images=len(val_items),
        negative_images=negative_images,
        missing_labels=missing_labels,
    )
