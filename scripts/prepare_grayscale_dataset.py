from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

from weld_inspector.config import ImagePreprocessSettings, load_app_config
from weld_inspector.dataset import build_yolo_dataset
from weld_inspector.preprocess import apply_image_preprocess
from weld_inspector.utils.io import ensure_dir, iter_image_files


DEFAULT_RAW_OUTPUT_DIR = "data/datasets_grayscale"
DEFAULT_DATASET_OUTPUT_DIR = "data/datasets_grayscale"
DEFAULT_CONFIG = "configs/app_grayscale.yaml"
DEFAULT_DATASET_CONFIG = "configs/dataset/weld_grayscale.yaml"


@dataclass(slots=True)
class PreparedDatasetStats:
    total_images: int = 0
    missing_labels: int = 0
    split_images: dict[str, int] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a grayscale-enhanced YOLO dataset.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="App config used to read default preprocess settings.")
    parser.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG, help="Target dataset yaml copied for training.")
    parser.add_argument("--dataset-dir", default=None, help="Structured YOLO dataset root, e.g. data/datasets.")
    parser.add_argument("--source-yaml", default=None, help="Source dataset yaml. Defaults to <dataset-dir>/data.yaml.")
    parser.add_argument("--image-dir", default=None, help="Raw image directory.")
    parser.add_argument("--label-dir", default=None, help="Raw YOLO label directory.")
    parser.add_argument("--output-dir", default=None, help="Output dataset directory.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train ratio for raw image mode.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for raw image mode.")
    parser.add_argument("--class-names", nargs="+", default=None, help="Optional class name override.")
    parser.add_argument("--preprocess-enabled", action="store_true", help="Enable grayscale preprocessing explicitly.")
    parser.add_argument("--preprocess-mode", choices=["color", "grayscale_weld"], default="grayscale_weld", help="Preprocess mode.")
    parser.add_argument("--clahe-clip-limit", type=float, default=None, help="CLAHE clip limit.")
    parser.add_argument("--clahe-tile-grid-size", type=int, default=None, help="CLAHE tile grid size.")
    parser.add_argument("--blur-kernel-size", type=int, default=None, help="Unsharp blur kernel size.")
    parser.add_argument("--unsharp-amount", type=float, default=None, help="Unsharp amount.")
    return parser.parse_args()


def choose(cli_value, config_value):
    return cli_value if cli_value is not None else config_value


def resolve_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def path_for_yaml(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def ensure_yaml_available() -> None:
    if yaml is None:  # pragma: no cover
        raise RuntimeError("PyYAML is required. Install it with `pip install pyyaml`.")


def normalize_class_names(raw_names: Any) -> list[str] | None:
    if isinstance(raw_names, dict):
        try:
            keys = sorted(raw_names, key=lambda item: int(item))
        except (TypeError, ValueError):
            keys = sorted(raw_names)
        return [str(raw_names[key]) for key in keys]

    if isinstance(raw_names, (list, tuple)):
        return [str(name) for name in raw_names]

    return None


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


def resolve_class_names(
    args: argparse.Namespace,
    source_dataset_yaml: dict[str, Any] | None = None,
) -> list[str]:
    if args.class_names:
        return list(args.class_names)

    if source_dataset_yaml:
        source_names = normalize_class_names(source_dataset_yaml.get("names"))
        if source_names:
            return source_names

    if args.config:
        config = load_app_config(args.config)
        if config.model.class_names:
            return list(config.model.class_names)

    return ["crack", "hole", "unwelded", "offset_weld"]


def validate_mode(args: argparse.Namespace) -> str:
    if args.dataset_dir:
        if args.image_dir or args.label_dir:
            raise ValueError("Use either --dataset-dir or --image-dir/--label-dir, not both.")
        return "structured"

    if args.image_dir and args.label_dir:
        return "raw"

    raise ValueError("Provide --dataset-dir for a structured YOLO dataset, or both --image-dir and --label-dir for raw mode.")


def infer_output_dir(args: argparse.Namespace, mode: str) -> Path:
    if args.output_dir:
        return resolve_path(args.output_dir)
    default_dir = DEFAULT_DATASET_OUTPUT_DIR if mode == "structured" else DEFAULT_RAW_OUTPUT_DIR
    return resolve_path(default_dir)


def infer_split_root(source_dataset_yaml: Path, dataset_root: Path, dataset_meta: dict[str, Any]) -> Path:
    declared_root = dataset_meta.get("path")
    if not declared_root:
        return dataset_root

    root_path = Path(str(declared_root))
    if root_path.is_absolute():
        return root_path
    return (source_dataset_yaml.parent / root_path).resolve()


def infer_label_relative_dir(image_relative_dir: str | Path) -> Path:
    image_relative_path = Path(str(image_relative_dir))
    if image_relative_path.name == "images":
        return image_relative_path.with_name("labels")
    return image_relative_path.parent / "labels"


def copy_or_preprocess_image(source: Path, destination: Path, preprocess_settings: ImagePreprocessSettings) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if preprocess_settings.is_active:
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read image for preprocessing: {source}")
        processed = apply_image_preprocess(image, preprocess_settings)
        cv2.imwrite(str(destination), processed)
        return

    shutil.copy2(source, destination)


def build_structured_dataset_yaml(
    output_root: Path,
    target_path: Path,
    split_entries: dict[str, str],
    class_names: list[str],
    source_meta: dict[str, Any],
) -> Path:
    payload: dict[str, Any] = {"path": path_for_yaml(output_root)}
    payload.update(split_entries)
    payload["nc"] = len(class_names)
    payload["names"] = class_names

    for key, value in source_meta.items():
        if key not in {"path", "train", "val", "test", "nc", "names"}:
            payload[key] = value

    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return target_path


def prepare_structured_dataset(
    dataset_dir: str | Path,
    output_dir: str | Path,
    dataset_config: str | Path,
    preprocess_settings: ImagePreprocessSettings,
    class_names: list[str] | None = None,
    source_yaml: str | Path | None = None,
) -> tuple[PreparedDatasetStats, Path]:
    ensure_yaml_available()
    dataset_root = resolve_path(dataset_dir)
    output_root = resolve_path(output_dir)
    source_yaml_path = resolve_path(source_yaml) if source_yaml else dataset_root / "data.yaml"

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    if not source_yaml_path.exists():
        raise FileNotFoundError(f"Source dataset yaml not found: {source_yaml_path}")

    dataset_meta = yaml.safe_load(source_yaml_path.read_text(encoding="utf-8")) or {}
    split_root = infer_split_root(source_yaml_path, dataset_root, dataset_meta)
    resolved_class_names = class_names or normalize_class_names(dataset_meta.get("names")) or []
    if not resolved_class_names:
        raise ValueError("Failed to resolve class names from args, config, or source dataset yaml.")

    output_root.mkdir(parents=True, exist_ok=True)
    stats = PreparedDatasetStats()
    split_entries: dict[str, str] = {}

    for split_key in ("train", "val", "test"):
        declared_image_dir = dataset_meta.get(split_key)
        if not declared_image_dir:
            continue

        image_relative_dir = Path(str(declared_image_dir))
        label_relative_dir = infer_label_relative_dir(image_relative_dir)
        source_image_dir = split_root / image_relative_dir
        source_label_dir = split_root / label_relative_dir
        output_image_dir = output_root / image_relative_dir
        output_label_dir = output_root / label_relative_dir

        if not source_image_dir.exists():
            raise FileNotFoundError(f"Split image directory not found: {source_image_dir}")

        image_count = 0
        for image_path in iter_image_files(source_image_dir):
            relative_path = image_path.relative_to(source_image_dir)
            destination_image = output_image_dir / relative_path
            copy_or_preprocess_image(image_path, destination_image, preprocess_settings)

            source_label = source_label_dir / relative_path.with_suffix(".txt")
            destination_label = output_label_dir / relative_path.with_suffix(".txt")
            destination_label.parent.mkdir(parents=True, exist_ok=True)
            if source_label.exists():
                shutil.copy2(source_label, destination_label)
            else:
                destination_label.write_text("", encoding="utf-8")
                stats.missing_labels += 1

            image_count += 1

        split_entries[split_key] = image_relative_dir.as_posix()
        stats.split_images[split_key] = image_count
        stats.total_images += image_count

    if not split_entries:
        raise ValueError("No train/val/test entries were found in the source dataset yaml.")

    generated_yaml = build_structured_dataset_yaml(
        output_root=output_root,
        target_path=output_root / "data.yaml",
        split_entries=split_entries,
        class_names=resolved_class_names,
        source_meta=dataset_meta,
    )

    target_yaml = resolve_path(dataset_config)
    target_yaml.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(generated_yaml, target_yaml)
    return stats, generated_yaml


def main() -> int:
    args = parse_args()
    mode = validate_mode(args)
    output_dir = infer_output_dir(args, mode)
    preprocess_settings = build_preprocess_settings(args)

    if mode == "structured":
        ensure_yaml_available()
        source_yaml_path = resolve_path(args.source_yaml) if args.source_yaml else resolve_path(args.dataset_dir) / "data.yaml"
        source_meta = yaml.safe_load(source_yaml_path.read_text(encoding="utf-8")) if source_yaml_path.exists() else {}
        stats, generated_yaml = prepare_structured_dataset(
            dataset_dir=args.dataset_dir,
            output_dir=output_dir,
            dataset_config=args.dataset_config,
            preprocess_settings=preprocess_settings,
            class_names=resolve_class_names(args, source_meta),
            source_yaml=args.source_yaml,
        )
        print("grayscale dataset prepared from structured YOLO dataset")
        print(f"total_images: {stats.total_images}")
        for split_key, count in stats.split_images.items():
            print(f"{split_key}_images: {count}")
        print(f"missing_labels: {stats.missing_labels}")
        print(f"dataset_yaml: {generated_yaml}")
        print(f"config_yaml:  {resolve_path(args.dataset_config)}")
        print(f"preprocess_enabled: {preprocess_settings.enabled}")
        print(f"preprocess_mode: {preprocess_settings.mode}")
        return 0

    stats = build_yolo_dataset(
        image_dir=resolve_path(args.image_dir),
        label_dir=resolve_path(args.label_dir),
        output_dir=output_dir,
        class_names=resolve_class_names(args),
        train_ratio=args.train_ratio,
        seed=args.seed,
        preprocess_settings=preprocess_settings,
    )
    generated_yaml = output_dir / "weld.yaml"
    output_data_yaml = output_dir / "data.yaml"
    target_yaml = resolve_path(args.dataset_config)
    target_yaml.parent.mkdir(parents=True, exist_ok=True)
    if generated_yaml.resolve() != output_data_yaml.resolve():
        shutil.copy2(generated_yaml, output_data_yaml)
    source_for_target = output_data_yaml if output_data_yaml.exists() else generated_yaml
    if source_for_target.resolve() != target_yaml.resolve():
        shutil.copy2(source_for_target, target_yaml)

    print("grayscale dataset prepared from raw image directory")
    for key, value in stats.to_dict().items():
        print(f"{key}: {value}")
    print(f"dataset_yaml: {output_data_yaml}")
    print(f"config_yaml:  {target_yaml}")
    print(f"preprocess_enabled: {preprocess_settings.enabled}")
    print(f"preprocess_mode: {preprocess_settings.mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
