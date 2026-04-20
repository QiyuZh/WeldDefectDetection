from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

from weld_inspector.config import ImagePreprocessSettings, load_app_config
from weld_inspector.preprocess import apply_image_preprocess
from weld_inspector.utils.io import ensure_dir, iter_image_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch export side-by-side original vs processed comparison images.")
    parser.add_argument("--config", default=None, help="Optional app config used to read dataset_preprocess defaults.")
    parser.add_argument("--image-dir", default=None, help="Single image directory to compare.")
    parser.add_argument("--dataset-dir", default=None, help="YOLO dataset root, for example data/datasets.")
    parser.add_argument("--source-yaml", default=None, help="Dataset yaml path. Defaults to <dataset-dir>/data.yaml.")
    parser.add_argument("--output-dir", default="artifacts/preprocess_compare", help="Comparison image output directory.")
    parser.add_argument("--limit", type=int, default=0, help="Limit images per input directory. 0 means all.")
    parser.add_argument("--preprocess-enabled", action="store_true", help="Explicitly enable preprocessing.")
    parser.add_argument("--preprocess-mode", choices=["color", "grayscale_weld"], default=None, help="Preprocess mode.")
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


def ensure_yaml_available() -> None:
    if yaml is None:  # pragma: no cover
        raise RuntimeError("PyYAML is required. Install it with `pip install pyyaml`.")


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


def draw_panel_label(image: np.ndarray, text: str) -> np.ndarray:
    labeled = image.copy()
    cv2.rectangle(labeled, (0, 0), (220, 44), (20, 20, 20), thickness=-1)
    cv2.putText(
        labeled,
        text,
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return labeled


def build_comparison_canvas(original: np.ndarray, processed: np.ndarray) -> np.ndarray:
    original_panel = draw_panel_label(original, "Original")
    processed_panel = draw_panel_label(processed, "Processed")
    separator = np.full((original.shape[0], 18, 3), 18, dtype=np.uint8)
    return np.hstack([original_panel, separator, processed_panel])


def iter_limited_images(image_dir: Path, limit: int) -> list[Path]:
    images = list(iter_image_files(image_dir))
    if limit > 0:
        return images[:limit]
    return images


def infer_dataset_root(source_yaml: Path, dataset_dir: Path, dataset_meta: dict[str, Any]) -> Path:
    declared_root = dataset_meta.get("path")
    if not declared_root:
        return dataset_dir

    root_path = Path(str(declared_root))
    if root_path.is_absolute():
        return root_path
    return (source_yaml.parent / root_path).resolve()


def resolve_image_dirs(args: argparse.Namespace) -> list[tuple[str, Path]]:
    if args.dataset_dir and args.image_dir:
        raise ValueError("Use either --dataset-dir or --image-dir, not both.")

    if args.image_dir:
        return [("images", resolve_path(args.image_dir))]

    if args.dataset_dir:
        ensure_yaml_available()
        dataset_dir = resolve_path(args.dataset_dir)
        source_yaml = resolve_path(args.source_yaml) if args.source_yaml else dataset_dir / "data.yaml"
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset root not found: {dataset_dir}")
        if not source_yaml.exists():
            raise FileNotFoundError(f"Dataset yaml not found: {source_yaml}")

        dataset_meta = yaml.safe_load(source_yaml.read_text(encoding="utf-8")) or {}
        dataset_root = infer_dataset_root(source_yaml, dataset_dir, dataset_meta)
        split_alias = {"train": "train", "val": "valid", "test": "test"}
        image_dirs: list[tuple[str, Path]] = []

        for dataset_key, output_name in split_alias.items():
            relative_dir = dataset_meta.get(dataset_key)
            if not relative_dir:
                continue
            image_dir = dataset_root / Path(str(relative_dir))
            if not image_dir.exists():
                raise FileNotFoundError(f"Image directory not found for split '{dataset_key}': {image_dir}")
            image_dirs.append((output_name, image_dir))

        if not image_dirs:
            raise ValueError("No train/val/test image directories were found in the dataset yaml.")
        return image_dirs

    raise ValueError("Provide --image-dir or --dataset-dir.")


def export_comparisons(
    image_dirs: list[tuple[str, Path]],
    output_dir: Path,
    preprocess_settings: ImagePreprocessSettings,
    limit: int,
) -> dict[str, int]:
    exported_counts: dict[str, int] = {}

    for output_name, image_dir in image_dirs:
        images = iter_limited_images(image_dir, limit)
        if not images:
            raise FileNotFoundError(f"No images found in directory: {image_dir}")

        split_output_dir = output_dir if len(image_dirs) == 1 and output_name == "images" else output_dir / output_name
        exported = 0

        for image_path in images:
            original = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if original is None:
                print(f"skip unreadable image: {image_path}")
                continue

            processed = apply_image_preprocess(original, preprocess_settings)
            comparison = build_comparison_canvas(original, processed)

            relative_path = image_path.relative_to(image_dir)
            destination = split_output_dir / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(destination), comparison)
            exported += 1

        exported_counts[output_name] = exported

    return exported_counts


def main() -> int:
    args = parse_args()
    output_dir = ensure_dir(resolve_path(args.output_dir))
    preprocess_settings = build_preprocess_settings(args)

    if not preprocess_settings.is_active:
        raise ValueError(
            "Current preprocess is not active. Pass --preprocess-enabled --preprocess-mode grayscale_weld, "
            "or provide a config with dataset_preprocess enabled."
        )

    image_dirs = resolve_image_dirs(args)
    exported_counts = export_comparisons(
        image_dirs=image_dirs,
        output_dir=output_dir,
        preprocess_settings=preprocess_settings,
        limit=args.limit,
    )

    print(f"output_dir: {output_dir}")
    for split_name, count in exported_counts.items():
        print(f"{split_name}_images: {count}")
    print(f"preprocess_mode: {preprocess_settings.mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
