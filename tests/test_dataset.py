from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
import shutil
from types import SimpleNamespace

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
TEST_TMP_DIR = PROJECT_ROOT / ".tmp_tests"
TEST_TMP_DIR.mkdir(parents=True, exist_ok=True)

from weld_inspector.config import ImagePreprocessSettings
from weld_inspector.dataset import build_yolo_dataset, split_items
from compare_preprocess import export_comparisons, resolve_image_dirs
from prepare_grayscale_dataset import prepare_structured_dataset


class DatasetTests(unittest.TestCase):
    def test_split_items(self) -> None:
        items = [Path(f"image_{idx}.jpg") for idx in range(10)]
        train_items, val_items = split_items(items, train_ratio=0.8, seed=7)
        self.assertEqual(len(train_items), 8)
        self.assertEqual(len(val_items), 2)
        self.assertFalse(set(train_items) & set(val_items))

    def test_build_yolo_dataset(self) -> None:
        case_dir = TEST_TMP_DIR / "dataset_case"
        shutil.rmtree(case_dir, ignore_errors=True)
        temp_root = case_dir
        image_dir = temp_root / "images"
        label_dir = temp_root / "labels"
        output_dir = temp_root / "dataset"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        for idx in range(4):
            (image_dir / f"sample_{idx}.jpg").write_bytes(b"fake-image")
        (label_dir / "sample_0.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
        (label_dir / "sample_1.txt").write_text("1 0.4 0.4 0.2 0.2\n", encoding="utf-8")

        stats = build_yolo_dataset(
            image_dir=image_dir,
            label_dir=label_dir,
            output_dir=output_dir,
            class_names=["crack", "hole", "unwelded", "offset_weld"],
            train_ratio=0.75,
            seed=3,
        )

        self.assertEqual(stats.total_images, 4)
        self.assertEqual(stats.train_images, 3)
        self.assertEqual(stats.val_images, 1)
        self.assertTrue((output_dir / "weld.yaml").exists())
        self.assertEqual(len(list((output_dir / "images" / "train").glob("*.jpg"))), 3)
        self.assertEqual(len(list((output_dir / "images" / "val").glob("*.jpg"))), 1)

    def test_build_yolo_dataset_with_grayscale_preprocess(self) -> None:
        case_dir = TEST_TMP_DIR / "dataset_gray_case"
        shutil.rmtree(case_dir, ignore_errors=True)
        image_dir = case_dir / "images"
        label_dir = case_dir / "labels"
        output_dir = case_dir / "dataset_gray"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        source_image = np.zeros((48, 64, 3), dtype=np.uint8)
        source_image[:, :, 1] = 120
        source_image[12:36, 20:44] = (240, 240, 240)
        cv2.imwrite(str(image_dir / "sample_0.jpg"), source_image)
        (label_dir / "sample_0.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

        stats = build_yolo_dataset(
            image_dir=image_dir,
            label_dir=label_dir,
            output_dir=output_dir,
            class_names=["porosity"],
            preprocess_settings=ImagePreprocessSettings(
                enabled=True,
                mode="grayscale_weld",
                clahe_clip_limit=2.5,
                clahe_tile_grid_size=8,
                blur_kernel_size=3,
                unsharp_amount=1.0,
            ),
        )

        processed_path = next((output_dir / "images" / "train").glob("*.jpg"))
        processed_image = cv2.imread(str(processed_path), cv2.IMREAD_COLOR)
        self.assertIsNotNone(processed_image)
        self.assertEqual(stats.total_images, 1)
        self.assertTrue(np.array_equal(processed_image[:, :, 0], processed_image[:, :, 1]))
        self.assertTrue(np.array_equal(processed_image[:, :, 1], processed_image[:, :, 2]))

    def test_prepare_structured_grayscale_dataset_reuses_labels(self) -> None:
        case_dir = TEST_TMP_DIR / "structured_gray_case"
        shutil.rmtree(case_dir, ignore_errors=True)

        dataset_root = case_dir / "datasets"
        output_root = case_dir / "datasets_gray"
        dataset_config = case_dir / "weld_grayscale.yaml"

        for split_name in ("train", "valid", "test"):
            (dataset_root / split_name / "images").mkdir(parents=True, exist_ok=True)
            (dataset_root / split_name / "labels").mkdir(parents=True, exist_ok=True)

        source_image = np.zeros((40, 60, 3), dtype=np.uint8)
        source_image[:, :, 2] = 80
        source_image[10:30, 18:42] = (255, 255, 255)
        cv2.imwrite(str(dataset_root / "train" / "images" / "sample_train.jpg"), source_image)
        cv2.imwrite(str(dataset_root / "valid" / "images" / "sample_valid.jpg"), source_image)
        cv2.imwrite(str(dataset_root / "test" / "images" / "sample_test.jpg"), source_image)

        train_label = "0 0.5 0.5 0.2 0.2\n"
        valid_label = "1 0.4 0.4 0.3 0.3\n"
        test_label = "2 0.6 0.6 0.1 0.1\n"
        (dataset_root / "train" / "labels" / "sample_train.txt").write_text(train_label, encoding="utf-8")
        (dataset_root / "valid" / "labels" / "sample_valid.txt").write_text(valid_label, encoding="utf-8")
        (dataset_root / "test" / "labels" / "sample_test.txt").write_text(test_label, encoding="utf-8")

        (dataset_root / "data.yaml").write_text(
            "\n".join(
                [
                    "train: train/images",
                    "val: valid/images",
                    "test: test/images",
                    "names:",
                    "  - Geometric defect",
                    "  - porosity",
                    "  - spatters",
                ]
            ),
            encoding="utf-8",
        )

        stats, generated_yaml = prepare_structured_dataset(
            dataset_dir=dataset_root,
            output_dir=output_root,
            dataset_config=dataset_config,
            preprocess_settings=ImagePreprocessSettings(
                enabled=True,
                mode="grayscale_weld",
                clahe_clip_limit=2.5,
                clahe_tile_grid_size=8,
                blur_kernel_size=3,
                unsharp_amount=1.0,
            ),
        )

        self.assertEqual(stats.total_images, 3)
        self.assertEqual(stats.split_images["train"], 1)
        self.assertEqual(stats.split_images["val"], 1)
        self.assertEqual(stats.split_images["test"], 1)
        self.assertTrue(generated_yaml.exists())
        self.assertEqual(
            (output_root / "train" / "labels" / "sample_train.txt").read_text(encoding="utf-8"),
            train_label,
        )
        self.assertEqual(
            (output_root / "valid" / "labels" / "sample_valid.txt").read_text(encoding="utf-8"),
            valid_label,
        )
        self.assertEqual(
            (output_root / "test" / "labels" / "sample_test.txt").read_text(encoding="utf-8"),
            test_label,
        )

        processed_train_image = cv2.imread(str(output_root / "train" / "images" / "sample_train.jpg"), cv2.IMREAD_COLOR)
        self.assertIsNotNone(processed_train_image)
        self.assertTrue(np.array_equal(processed_train_image[:, :, 0], processed_train_image[:, :, 1]))
        self.assertTrue(np.array_equal(processed_train_image[:, :, 1], processed_train_image[:, :, 2]))

    def test_compare_preprocess_supports_dataset_dir(self) -> None:
        case_dir = TEST_TMP_DIR / "compare_structured_case"
        shutil.rmtree(case_dir, ignore_errors=True)

        dataset_root = case_dir / "datasets"
        output_root = case_dir / "preprocess_compare"

        for split_name in ("train", "valid", "test"):
            (dataset_root / split_name / "images").mkdir(parents=True, exist_ok=True)

        source_image = np.zeros((32, 48, 3), dtype=np.uint8)
        source_image[:, :, 0] = 60
        source_image[8:24, 12:36] = (255, 255, 255)
        cv2.imwrite(str(dataset_root / "train" / "images" / "sample_train.jpg"), source_image)
        cv2.imwrite(str(dataset_root / "valid" / "images" / "sample_valid.jpg"), source_image)
        cv2.imwrite(str(dataset_root / "test" / "images" / "sample_test.jpg"), source_image)

        (dataset_root / "data.yaml").write_text(
            "\n".join(
                [
                    "train: train/images",
                    "val: valid/images",
                    "test: test/images",
                    "names:",
                    "  - crack",
                ]
            ),
            encoding="utf-8",
        )

        image_dirs = resolve_image_dirs(
            SimpleNamespace(
                image_dir=None,
                dataset_dir=str(dataset_root),
                source_yaml=None,
            )
        )
        exported_counts = export_comparisons(
            image_dirs=image_dirs,
            output_dir=output_root,
            preprocess_settings=ImagePreprocessSettings(
                enabled=True,
                mode="grayscale_weld",
                clahe_clip_limit=2.5,
                clahe_tile_grid_size=8,
                blur_kernel_size=3,
                unsharp_amount=1.0,
            ),
            limit=1,
        )

        self.assertEqual(exported_counts["train"], 1)
        self.assertEqual(exported_counts["valid"], 1)
        self.assertEqual(exported_counts["test"], 1)
        self.assertTrue((output_root / "train" / "sample_train.jpg").exists())
        self.assertTrue((output_root / "valid" / "sample_valid.jpg").exists())
        self.assertTrue((output_root / "test" / "sample_test.jpg").exists())


if __name__ == "__main__":
    unittest.main()
