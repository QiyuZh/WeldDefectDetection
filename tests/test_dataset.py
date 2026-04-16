from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
import shutil

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
TEST_TMP_DIR = PROJECT_ROOT / ".tmp_tests"
TEST_TMP_DIR.mkdir(parents=True, exist_ok=True)

from weld_inspector.dataset import build_yolo_dataset, split_items


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


if __name__ == "__main__":
    unittest.main()
