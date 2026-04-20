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

from weld_inspector.config import (
    AppConfig,
    infer_backend_from_model,
    load_app_config,
    save_app_config,
    yaml,
)


class ConfigTests(unittest.TestCase):
    def test_infer_backend_from_model(self) -> None:
        self.assertEqual(infer_backend_from_model("best.pt"), "ultralytics")
        self.assertEqual(infer_backend_from_model("best.onnx"), "onnx")
        self.assertEqual(infer_backend_from_model("best.trt"), "tensorrt")
        self.assertEqual(infer_backend_from_model("best.engine"), "tensorrt")

    @unittest.skipUnless(yaml is not None, "PyYAML not installed")
    def test_save_and_load_config(self) -> None:
        case_dir = TEST_TMP_DIR / "config_case"
        shutil.rmtree(case_dir, ignore_errors=True)
        case_dir.mkdir(parents=True, exist_ok=True)
        config_path = case_dir / "app.yaml"
        config = AppConfig()
        config.model.model_path = "artifacts/models/custom.onnx"
        config.dataset_preprocess.enabled = True
        config.dataset_preprocess.mode = "grayscale_weld"
        config.training.best_output_name = "best_gray.pt"
        save_app_config(config, config_path)
        loaded = load_app_config(config_path)
        self.assertEqual(loaded.model.model_path, "artifacts/models/custom.onnx")
        self.assertEqual(loaded.model.effective_backend, "onnx")
        self.assertTrue(loaded.dataset_preprocess.enabled)
        self.assertEqual(loaded.dataset_preprocess.mode, "grayscale_weld")
        self.assertEqual(loaded.training.best_output_name, "best_gray.pt")

    @unittest.skipUnless(yaml is not None, "PyYAML not installed")
    def test_load_class_names_from_dataset_yaml_when_missing_in_app_config(self) -> None:
        case_dir = TEST_TMP_DIR / "dataset_name_sync_case"
        shutil.rmtree(case_dir, ignore_errors=True)
        case_dir.mkdir(parents=True, exist_ok=True)
        dataset_yaml = case_dir / "data.yaml"
        dataset_yaml.write_text(
            "\n".join(
                [
                    "train: train/images",
                    "val: valid/images",
                    "names:",
                    "  - Geometric defect",
                    "  - Non-fusion defect",
                    "  - crack",
                ]
            ),
            encoding="utf-8",
        )
        config_path = case_dir / "app.yaml"
        config_path.write_text(
            "\n".join(
                [
                    "project_name: sync-test",
                    "model:",
                    "  backend: onnx",
                    "  model_path: artifacts/models/custom.onnx",
                    "training:",
                    f"  data_yaml: {dataset_yaml.as_posix()}",
                ]
            ),
            encoding="utf-8",
        )

        loaded = load_app_config(config_path)
        self.assertEqual(
            loaded.model.class_names,
            ["Geometric defect", "Non-fusion defect", "crack"],
        )

    @unittest.skipUnless(yaml is not None, "PyYAML not installed")
    def test_explicit_class_names_override_dataset_yaml(self) -> None:
        case_dir = TEST_TMP_DIR / "explicit_class_names_case"
        shutil.rmtree(case_dir, ignore_errors=True)
        case_dir.mkdir(parents=True, exist_ok=True)
        dataset_yaml = case_dir / "data.yaml"
        dataset_yaml.write_text(
            "\n".join(
                [
                    "train: train/images",
                    "val: valid/images",
                    "names:",
                    "  - should_not",
                    "  - be_used",
                ]
            ),
            encoding="utf-8",
        )
        config_path = case_dir / "app.yaml"
        config_path.write_text(
            "\n".join(
                [
                    "project_name: explicit-test",
                    "model:",
                    "  class_names:",
                    "    - custom_a",
                    "    - custom_b",
                    "training:",
                    f"  data_yaml: {dataset_yaml.as_posix()}",
                ]
            ),
            encoding="utf-8",
        )

        loaded = load_app_config(config_path)
        self.assertEqual(loaded.model.class_names, ["custom_a", "custom_b"])


if __name__ == "__main__":
    unittest.main()
