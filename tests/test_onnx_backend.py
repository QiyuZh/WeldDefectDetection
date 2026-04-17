from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if "cv2" not in sys.modules:
    sys.modules["cv2"] = types.SimpleNamespace(
        dnn=types.SimpleNamespace(NMSBoxes=None),
        FONT_HERSHEY_SIMPLEX=0,
        LINE_AA=0,
        BORDER_CONSTANT=0,
        INTER_LINEAR=0,
        COLOR_BGR2RGB=0,
    )

from weld_inspector.inference.onnx_backend import _resolve_runtime_input_size


class OnnxBackendHelperTests(unittest.TestCase):
    def test_fixed_model_shape_overrides_configured_input_size(self) -> None:
        size, from_model = _resolve_runtime_input_size((1, 3, 640, 640), 960)
        self.assertEqual(size, (640, 640))
        self.assertTrue(from_model)

    def test_dynamic_model_shape_uses_configured_input_size(self) -> None:
        size, from_model = _resolve_runtime_input_size((1, 3, "height", "width"), 960)
        self.assertEqual(size, (960, 960))
        self.assertFalse(from_model)


if __name__ == "__main__":
    unittest.main()
