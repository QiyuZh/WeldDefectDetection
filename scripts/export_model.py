from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出 YOLOv8 模型为 ONNX 或 TensorRT Engine")
    parser.add_argument("--weights", required=True, help="PyTorch 权重路径")
    parser.add_argument("--format", choices=["onnx", "engine"], default="onnx")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=13)
    parser.add_argument("--device", default="0")
    parser.add_argument("--half", action="store_true", help="使用 FP16")
    parser.add_argument("--dynamic", action="store_true", help="导出动态输入尺寸的 ONNX")
    parser.add_argument("--workspace", type=float, default=4.0, help="TensorRT workspace GB")
    parser.add_argument("--output", default=None, help="指定导出产物路径")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("未安装 ultralytics，请先 `pip install -r requirements/train.txt`。") from exc

    weights_path = Path(args.weights)
    model = YOLO(str(weights_path))
    export_kwargs = {
        "format": args.format,
        "imgsz": args.imgsz,
        "device": args.device,
        "half": args.half,
    }
    if args.format == "onnx":
        export_kwargs["opset"] = args.opset
        export_kwargs["simplify"] = True
        export_kwargs["dynamic"] = args.dynamic
    if args.format == "engine":
        export_kwargs["workspace"] = args.workspace
    exported_path = model.export(**export_kwargs)
    exported_path = Path(exported_path)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if exported_path.resolve() != output_path.resolve():
            shutil.copy2(exported_path, output_path)
        print(f"导出完成: {output_path}")
    else:
        print(f"导出完成: {exported_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
