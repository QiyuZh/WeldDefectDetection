from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从 ONNX 构建 TensorRT 引擎")
    parser.add_argument("--onnx", required=True, help="ONNX 模型路径")
    parser.add_argument("--engine", required=True, help="输出 TensorRT 引擎路径")
    parser.add_argument("--weights", default=None, help="可选，若未安装 trtexec 时回退到 ultralytics 导出")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--workspace", type=int, default=4096, help="workspace MB")
    parser.add_argument("--trtexec", default="trtexec", help="trtexec 可执行文件路径")
    return parser.parse_args()


def has_module(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def main() -> int:
    args = parse_args()
    onnx_path = Path(args.onnx)
    engine_path = Path(args.engine)
    engine_path.parent.mkdir(parents=True, exist_ok=True)

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX 文件不存在: {onnx_path}")

    trtexec_path = shutil.which(args.trtexec)
    if trtexec_path:
        command = [
            trtexec_path,
            f"--onnx={onnx_path}",
            f"--saveEngine={engine_path}",
            f"--workspace={args.workspace}",
            "--skipInference",
        ]
        if args.fp16:
            command.append("--fp16")
        print(" ".join(command))
        subprocess.run(command, check=True)
        print(f"TensorRT 引擎已生成: {engine_path}")
        return 0

    if args.weights:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "未找到 trtexec，且 ultralytics 不可用，无法回退生成 engine。"
            ) from exc
        model = YOLO(args.weights)
        exported = model.export(
            format="engine",
            imgsz=args.imgsz,
            half=args.fp16,
            workspace=max(1, int(args.workspace / 1024)),
            device=0,
        )
        shutil.copy2(exported, engine_path)
        print(f"TensorRT 引擎已生成: {engine_path}")
        return 0

    tensorrt_installed = has_module("tensorrt")
    pycuda_installed = has_module("pycuda")
    missing_parts: list[str] = []
    if not trtexec_path:
        missing_parts.append("trtexec.exe")
    if not tensorrt_installed:
        missing_parts.append("tensorrt Python 包")
    if not pycuda_installed:
        missing_parts.append("pycuda")

    details = "、".join(missing_parts) if missing_parts else "TensorRT 组件"
    raise FileNotFoundError(
        "当前无法构建 TensorRT 引擎，缺少："
        f"{details}。\n"
        "处理方式：\n"
        "1. 已安装 TensorRT 但未配置 PATH：把 TensorRT/bin 加入 PATH，或通过 --trtexec 传入 trtexec.exe 的完整路径。\n"
        "2. 未安装 TensorRT：先安装 TensorRT，再安装 tensorrt Python 包；需要运行 TensorRT 后端时再安装 pycuda。\n"
        "3. 如果只是先把系统跑通：直接使用 ONNX 后端，不必先生成 .trt。\n"
        "   可在 configs/app.yaml 中设置 model.backend=onnx，model.model_path=artifacts/models/model.onnx。\n"
        "4. 如果你已有 .pt 权重并且本机已装好 TensorRT，也可以传入 --weights 让脚本回退到 ultralytics 导出 engine。"
    )


if __name__ == "__main__":
    raise SystemExit(main())
