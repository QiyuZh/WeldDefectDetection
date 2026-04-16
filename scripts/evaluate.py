from __future__ import annotations

import argparse
from datetime import datetime
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from weld_inspector.config import load_app_config, resolve_project_path
from weld_inspector.utils.io import write_json, write_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估焊缝缺陷 YOLOv8 模型")
    parser.add_argument("--config", default="configs/app.yaml")
    parser.add_argument("--weights", required=True, help="权重路径")
    parser.add_argument("--data", default=None, help="数据集 yaml")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_app_config(args.config)

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("未安装 ultralytics，请先 `pip install -r requirements/train.txt`。") from exc

    model = YOLO(args.weights)
    metrics = model.val(
        data=str(resolve_project_path(args.data or config.training.data_yaml)),
        imgsz=config.training.imgsz,
        batch=config.training.batch,
        conf=config.model.conf_threshold,
        iou=config.model.iou_threshold,
        device=config.training.device,
    )

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "weights": args.weights,
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "speed_ms": {key: float(value) for key, value in metrics.speed.items()},
    }

    report_dir = PROJECT_ROOT / "artifacts" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = write_json(report_dir / f"eval_{timestamp}.json", payload)
    markdown = "\n".join(
        [
            "# 焊缝缺陷模型评估报告",
            "",
            f"- 生成时间：{payload['generated_at']}",
            f"- 权重文件：{payload['weights']}",
            f"- mAP@0.5：{payload['map50']:.4f}",
            f"- mAP@0.5:0.95：{payload['map50_95']:.4f}",
            f"- Precision：{payload['precision']:.4f}",
            f"- Recall：{payload['recall']:.4f}",
            f"- 推理耗时(ms)：{payload['speed_ms']}",
        ]
    )
    md_path = write_text(report_dir / f"eval_{timestamp}.md", markdown)
    print(f"评估报告已生成: {json_path}")
    print(f"评估摘要已生成: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

