from __future__ import annotations

import numpy as np

from ..schemas import Detection
from .base import DetectionBackend


class UltralyticsBackend(DetectionBackend):
    def __init__(self, settings):
        super().__init__(settings)
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("未安装 ultralytics，请执行 `pip install ultralytics`。") from exc

        self._model = YOLO(settings.model_path)

    def predict(self, image: np.ndarray) -> list[Detection]:
        results = self._model.predict(
            source=image,
            conf=self.settings.conf_threshold,
            iou=self.settings.iou_threshold,
            imgsz=self.settings.input_size,
            device=self.settings.device,
            half=self.settings.use_fp16,
            verbose=False,
        )
        result = results[0]
        detections: list[Detection] = []
        for box in result.boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            class_id = int(box.cls.item())
            label = (
                self.settings.class_names[class_id]
                if class_id < len(self.settings.class_names)
                else str(class_id)
            )
            detections.append(
                Detection(
                    label=label,
                    confidence=float(box.conf.item()),
                    bbox=(x1, y1, x2, y2),
                )
            )
        return detections

