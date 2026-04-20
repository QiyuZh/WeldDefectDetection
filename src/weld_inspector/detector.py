from __future__ import annotations

import csv
import time

import cv2
import numpy as np

from .config import AppConfig
from .inference.factory import create_backend
from .preprocess import apply_image_preprocess
from .schemas import FrameResult
from .utils.io import ensure_dir, resolve_path, timestamp_string
from .utils.logging import get_logger
from .utils.vision import annotate_frame


class InspectionEngine:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.logger = get_logger("weld_inspector", log_dir=resolve_path("artifacts/logs"))
        self.runtime_dir = ensure_dir(resolve_path(self.config.runtime.save_dir))
        self._csv_path = self.runtime_dir / "inspection_events.csv"
        self._backend = create_backend(self.config.model)

    @property
    def backend_name(self) -> str:
        return self.config.model.effective_backend

    def reload_model(self, model_path: str | None = None, backend: str | None = None) -> None:
        if model_path:
            self.config.model.model_path = model_path
        if backend:
            self.config.model.backend = backend
        self._backend.close()
        self._backend = create_backend(self.config.model)
        self.logger.info("模型已重载: backend=%s path=%s", self.backend_name, self.config.model.model_path)

    def infer(self, frame: np.ndarray, source_name: str, frame_id: int = 0) -> tuple[FrameResult, np.ndarray]:
        started = time.perf_counter()
        model_input = apply_image_preprocess(frame, self.config.inference_preprocess)
        detections = self._backend.predict(model_input)
        elapsed_ms = (time.perf_counter() - started) * 1000
        fps = 1000.0 / elapsed_ms if elapsed_ms > 0 else 0.0
        status = self.config.alarm.ng_text if detections else self.config.alarm.ok_text

        result = FrameResult(
            frame_id=frame_id,
            source=source_name,
            detections=detections,
            inference_ms=elapsed_ms,
            fps=fps,
            status=status,
            backend=self.backend_name,
            model_path=self.config.model.model_path,
            image_size=(frame.shape[1], frame.shape[0]),
        )
        annotated = annotate_frame(frame, result, line_thickness=self.config.runtime.line_thickness)
        result.saved_path = self._persist_outputs(frame=frame, annotated=annotated, result=result)

        if self.config.runtime.save_csv_log:
            self._append_csv_row(result)
        return result, annotated

    def _persist_outputs(self, frame: np.ndarray, annotated: np.ndarray, result: FrameResult) -> str | None:
        should_save = self.config.runtime.save_all_frames or (
            self.config.runtime.save_ng_images and result.has_defect
        )
        if not should_save:
            return None

        timestamp = timestamp_string()
        sub_dir = "ng" if result.has_defect else "ok"
        output_dir = ensure_dir(self.runtime_dir / sub_dir)
        image_name = f"{timestamp}_{result.status}_{result.frame_id:06d}.jpg"
        image_path = output_dir / image_name
        target_image = annotated if self.config.runtime.save_annotated_images else frame
        cv2.imwrite(str(image_path), target_image)
        return str(image_path)

    def _append_csv_row(self, result: FrameResult) -> None:
        file_exists = self._csv_path.exists()
        with self._csv_path.open("a", newline="", encoding="utf-8-sig") as csv_file:
            writer = csv.writer(csv_file)
            if not file_exists:
                writer.writerow(
                    [
                        "frame_id",
                        "source",
                        "status",
                        "defect_count",
                        "backend",
                        "model_path",
                        "inference_ms",
                        "fps",
                        "saved_path",
                    ]
                )
            writer.writerow(
                [
                    result.frame_id,
                    result.source,
                    result.status,
                    result.defect_count,
                    result.backend,
                    result.model_path,
                    f"{result.inference_ms:.3f}",
                    f"{result.fps:.3f}",
                    result.saved_path or "",
                ]
            )

    def close(self) -> None:
        self._backend.close()
