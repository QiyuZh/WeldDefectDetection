from __future__ import annotations

import ast

import numpy as np

from ..schemas import Detection
from ..utils.vision import postprocess_yolov8_output, preprocess_image
from .base import DetectionBackend


class OnnxRuntimeBackend(DetectionBackend):
    def __init__(self, settings):
        super().__init__(settings)
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "未安装 onnxruntime，请执行 `pip install onnxruntime` 或 `onnxruntime-gpu`。"
            ) from exc

        providers = ["CPUExecutionProvider"]
        if settings.device.lower() != "cpu":
            available_providers = ort.get_available_providers()
            if "CUDAExecutionProvider" in available_providers:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self._session = ort.InferenceSession(settings.model_path, providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name
        self._class_names = list(settings.class_names) or self._load_class_names_from_metadata() or []

    def _load_class_names_from_metadata(self) -> list[str] | None:
        try:
            metadata_map = self._session.get_modelmeta().custom_metadata_map or {}
        except Exception:  # pragma: no cover
            return None

        raw_names = metadata_map.get("names")
        if not raw_names:
            return None

        try:
            parsed = ast.literal_eval(raw_names)
        except (ValueError, SyntaxError):
            return None

        if isinstance(parsed, dict):
            resolved: list[str] = []
            for key in sorted(parsed, key=lambda item: int(item)):
                resolved.append(str(parsed[key]))
            return resolved or None
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
        return None

    def predict(self, image: np.ndarray) -> list[Detection]:
        tensor, ratio, pad = preprocess_image(
            image=image,
            input_size=self.settings.input_size,
            use_fp16=False,
        )
        outputs = self._session.run([self._output_name], {self._input_name: tensor})
        return postprocess_yolov8_output(
            raw_output=outputs[0],
            original_shape=image.shape,
            ratio=ratio,
            pad=pad,
            class_names=self._class_names,
            conf_threshold=self.settings.conf_threshold,
            iou_threshold=self.settings.iou_threshold,
        )
