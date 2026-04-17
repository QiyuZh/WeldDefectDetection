from __future__ import annotations

import ast
import warnings
from typing import Any, Sequence

import numpy as np

from ..schemas import Detection
from ..utils.vision import postprocess_yolov8_output, preprocess_image
from .base import DetectionBackend


def _normalize_input_size(value: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(value, int):
        return (value, value)
    if len(value) != 2:
        raise ValueError(f"input_size 必须是 int 或长度为 2 的序列，当前收到: {value!r}")
    return (int(value[0]), int(value[1]))


def _is_fixed_onnx_dimension(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _resolve_runtime_input_size(
    input_shape: Sequence[Any],
    configured_input_size: int | Sequence[int],
) -> tuple[tuple[int, int], bool]:
    configured_size = _normalize_input_size(configured_input_size)
    if len(input_shape) >= 4:
        height = input_shape[-2]
        width = input_shape[-1]
        if _is_fixed_onnx_dimension(height) and _is_fixed_onnx_dimension(width):
            return (int(height), int(width)), True
    return configured_size, False


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
        self._input_meta = self._session.get_inputs()[0]
        self._input_name = self._input_meta.name
        self._output_name = self._session.get_outputs()[0].name
        self._input_size, self._uses_fixed_model_input = _resolve_runtime_input_size(
            input_shape=self._input_meta.shape,
            configured_input_size=self.settings.input_size,
        )
        configured_input_size = _normalize_input_size(self.settings.input_size)
        if self._uses_fixed_model_input and self._input_size != configured_input_size:
            warnings.warn(
                (
                    f"当前 ONNX 模型输入固定为 {self._input_size[0]}x{self._input_size[1]}，"
                    f"app.yaml 中的 input_size={configured_input_size[0]}x{configured_input_size[1]} "
                    "不会改变已导出的 ONNX 图结构。若要按新尺寸推理，请重新导出 ONNX "
                    "或使用动态输入导出。"
                ),
                RuntimeWarning,
                stacklevel=2,
            )
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
            input_size=self._input_size,
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
