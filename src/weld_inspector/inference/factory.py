from __future__ import annotations

from ..config import ModelSettings
from .base import DetectionBackend
from .onnx_backend import OnnxRuntimeBackend
from .tensorrt_backend import TensorRTBackend
from .ultralytics_backend import UltralyticsBackend


def create_backend(settings: ModelSettings) -> DetectionBackend:
    backend_name = settings.effective_backend
    if backend_name == "ultralytics":
        return UltralyticsBackend(settings)
    if backend_name == "onnx":
        return OnnxRuntimeBackend(settings)
    if backend_name == "tensorrt":
        return TensorRTBackend(settings)
    raise ValueError(f"不支持的后端: {backend_name}")

