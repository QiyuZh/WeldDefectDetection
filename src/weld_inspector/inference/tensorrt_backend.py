from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..schemas import Detection
from ..utils.vision import postprocess_yolov8_output, preprocess_image
from .base import DetectionBackend


@dataclass(slots=True)
class HostDeviceBuffer:
    name: str
    host: np.ndarray
    device: object
    shape: tuple[int, ...]


class TensorRTBackend(DetectionBackend):
    def __init__(self, settings):
        super().__init__(settings)
        try:
            import pycuda.autoinit  # noqa: F401
            import pycuda.driver as cuda
            import tensorrt as trt
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "TensorRT backend requires both TensorRT Python bindings and PyCUDA. "
                "Install them first, for example with `pip install pycuda`, and make "
                "sure the NVIDIA TensorRT runtime is available on PATH."
            ) from exc

        self._cuda = cuda
        self._trt = trt
        self._logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(self._logger)

        with open(settings.model_path, "rb") as engine_file:
            self._engine = self._runtime.deserialize_cuda_engine(engine_file.read())
        if self._engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {settings.model_path}")

        self._context = self._engine.create_execution_context()
        if self._context is None:
            raise RuntimeError("Failed to create TensorRT execution context.")

        self._stream = cuda.Stream()
        self._bindings: list[int] = []
        self._inputs: list[HostDeviceBuffer] = []
        self._outputs: list[HostDeviceBuffer] = []
        self._uses_name_based_api = hasattr(self._engine, "num_io_tensors")
        self._allocate_buffers()

    def _input_shape(self) -> tuple[int, int, int, int]:
        return (1, 3, self.settings.input_size, self.settings.input_size)

    def _binding_indices(self) -> list[int]:
        if self._uses_name_based_api:
            return []
        return list(range(self._engine.num_bindings))

    def _allocate_buffers_name_based(self) -> None:
        input_shape = self._input_shape()
        tensor_mode_input = self._trt.TensorIOMode.INPUT

        for tensor_index in range(self._engine.num_io_tensors):
            tensor_name = self._engine.get_tensor_name(tensor_index)
            if self._engine.get_tensor_mode(tensor_name) == tensor_mode_input:
                self._context.set_input_shape(tensor_name, input_shape)

        for tensor_index in range(self._engine.num_io_tensors):
            tensor_name = self._engine.get_tensor_name(tensor_index)
            shape = tuple(int(dim) for dim in self._context.get_tensor_shape(tensor_name))
            dtype = self._trt.nptype(self._engine.get_tensor_dtype(tensor_name))
            size = int(np.prod(shape))
            host = self._cuda.pagelocked_empty(size, dtype)
            device = self._cuda.mem_alloc(host.nbytes)
            buffer = HostDeviceBuffer(name=tensor_name, host=host, device=device, shape=shape)
            self._context.set_tensor_address(tensor_name, int(device))

            if self._engine.get_tensor_mode(tensor_name) == tensor_mode_input:
                self._inputs.append(buffer)
            else:
                self._outputs.append(buffer)

    def _allocate_buffers_binding_based(self) -> None:
        input_shape = self._input_shape()
        for binding_index in self._binding_indices():
            if self._engine.binding_is_input(binding_index):
                self._context.set_binding_shape(binding_index, input_shape)

        for binding_index in self._binding_indices():
            name = self._engine.get_binding_name(binding_index)
            shape = tuple(int(dim) for dim in self._context.get_binding_shape(binding_index))
            dtype = self._trt.nptype(self._engine.get_binding_dtype(binding_index))
            size = int(np.prod(shape))
            host = self._cuda.pagelocked_empty(size, dtype)
            device = self._cuda.mem_alloc(host.nbytes)
            buffer = HostDeviceBuffer(name=name, host=host, device=device, shape=shape)
            self._bindings.append(int(device))
            if self._engine.binding_is_input(binding_index):
                self._inputs.append(buffer)
            else:
                self._outputs.append(buffer)

    def _allocate_buffers(self) -> None:
        if self._uses_name_based_api:
            self._allocate_buffers_name_based()
        else:
            self._allocate_buffers_binding_based()

        if not self._inputs or not self._outputs:
            raise RuntimeError("TensorRT engine does not expose valid input/output tensors.")

    def predict(self, image: np.ndarray) -> list[Detection]:
        tensor, ratio, pad = preprocess_image(
            image=image,
            input_size=self.settings.input_size,
            use_fp16=self.settings.use_fp16,
        )
        input_buffer = self._inputs[0]
        np.copyto(input_buffer.host, tensor.ravel())
        self._cuda.memcpy_htod_async(input_buffer.device, input_buffer.host, self._stream)

        if self._uses_name_based_api and hasattr(self._context, "execute_async_v3"):
            self._context.execute_async_v3(stream_handle=self._stream.handle)
        else:
            self._context.execute_async_v2(bindings=self._bindings, stream_handle=self._stream.handle)

        for output_buffer in self._outputs:
            self._cuda.memcpy_dtoh_async(output_buffer.host, output_buffer.device, self._stream)
        self._stream.synchronize()

        raw_output = self._outputs[0].host.reshape(self._outputs[0].shape)
        return postprocess_yolov8_output(
            raw_output=raw_output,
            original_shape=image.shape,
            ratio=ratio,
            pad=pad,
            class_names=self.settings.class_names,
            conf_threshold=self.settings.conf_threshold,
            iou_threshold=self.settings.iou_threshold,
        )

    def close(self) -> None:
        self._inputs.clear()
        self._outputs.clear()
        self._bindings.clear()
