# Conda 环境清单

这套项目建议分成两种环境：

- `weld-qc-cpu`：仅做代码验证、接口联调、CPU 推理
- `weld-qc-gpu`：训练、GPU 推理、桌面端演示、ONNXRuntime GPU

## 1. 为什么建议 Python 3.10

当前项目虽然代码本身兼容更高版本 Python，但如果你要落到：

- PyTorch GPU
- ONNXRuntime GPU
- TensorRT Python 绑定
- PyInstaller 打包

那么 `Python 3.10` 在 Windows 工控机上通常最稳，兼容性明显好于 3.12。

## 2. 创建环境

### CPU 环境

```powershell
conda env create -f conda/environment.cpu.yml
conda activate weld-qc-cpu
```

### GPU 环境

```powershell
conda env create -f conda/environment.gpu.yml
conda activate weld-qc-gpu
```

如果你的机器访问 `https://conda.anaconda.org/pytorch/` 失败，改用后备环境文件：

```powershell
conda env create -f conda/environment.gpu-pip-torch.yml
conda activate weld-qc-gpu
```

这个后备方案仍然使用 `conda` 管理环境，只是把 `torch` / `torchvision` 改成通过 `pip` 从官方 CUDA 11.8 wheel 安装。

另外，`pycuda` 我不再放进环境创建阶段统一安装，因为它在 Windows 上经常受 CUDA、Visual Studio、TensorRT 版本影响。建议在 TensorRT 阶段单独安装。

## 3. 包用途说明

### 通用基础包

- `python=3.10`：统一运行时版本，兼顾 PyTorch / PyInstaller / TensorRT 兼容性
- `pip`：补充安装 pip 生态里的视觉和部署包
- `numpy`：张量与数组计算
- `pyyaml`：读取 `configs/app.yaml`
- `opencv`：图像读写、视频流、绘框、编码

### 训练与导出

- `pytorch`、`torchvision`：YOLOv8 训练和 `.pt` 推理
- `ultralytics`：YOLOv8 主框架
- `onnx`：导出 ONNX
- `onnxsim`：简化 ONNX 图

### 推理与服务

- `onnxruntime` / `onnxruntime-gpu`：ONNX 推理后端
- `PySide6`：桌面端 GUI
- `fastapi`：HTTP 推理服务
- `uvicorn[standard]`：FastAPI 服务启动
- `python-multipart`：API 接收图片文件上传

### 打包与开发

- `pyinstaller`：打包桌面端 / API 为 exe
- `ruff`：代码检查

### 仅 GPU 环境额外包

- `pytorch-cuda=11.8`：PyTorch 的 CUDA 运行时
- `pycuda`：TensorRT 推理时的显存管理和数据拷贝

## 4. TensorRT 说明

`TensorRT` 本身没有直接写进 `environment.gpu.yml`，原因不是漏掉，而是：

- Windows 上 TensorRT Python 包通常要和本机 CUDA、驱动、TensorRT 版本严格匹配
- 最稳妥的做法是先安装 NVIDIA 官方 TensorRT，再把对应的 Python 包装进当前环境

也就是说：

- 训练和普通 GPU 推理：`environment.gpu.yml` 已够用
- `.trt` / `.engine` 真正部署：还需要你在目标机额外安装 TensorRT

## 5. 推荐安装顺序

1. 安装显卡驱动
2. 创建 `weld-qc-gpu` 环境
3. 验证 `torch.cuda.is_available()`
4. 再安装 TensorRT
5. 执行 `scripts/build_tensorrt.py`
6. 最后执行 PyInstaller 打包

## 6. 快速验证命令

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -c "import cv2, yaml, ultralytics; print(cv2.__version__)"
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

如果你已经装好 TensorRT，再额外验证：

```powershell
python -c "import tensorrt as trt; print(trt.__version__)"
python -c "import pycuda.driver as cuda; import pycuda.autoinit; print('pycuda ok')"
```
