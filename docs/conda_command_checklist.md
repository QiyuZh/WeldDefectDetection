# 基于 Conda 的完整安装与启动命令清单

下面默认你在 `D:\WeldDefectDetection` 项目根目录下执行命令，终端使用 PowerShell。

## 1. 首次创建 GPU 环境

推荐正式开发、训练、导出、桌面端演示都用这个环境。

```powershell
cd D:\WeldDefectDetection
conda env create -f conda\environment.gpu.yml
conda activate weld-qc-gpu
```

### 如果这里报 `CondaHTTPError: HTTP 000` 且卡在 `conda.anaconda.org/pytorch`

说明你当前机器访问 `pytorch` 的 conda 渠道失败。不要反复硬重试，直接切到下面这套后备方案：

```powershell
cd D:\WeldDefectDetection
conda env remove -n weld-qc-gpu -y
conda env create -f conda\environment.gpu-pip-torch.yml
conda activate weld-qc-gpu
```

这套后备方案仍然是 `conda` 环境，只是把 `torch` 和 `torchvision` 改成在环境里通过 `pip` 安装官方 CUDA 11.8 wheel，避开 `pytorch` 的 conda 下载地址。

同时我已经把环境文件改成了“PyTorch 直链 wheel + 其它包走默认 PyPI”，这样不会再出现 `ultralytics` 被错误地去 PyTorch 源里查找的问题。

## 2. 首次创建 CPU 环境

如果你当前机器没有 NVIDIA GPU，或者只想做代码联调，用这个环境。

```powershell
cd D:\WeldDefectDetection
conda env create -f conda\environment.cpu.yml
conda activate weld-qc-cpu
```

## 3. 环境基础验证

### GPU 环境验证

```powershell
conda activate weld-qc-gpu
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -c "import cv2, yaml, ultralytics; print(cv2.__version__)"
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

### CPU 环境验证

```powershell
conda activate weld-qc-cpu
python -c "import cv2, yaml, ultralytics; print(cv2.__version__)"
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

## 4. 可选：安装 TensorRT 后验证

这一步只在你准备生成或运行 `.trt` / `.engine` 模型时需要。

```powershell
conda activate weld-qc-gpu
python -c "import tensorrt as trt; print(trt.__version__)"
python -c "import pycuda.driver as cuda; import pycuda.autoinit; print('pycuda ok')"
trtexec --help
```

## 5. 准备原始数据

先把你的图片和标签放到这两个目录：

```text
data/raw/images
data/raw/labels
```

如果目录还没建，可以先执行：

```powershell
mkdir data\raw\images -Force
mkdir data\raw\labels -Force
```

## 6. 整理成 YOLOv8 数据集

```powershell
conda activate weld-qc-gpu
python scripts\prepare_dataset.py `
  --image-dir data\raw\images `
  --label-dir data\raw\labels `
  --output-dir data\dataset
```

执行完成后，你会得到：

- `data/dataset/images/train`
- `data/dataset/images/val`
- `data/dataset/labels/train`
- `data/dataset/labels/val`
- `configs/dataset/weld.yaml`

## 7. 训练 YOLOv8

```powershell
conda activate weld-qc-gpu
python scripts\train_yolov8.py --config configs\app.yaml
```

如果你想临时覆盖参数：

```powershell
conda activate weld-qc-gpu
python scripts\train_yolov8.py `
  --config configs\app.yaml `
  --epochs 150 `
  --imgsz 1280 `
  --batch 4 `
  --device 0
```

训练完成后重点看：

- `artifacts/models/best.pt`
- `artifacts/models/last.pt`
- `artifacts/models/yolov8_train/`

## 8. 评估模型

```powershell
conda activate weld-qc-gpu
python scripts\evaluate.py `
  --config configs\app.yaml `
  --weights artifacts\models\best.pt
```

评估结果会输出到：

- `artifacts/reports/*.json`
- `artifacts/reports/*.md`

## 9. 导出 ONNX

```powershell
conda activate weld-qc-gpu
python scripts\export_model.py `
  --weights artifacts\models\best.pt `
  --format onnx `
  --imgsz 640 `
  --device 0 `
  --output artifacts\models\best.onnx
```

## 10. 构建 TensorRT 引擎

### 方式 A：本机已安装 `trtexec`

```powershell
conda activate weld-qc-gpu
python scripts\build_tensorrt.py `
  --onnx artifacts\models\best.onnx `
  --engine artifacts\models\best.trt `
  --fp16
```

### 方式 B：没有 `trtexec`，回退到 Ultralytics

```powershell
conda activate weld-qc-gpu
python scripts\build_tensorrt.py `
  --onnx artifacts\models\best.onnx `
  --engine artifacts\models\best.trt `
  --weights artifacts\models\best.pt `
  --fp16
```

## 11. 运行桌面端

如果你先想用 `.pt` 模型验证，先确保 `configs/app.yaml` 里的 `model_path` 指向：

```text
artifacts/models/best.pt
```

启动命令：

```powershell
conda activate weld-qc-gpu
python scripts\run_desktop.py --config configs\app.yaml
```

如果你要切到 ONNX 或 TensorRT，修改 `configs/app.yaml` 的：

- `model.backend`
- `model.model_path`

然后重新启动桌面端。

## 12. 运行 HTTP 推理服务

```powershell
conda activate weld-qc-gpu
python scripts\run_api.py --config configs\app.yaml
```

启动后可验证：

```powershell
curl http://127.0.0.1:18080/health
```

## 13. 打包桌面端 exe

```powershell
conda activate weld-qc-gpu
powershell -ExecutionPolicy Bypass -File scripts\package_desktop.ps1
```

## 14. 打包 API 服务 exe

```powershell
conda activate weld-qc-gpu
powershell -ExecutionPolicy Bypass -File scripts\package_api.ps1
```

## 15. 每天常用最短命令流

### 日常训练

```powershell
cd D:\WeldDefectDetection
conda activate weld-qc-gpu
python scripts\train_yolov8.py --config configs\app.yaml
```

### 日常桌面端验证

```powershell
cd D:\WeldDefectDetection
conda activate weld-qc-gpu
python scripts\run_desktop.py --config configs\app.yaml
```

### 日常 API 验证

```powershell
cd D:\WeldDefectDetection
conda activate weld-qc-gpu
python scripts\run_api.py --config configs\app.yaml
```

## 16. 环境维护命令

### 更新环境

```powershell
conda env update -f conda\environment.gpu.yml --prune
```

### 查看环境

```powershell
conda env list
conda list -n weld-qc-gpu
```

### 删除环境

```powershell
conda remove -n weld-qc-gpu --all
conda remove -n weld-qc-cpu --all
```

## 17. 最后确认

完整交付链路对应关系如下：

1. `conda env create`
2. `prepare_dataset.py`
3. `train_yolov8.py`
4. `evaluate.py`
5. `export_model.py`
6. `build_tensorrt.py`
7. `run_desktop.py` / `run_api.py`
8. `package_desktop.ps1` / `package_api.ps1`

只要你把真实数据和真实模型跑起来，这套命令就能直接覆盖从开发到交付的完整流程。

## 18. 你现在这个报错的最短修复命令

如果你当前看到的是：

- `CondaHTTPError: HTTP 000 CONNECTION FAILED`
- 失败地址是 `https://conda.anaconda.org/pytorch/...`

直接执行下面这组：

```powershell
cd D:\WeldDefectDetection
conda env remove -n weld-qc-gpu -y
conda env create -f conda\environment.gpu-pip-torch.yml
conda activate weld-qc-gpu
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

如果最后一条能正常输出版本号和 `True`，就继续往下跑：

```powershell
python scripts\prepare_dataset.py --image-dir data\raw\images --label-dir data\raw\labels --output-dir data\dataset
python scripts\train_yolov8.py --config configs\app.yaml
```

## 19. 关于 PyCUDA

我把 `pycuda` 从环境创建阶段移出了，因为它在 Windows 上经常依赖本机 CUDA / Visual Studio / TensorRT 版本，容易导致整套环境创建失败。

只有在你准备跑 TensorRT 后端时，再单独安装它：

```powershell
conda activate weld-qc-gpu
python -m pip install pycuda==2024.1
```

如果这一步失败，不影响你先完成：

- YOLOv8 训练
- `.pt` 推理
- `.onnx` 推理
- 桌面端演示
- API 服务

只会影响 `.trt` / `.engine` 的 TensorRT 推理链路。
