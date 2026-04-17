# 基于 Conda 的完整安装与启动命令清单

下面默认你已经切换到项目根目录，终端使用 PowerShell。

如果当前不在项目根目录，先执行：

```powershell
Set-Location <项目根目录>
```

## 1. 创建 GPU 环境

优先使用标准 GPU 环境：

```powershell
conda env create -f conda\environment.gpu.yml
conda activate weld-qc-gpu
```

如果遇到 `CondaHTTPError: HTTP 000`，说明 `pytorch` 的 conda 渠道访问失败，改用后备方案：

```powershell
conda env remove -n weld-qc-gpu -y
conda env create -f conda\environment.gpu-pip-torch.yml
conda activate weld-qc-gpu
```

## 2. 环境验证

```powershell
conda activate weld-qc-gpu
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -c "import cv2, yaml, ultralytics; print(cv2.__version__)"
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

如果准备构建 TensorRT，再额外验证：

```powershell
conda activate weld-qc-gpu
python -c "import tensorrt as trt; print(trt.__version__)"
python -c "import pycuda.driver as cuda; import pycuda.autoinit; print('pycuda ok')"
trtexec --help
```

## 3. 整理数据集

```powershell
conda activate weld-qc-gpu
python scripts\prepare_dataset.py `
  --image-dir data\raw\images `
  --label-dir data\raw\labels `
  --output-dir data\dataset
```

## 4. 训练模型

按 `configs/app.yaml` 训练：

```powershell
conda activate weld-qc-gpu
python scripts\train_yolov8.py --config configs\app.yaml
```

临时覆盖参数：

```powershell
conda activate weld-qc-gpu
python scripts\train_yolov8.py `
  --config configs\app.yaml `
  --epochs 150 `
  --imgsz 960 `
  --batch 4 `
  --device 0
```

## 5. 评估模型

```powershell
conda activate weld-qc-gpu
python scripts\evaluate.py `
  --config configs\app.yaml `
  --weights artifacts\models\best.pt
```

## 6. 导出固定 `960x960` 的 ONNX

这是推荐的现场联调和 TensorRT 基线：

```powershell
conda activate weld-qc-gpu
python scripts\export_model.py `
  --weights artifacts\models\best.pt `
  --format onnx `
  --imgsz 960 `
  --device 0 `
  --output artifacts\models\best_960.onnx
```

## 7. 导出动态 ONNX

这是推荐的算法验证和输入尺寸对比基线：

```powershell
conda activate weld-qc-gpu
python scripts\export_model.py `
  --weights artifacts\models\best.pt `
  --format onnx `
  --imgsz 960 `
  --dynamic `
  --device 0 `
  --output artifacts\models\best_dynamic.onnx
```

## 8. 验证 ONNX 的真实输入尺寸

查看输入 shape：

```powershell
conda activate weld-qc-gpu
python -c "import onnxruntime as ort; s=ort.InferenceSession('artifacts/models/best_960.onnx', providers=['CPUExecutionProvider']); print(s.get_inputs()[0].shape)"
```

查看元数据：

```powershell
conda activate weld-qc-gpu
python -c "import onnxruntime as ort; s=ort.InferenceSession('artifacts/models/best_960.onnx', providers=['CPUExecutionProvider']); print(s.get_modelmeta().custom_metadata_map)"
```

解释：

- 固定 ONNX：输入 shape 预期为 `[1, 3, 960, 960]`
- 动态 ONNX：输入 shape 可能类似 `[1, 3, 'height', 'width']`

## 9. 切换 `app.yaml`

### 固定 `960 ONNX`

```yaml
model:
  backend: onnx
  model_path: artifacts/models/best_960.onnx
  input_size: 960
```

### 动态 ONNX

```yaml
model:
  backend: onnx
  model_path: artifacts/models/best_dynamic.onnx
  input_size: 960
```

注意：

- 固定 ONNX 不会因为你改 `app.yaml` 就改变模型真实输入尺寸
- 动态 ONNX 才会在运行时读取 `model.input_size`

## 10. 启动桌面端

```powershell
conda activate weld-qc-gpu
python scripts\run_desktop.py --config configs\app.yaml
```

## 11. 启动 HTTP 服务

```powershell
conda activate weld-qc-gpu
python scripts\run_api.py --config configs\app.yaml
```

健康检查：

```powershell
curl http://127.0.0.1:18080/health
```

## 12. 构建 TensorRT 引擎

说明：

- 当前项目里的 `scripts/build_tensorrt.py` 已兼容 TensorRT 新旧版 `trtexec`
- 对 TensorRT 10.x，脚本会自动改用 `--memPoolSize=workspace:...`

### 方式 A：本机已安装 `trtexec`

```powershell
conda activate weld-qc-gpu
python scripts\build_tensorrt.py `
  --onnx artifacts\models\best_960.onnx `
  --engine artifacts\models\best.trt `
  --imgsz 960 `
  --fp16
```

### 方式 B：没有 `trtexec`，回退到 Ultralytics 导出

```powershell
conda activate weld-qc-gpu
python scripts\build_tensorrt.py `
  --onnx artifacts\models\best_960.onnx `
  --engine artifacts\models\best.trt `
  --weights artifacts\models\best.pt `
  --imgsz 960 `
  --fp16
```

## 13. 切换到 TensorRT 运行

```yaml
model:
  backend: tensorrt
  model_path: artifacts/models/best.trt
  input_size: 960
```

启动命令：

```powershell
conda activate weld-qc-gpu
python scripts\run_desktop.py --config configs\app.yaml
```

## 14. 打包桌面端和 API

```powershell
conda activate weld-qc-gpu
powershell -ExecutionPolicy Bypass -File scripts\package_desktop.ps1
powershell -ExecutionPolicy Bypass -File scripts\package_api.ps1
```

## 15. 常用最短命令流

### 日常训练

```powershell
conda activate weld-qc-gpu
python scripts\train_yolov8.py --config configs\app.yaml
```

### 日常桌面端验证

```powershell
conda activate weld-qc-gpu
python scripts\run_desktop.py --config configs\app.yaml
```

### 日常 API 验证

```powershell
conda activate weld-qc-gpu
python scripts\run_api.py --config configs\app.yaml
```

## 16. 当前推荐的完整链路

1. `conda env create`
2. `prepare_dataset.py`
3. `train_yolov8.py`
4. `evaluate.py`
5. `export_model.py`
6. `run_desktop.py` 或 `run_api.py`
7. `build_tensorrt.py`
8. 再切到 `tensorrt` 后端复测

如果你已经进入部署阶段，建议固定成：

1. `best.pt`
2. `best_960.onnx`
3. `best.trt`

## 17. 可交付版打包说明

当前 `scripts/package_desktop.ps1` 和 `scripts/package_api.ps1` 已经不是只打一个裸 `exe`。
打包完成后，`dist/weld-inspector` 或 `dist/weld-inspector-api` 下会额外同步这些内容：

- `configs/`
- `artifacts/models/` 下的 `*.pt` / `*.onnx` / `*.trt` / `*.engine`
- `data/datasets/data.yaml`
- TensorRT 运行时 DLL

直接执行：

```powershell
conda activate weld-qc-gpu
powershell -ExecutionPolicy Bypass -File scripts\package_desktop.ps1
powershell -ExecutionPolicy Bypass -File scripts\package_api.ps1
```

生成后的目录就是交付目录，后续请整体拷贝 `dist/weld-inspector` 或 `dist/weld-inspector-api`，不要只拷贝单个 `exe`。
