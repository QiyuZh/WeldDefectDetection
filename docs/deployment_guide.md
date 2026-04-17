# 部署指南

## 1. 推荐的交付形态

建议把模型产物分成 4 类管理：

- `best.pt`：训练基线和最终精度参考，不直接作为正式部署产物
- `best_960.onnx`：固定 `960x960` 的 ONNX，推荐作为现场联调和正式部署基线
- `best_dynamic.onnx`：动态输入 ONNX，适合算法验证、不同输入尺寸的快速试验
- `best.trt`：目标部署机上生成的 TensorRT 引擎，作为最终实时运行产物

推荐的推进顺序：

1. `best.pt` 训练和验证通过
2. 导出固定 `960x960` 的 ONNX 做现场联调
3. 相机链路稳定后，在目标机上构建 `best.trt`
4. 最终部署时优先跑 TensorRT，保留 ONNX 作为回退后端

## 2. 固定 ONNX 与动态 ONNX

### 固定 ONNX

固定 ONNX 指导出时就把输入尺寸写死在图结构里，例如 `960x960`。  
优点是部署稳定、行为可预期，也更适合后续转 TensorRT。

推荐用途：

- 车间试运行
- 相机联调
- TensorRT 构建输入基线

### 动态 ONNX

动态 ONNX 指导出时允许输入尺寸变化，例如后续可以在 `app.yaml` 中把 `input_size` 从 `960` 改为 `1280`。  
优点是灵活，适合做尺寸对比实验；缺点是正式部署时不如固定 ONNX 稳。

推荐用途：

- 模型验证
- 推理尺寸对比
- 算法调参阶段

## 3. 当前项目里的运行逻辑

当前 ONNX 后端会先读取模型真实输入 shape，再决定实际预处理尺寸：

- 如果 ONNX 是固定输入，例如 `[1, 3, 640, 640]`，运行时会按模型真实尺寸走
- 如果 `app.yaml` 里写了 `960`，但 ONNX 仍然是固定 `640`，系统会自动按 `640` 预处理，并给出提示
- 如果 ONNX 是动态输入，例如 `[1, 3, height, width]`，运行时才会使用 `app.yaml -> model.input_size`

这意味着：

- 改 `app.yaml` 不能把已经导出的固定 ONNX 从 `640` 变成 `960`
- 真要按新尺寸推理，必须重新导出固定 ONNX，或者导出动态 ONNX

## 4. 导出命令

### 导出固定 `960x960` 的 ONNX

```powershell
conda activate weld-qc-gpu
python scripts\export_model.py `
  --weights artifacts\models\best.pt `
  --format onnx `
  --imgsz 960 `
  --device 0 `
  --output artifacts\models\best_960.onnx
```

### 导出动态 ONNX

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

## 5. `app.yaml` 配置建议

### 正式部署推荐：固定 `960x960` ONNX

```yaml
model:
  backend: onnx
  model_path: artifacts/models/best_960.onnx
  input_size: 960
```

### 算法验证推荐：动态 ONNX

```yaml
model:
  backend: onnx
  model_path: artifacts/models/best_dynamic.onnx
  input_size: 960
```

## 6. 如何验证导出是否正确

### 查看 ONNX 输入 shape

```powershell
conda activate weld-qc-gpu
python -c "import onnxruntime as ort; s=ort.InferenceSession('artifacts/models/best_960.onnx', providers=['CPUExecutionProvider']); print(s.get_inputs()[0].shape)"
```

预期结果：

- 固定 ONNX：`[1, 3, 960, 960]`
- 动态 ONNX：类似 `[1, 3, 'height', 'width']`

### 查看 ONNX 元数据里的 `imgsz`

```powershell
conda activate weld-qc-gpu
python -c "import onnxruntime as ort; s=ort.InferenceSession('artifacts/models/best_960.onnx', providers=['CPUExecutionProvider']); print(s.get_modelmeta().custom_metadata_map)"
```

## 7. 关于首帧慢

固定 ONNX 可以减少部署时的不确定性，但不能消除首帧预热开销。  
首帧慢主要来自：

- ONNX Runtime Session 初始化
- CUDA provider 首次建图
- 首次显存分配
- 第一次真正执行推理

因此：

- 固定 `960 ONNX`：推荐
- 启动预热：仍然推荐

## 8. TensorRT 部署建议

TensorRT 建议在目标机上基于固定 `960x960` 的 ONNX 构建，不建议跨 CUDA / TensorRT 版本直接复用 `.trt`。

推荐链路：

1. `best.pt`
2. `best_960.onnx`
3. `best.trt`

如果现场需要快速回退，可在 `app.yaml` 中把：

- `model.backend` 切回 `onnx`
- `model.model_path` 切回 `artifacts/models/best_960.onnx`
