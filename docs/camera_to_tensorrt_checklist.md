# 接相机到 TensorRT 部署实施清单

## 1. 目标

把当前项目从“图片 / 视频 / USB 摄像头验证”推进到：

1. 工业相机采图可用
2. ONNX 现场联调可用
3. TensorRT 实时部署可用

建议顺序不要跳：

1. 固定模型
2. 接相机
3. 跑 ONNX
4. 转 TensorRT

## 2. Phase A：冻结部署基线

### 目标

先冻结一版稳定模型，避免后面一边联调相机一边换模型。

### 文件

- `configs/app.yaml`
- `scripts/export_model.py`
- `docs/deployment_guide.md`

### 要做的事

- 固定当前最优权重 `best.pt`
- 导出固定 `960x960` 的 ONNX，命名为 `best_960.onnx`
- 在 `app.yaml` 中把部署基线改成 `best_960.onnx`

### 命令

```powershell
conda activate weld-qc-gpu
python scripts\export_model.py `
  --weights artifacts\models\best.pt `
  --format onnx `
  --imgsz 960 `
  --device 0 `
  --output artifacts\models\best_960.onnx
```

### 验收点

- `best_960.onnx` 能正常加载
- 输入 shape 为 `[1, 3, 960, 960]`
- 桌面端跑单图结果与 `best.pt` 基本一致

## 3. Phase B：接工业相机

### 目标

把当前的图片 / USB 摄像头链路，换成真实工业相机采图链路。

### 现状

当前仓库里海康相机还是占位类，尚未接入真实 SDK：

- `src/weld_inspector/camera/hikrobot.py`

### 重点文件

- `src/weld_inspector/camera/hikrobot.py`
- `src/weld_inspector/ui/main_window.py`
- `configs/app.yaml`

### 要做的事

#### `src/weld_inspector/camera/hikrobot.py`

- 接入海康 MVS SDK
- 实现设备枚举
- 实现打开设备
- 实现开始采图 / 取流
- 实现关闭和释放资源

#### `src/weld_inspector/ui/main_window.py`

- 增加工业相机模式入口
- 支持相机打开失败提示
- 支持断流后报错和恢复

#### `configs/app.yaml`

- 增加工业相机配置项，例如：
  - 相机类型
  - SDK 路径
  - 曝光
  - 增益
  - 触发模式

### 命令

相机 SDK 验证命令由厂家工具为主。项目内联调阶段，至少保留下面的启动命令：

```powershell
conda activate weld-qc-gpu
python scripts\run_desktop.py --config configs\app.yaml
```

### 验收点

- 能枚举到工业相机
- 能打开并持续取流
- 连续采图 10 分钟无崩溃
- 曝光 / 增益可调
- 相机断开后界面能报错，不无响应

## 4. Phase C：先用 ONNX 跑通现场链路

### 目标

在工业相机真实采图条件下，先用 ONNX 跑通整条链路。

### 重点文件

- `configs/app.yaml`
- `src/weld_inspector/detector.py`
- `src/weld_inspector/inference/onnx_backend.py`
- `src/weld_inspector/ui/main_window.py`

### 要做的事

- 固定 ONNX 输入为 `960`
- 检查首帧耗时和稳态耗时
- 检查现场图像与训练图像是否存在分布偏差
- 确认截图、CSV、日志能正常落盘

### 命令

```powershell
conda activate weld-qc-gpu
python scripts\run_desktop.py --config configs\app.yaml
```

如果要走 API 联调：

```powershell
conda activate weld-qc-gpu
python scripts\run_api.py --config configs\app.yaml
```

### 验收点

- 工业相机图像能被稳定送入 ONNX 后端
- 单图推理结果与离线验证差异可接受
- 连续推理时 FPS 稳定
- 首帧慢只出现一次，不影响后续运行

## 5. Phase D：构建 TensorRT 引擎

### 目标

把固定 `960x960` 的 ONNX 转成目标机可用的 TensorRT 引擎。

### 重点文件

- `scripts/build_tensorrt.py`
- `src/weld_inspector/inference/tensorrt_backend.py`
- `configs/app.yaml`

### 要做的事

#### `scripts/build_tensorrt.py`

- 在目标机上构建 `.trt`
- 明确 `imgsz=960`
- 明确 `fp16` 是否启用
- 兼容 TensorRT 10.x 的 `--memPoolSize` 新参数

#### `src/weld_inspector/inference/tensorrt_backend.py`

- 确认输入 shape 与 `960x960` 一致
- 确认输出 shape 与当前后处理兼容
- 确认推理结果能映射回原图

#### `configs/app.yaml`

- 把 `model.backend` 切换成 `tensorrt`
- 把 `model.model_path` 指向 `best.trt`

### 命令

#### 方式 A：用 `trtexec`

```powershell
conda activate weld-qc-gpu
python scripts\build_tensorrt.py `
  --onnx artifacts\models\best_960.onnx `
  --engine artifacts\models\best.trt `
  --imgsz 960 `
  --fp16
```

#### 方式 B：验证 `trtexec`

```powershell
trtexec --help
```

### 验收点

- `best.trt` 成功生成
- TensorRT 加载成功
- 相同图片下，TensorRT 与 ONNX 的检测结果基本一致
- TensorRT 稳态延迟显著低于 ONNX

## 6. Phase E：正式部署前硬化

### 目标

把“能跑”推进到“能长时间稳定跑”。

### 重点文件

- `src/weld_inspector/ui/main_window.py`
- `src/weld_inspector/api.py`
- `src/weld_inspector/utils/logging.py`
- `src/weld_inspector/detector.py`

### 要做的事

- 增加启动预热，去掉首帧超慢体验
- 增加相机断开重连策略
- 增加运行日志归档
- 增加运行截图 / CSV / 报告保留策略
- 明确 ONNX 回退开关

### 命令

桌面端回归：

```powershell
conda activate weld-qc-gpu
python scripts\run_desktop.py --config configs\app.yaml
```

服务端回归：

```powershell
conda activate weld-qc-gpu
python scripts\run_api.py --config configs\app.yaml
curl http://127.0.0.1:18080/health
```

### 验收点

- 连续运行 1 小时以上不崩溃
- 断相机 / 重连后系统能恢复
- 日志、截图、CSV 正常保存
- TensorRT 失败时能回退到 ONNX

## 7. 最终推荐文件落位

部署阶段建议最终固定如下：

- `artifacts/models/best.pt`
- `artifacts/models/best_960.onnx`
- `artifacts/models/best.trt`

`configs/app.yaml` 保留两套常用切换方式：

### ONNX 回退配置

```yaml
model:
  backend: onnx
  model_path: artifacts/models/best_960.onnx
  input_size: 960
```

### TensorRT 正式配置

```yaml
model:
  backend: tensorrt
  model_path: artifacts/models/best.trt
  input_size: 960
```

## 8. 一句话实施顺序

固定 ONNX -> 接工业相机 -> ONNX 联调 -> TensorRT 构建 -> 长稳验证。
