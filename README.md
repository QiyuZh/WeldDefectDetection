# YOLOv8 焊缝质检系统

这是基于《焊缝缺陷2D视觉检测项目实施方案（从0到1落地）》整理并落地的软件实现，覆盖数据集整理、YOLOv8 训练、ONNX/TensorRT 导出、桌面质检端、HTTP 服务端、部署打包脚本和验收文档。

## 快速索引

- `conda` 完整安装与启动命令清单：`docs/conda_command_checklist.md`
- `conda` 环境说明：`docs/conda_environment.md`
- 灰度增强训练与预测流程：`docs/grayscale_workflow.md`
- Codex 本地记录迁移说明：`docs/codex_migration_guide.md`
- 新人接手 5 分钟速读版：`docs/directory_guide_quickstart.md`
- 详细目录说明：`docs/directory_guide.md`

## 系统能力

- 缺陷类别：`crack`、`hole`、`unwelded`、`offset_weld`
- 支持输入：图片、视频、摄像头，预留海康工业相机接入口
- 支持模型：`best.pt`、`best.onnx`、`best.trt` / `best.engine`
- 支持部署形态：
  - 桌面端 `exe` 质检程序
  - FastAPI 推理服务
  - 工控机本地离线运行
- 结果输出：
  - 实时框选与 `OK/NG` 判定
  - 检测事件 CSV 日志
  - 缺陷截图归档
  - 验收评估报告

## 目录结构

```text
WeldDefectDetection/
├─ configs/                 # 系统配置、数据集配置
├─ docs/                    # 软件方案整理、部署文档
├─ requirements/            # 训练/运行/开发依赖
├─ scripts/                 # 数据准备、训练、导出、打包脚本
├─ src/weld_inspector/      # 主程序代码
├─ tests/                   # 基础单元测试
├─ data/                    # 原始数据和 YOLO 数据集目录
└─ artifacts/               # 模型、日志、报告、运行结果
```

## 快速开始

### 1. 安装环境

训练环境：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements\train.txt
```

桌面端 / 服务端运行环境：

```powershell
pip install -r requirements\runtime-gpu.txt
```

如果只做 CPU 验证：

```powershell
pip install -r requirements\runtime-cpu.txt
```

### 2. 准备数据集

将原始图片和标注放入以下目录：

```text
data/raw/images
data/raw/labels
```

执行：

```powershell
python scripts\prepare_dataset.py --image-dir data\raw\images --label-dir data\raw\labels --output-dir data\dataset
```

### 3. 训练 YOLOv8

```powershell
python scripts\train_yolov8.py --config configs\app.yaml
```

### 3A. 可选：灰度增强数据集与训练

```powershell
python scripts\prepare_grayscale_dataset.py --dataset-dir data\datasets
python scripts\train_grayscale_yolov8.py
```

如果想先看灰度增强效果，再决定要不要训练：

```powershell
python scripts\compare_preprocess.py --config configs\app_grayscale.yaml --dataset-dir data\datasets --output-dir artifacts\preprocess_compare --limit 50
```

这条命令会自动遍历 `train / valid / test` 三个 split，并分别输出到 `artifacts\preprocess_compare\train|valid|test`。如果你只想看某一个 split，也可以改回 `--image-dir data\datasets\train\images` 这种单目录模式。

默认输出目录：

```text
artifacts/models/yolov8_train/
```

### 4. 导出模型

导出 ONNX：

```powershell
python scripts\export_model.py --weights artifacts\models\best.pt --format onnx
```

构建 TensorRT：

```powershell
python scripts\build_tensorrt.py --onnx artifacts\models\best.onnx --engine artifacts\models\best.trt


python scripts\export_model.py `
  --weights artifacts\models\best.pt `
  --format onnx `
  --imgsz 960 `
  --output artifacts\models\best_960.onnx


python scripts\export_model.py `
  --weights artifacts\models\best.pt `
  --format onnx `
  --imgsz 960 `
  --dynamic `
  --output artifacts\models\best_dynamic.onnx


conda activate weld-qc-gpu
python scripts\build_tensorrt.py `
  --onnx artifacts\models\best_960.onnx `
  --engine artifacts\models\best_960.trt `
  --imgsz 960 `
  --fp16


```

### 5. 运行桌面端

```powershell
python scripts\run_desktop.py --config configs\app.yaml
```

灰度增强流程可以直接使用：

```powershell
python scripts\run_desktop_grayscale.py
```

### 6. 启动 HTTP 服务

```powershell
python scripts\run_api.py --config configs\app.yaml
```

## 打包部署

- 桌面端打包：`powershell -ExecutionPolicy Bypass -File scripts\package_desktop.ps1`
- 服务端打包：`powershell -ExecutionPolicy Bypass -File scripts\package_api.ps1`

打包前请确认：

- `configs/app.yaml` 中的模型路径已指向实际权重
- TensorRT 运行机已安装对应 CUDA / TensorRT / 驱动
- `artifacts/models/` 内已存在 `best.pt` / `best.onnx` / `best.trt`

## 验收建议

- 精度：`mAP@0.5 >= 0.80`
- 速度：GPU 工控机 `>= 30 FPS`
- 稳定性：连续运行 1 小时不崩溃
- 易用性：`exe` 双击运行，结果显示清晰

## 说明

- 本项目代码默认遵循“优先数据和简单参数，不改网络结构”的实施原则。
- 当前仓库不附带真实焊缝数据和训练权重；落地时只需替换 `data/` 和 `artifacts/models/` 即可。
