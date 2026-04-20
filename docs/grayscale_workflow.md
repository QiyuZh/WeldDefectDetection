# 灰度增强训练与预测流程

本文档说明如何在保留原有彩色图训练/预测流程不变的前提下，新增一套可选的灰度增强流程。

## 1. 目标

- 原有彩色流程继续使用 `configs/app.yaml`
- 新增灰度流程使用 `configs/app_grayscale.yaml`
- 数据集预处理、训练模型、运行输出彼此隔离，避免互相覆盖

## 2. 两套流程的区别

### 彩色流程

- 数据集：`data/dataset`
- 配置：`configs/app.yaml`
- 训练输出：`artifacts/models/best.pt`
- 桌面端入口：`scripts/run_desktop.py`

### 灰度增强流程

- 数据集：`data/datasets_grayscale`
- 配置：`configs/app_grayscale.yaml`
- 训练输出：`artifacts/models/best_gray.pt`
- 桌面端入口：`scripts/run_desktop_grayscale.py`

## 3. 灰度增强算法

当前可选预处理模式为 `grayscale_weld`，处理步骤如下：

1. 彩色图转灰度图
2. 使用 CLAHE 增强局部对比度
3. 使用 unsharp mask 锐化焊缝缺陷边缘
4. 归一化到 `0~255`
5. 转回 3 通道 BGR，继续兼容 YOLO 输入

对应代码：

- `src/weld_inspector/preprocess.py`
- `src/weld_inspector/dataset.py`
- `src/weld_inspector/detector.py`

## 4. 配置项

灰度增强新增了两组配置：

- `dataset_preprocess`
- `inference_preprocess`

其中：

- `dataset_preprocess` 用于生成灰度增强后的训练集
- `inference_preprocess` 用于桌面端、API、实时推理时的输入增强

典型配置如下：

```yaml
dataset_preprocess:
  enabled: true
  mode: grayscale_weld
  clahe_clip_limit: 2.5
  clahe_tile_grid_size: 8
  blur_kernel_size: 3
  unsharp_amount: 1.0

inference_preprocess:
  enabled: true
  mode: grayscale_weld
  clahe_clip_limit: 2.5
  clahe_tile_grid_size: 8
  blur_kernel_size: 3
  unsharp_amount: 1.0
```

## 5. 灰度数据集准备

### 方式 A：直接使用灰度专用脚本

如果你当前已经是标准 YOLO 数据集结构，例如 `data/datasets/train|valid|test`，直接用：

```powershell
conda activate weld-qc-gpu
python scripts\prepare_grayscale_dataset.py --dataset-dir data\datasets
```

这个模式会：

- 读取 `data/datasets/data.yaml`
- 保留 `train / val / test` 的现有划分
- 只重写图像为灰度增强版本
- 直接复用原来的 `labels`

如果你手头还是原始图片目录和标签目录，再用：

```powershell
conda activate weld-qc-gpu
python scripts\prepare_grayscale_dataset.py --image-dir data\raw\images --label-dir data\raw\labels
```

该命令会自动：

- 输出到 `data/datasets_grayscale`
- 生成 `data/datasets_grayscale/data.yaml`
- 同步更新 `configs/dataset/weld_grayscale.yaml`
- 启用 `grayscale_weld` 预处理

### 方式 B：使用通用脚本手动启用灰度增强

```powershell
conda activate weld-qc-gpu
python scripts\prepare_dataset.py `
  --image-dir data\raw\images `
  --label-dir data\raw\labels `
  --output-dir data\datasets_grayscale `
  --dataset-config configs\dataset\weld_grayscale.yaml `
  --preprocess-enabled `
  --preprocess-mode grayscale_weld
```

## 6. 灰度模型训练

### 使用灰度专用训练入口

```powershell
conda activate weld-qc-gpu
python scripts\train_grayscale_yolov8.py
```

该命令默认读取：

- `configs/app_grayscale.yaml`
- `data/datasets_grayscale/data.yaml`

训练完成后，模型输出默认为：

- `artifacts/models/best_gray.pt`
- `artifacts/models/last_gray.pt`

### 使用通用训练脚本

```powershell
conda activate weld-qc-gpu
python scripts\train_yolov8.py --config configs/app_grayscale.yaml
```

## 7. 灰度模型预测

### 桌面端

```powershell
conda activate weld-qc-gpu
python scripts\run_desktop_grayscale.py
```

### API

```powershell
conda activate weld-qc-gpu
python scripts\run_api_grayscale.py
```

## 8. 桌面端灰度模式开关

桌面端现在新增了“推理模式”下拉框：

- `彩色原图`
- `灰度增强`

这个开关只控制 `inference_preprocess`，不会改动训练配置，也不会影响原有彩色数据集。

切换特点：

- 不需要重载模型即可生效
- 切换后会自动保存到当前使用的配置文件
- 如果当前打开的是 `configs/app.yaml`，它会保存彩色项目的推理模式
- 如果当前打开的是 `configs/app_grayscale.yaml`，它会保存灰度项目的推理模式

## 9. 什么时候适合用灰度增强

更适合的场景：

- 缺陷主要靠亮暗对比区分
- 焊缝颜色信息价值不高
- 背景颜色变化较大，但纹理边缘更重要

不一定更好的场景：

- 缺陷本身依赖颜色差异区分
- 原始彩色模型已经稳定
- 数据增强后的灰度图反而放大了噪声

建议做法：

1. 保留原有彩色模型作为基线
2. 单独训练灰度增强模型
3. 对同一批验证集比较 `PT / ONNX / TRT` 下的效果
4. 根据漏检、误检、连续纹理大框问题判断是否采用灰度模型

## 10. 批量导出彩色 / 灰度增强对比图

如果你想先肉眼比较增强效果，再决定是否训练，可以直接导出并排对比图：

```powershell
conda activate weld-qc-gpu
python scripts\compare_preprocess.py `
  --config configs/app_grayscale.yaml `
  --dataset-dir data\datasets `
  --output-dir artifacts\preprocess_compare `
  --limit 50
```

这条命令会自动遍历 `train / valid / test` 三个 split，并分别输出到：

- `artifacts\preprocess_compare\train`
- `artifacts\preprocess_compare\valid`
- `artifacts\preprocess_compare\test`

如果你只想看单个 split，也可以改成：

- `--image-dir data\datasets\train\images`
- `--image-dir data\datasets\valid\images`
- `--image-dir data\datasets\test\images`

输出结果会是同一张图的：

- 左侧：原始彩色图
- 右侧：灰度增强图

如果不想读取配置文件，也可以手动传参数：

```powershell
conda activate weld-qc-gpu
python scripts\compare_preprocess.py `
  --dataset-dir data\datasets `
  --output-dir artifacts\preprocess_compare `
  --preprocess-enabled `
  --preprocess-mode grayscale_weld `
  --clahe-clip-limit 2.5 `
  --clahe-tile-grid-size 8 `
  --blur-kernel-size 3 `
  --unsharp-amount 1.0
```

## 11. 当前推荐工作方式

- 彩色流程继续作为稳定基线
- 灰度流程用于并行实验和对照验证
- 两套流程分别维护数据集、模型和运行目录

这样可以保证：

- 不破坏现有彩色系统
- 可以快速比较灰度增强是否真的提升焊缝缺陷检测效果
