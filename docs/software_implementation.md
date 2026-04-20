# 软件部分整理

本项目当前的软件实现可以分成 6 个交付模块。

## 1. 数据准备

- 原始数据目录：`data/raw/images`、`data/raw/labels`
- 训练数据目录：`data/dataset/images/{train,val}`、`data/dataset/labels/{train,val}`
- 数据整理脚本：`scripts/prepare_dataset.py`

当前还新增了一条可选灰度增强链路：

- 灰度训练集目录：`data/dataset_grayscale`
- 灰度数据整理脚本：`scripts/prepare_grayscale_dataset.py`
- 灰度预处理模块：`src/weld_inspector/preprocess.py`
- 灰度数据集配置：`configs/dataset/weld_grayscale.yaml`

职责：

- 检查图像和标签是否成对存在
- 划分训练集和验证集
- 生成训练用数据集 YAML

## 2. 训练与评估

- 训练脚本：`scripts/train_yolov8.py`
- 评估脚本：`scripts/evaluate.py`
- 训练配置来源：`configs/app.yaml -> training`

灰度增强训练采用独立入口和独立输出：

- 灰度训练脚本：`scripts/train_grayscale_yolov8.py`
- 灰度训练配置：`configs/app_grayscale.yaml -> training`
- 灰度模型默认输出：`artifacts/models/best_gray.pt`

当前策略：

- 优先通过模型规模、输入尺寸、训练轮数调优
- 保持主干训练链路简单，避免过早引入复杂结构改造
- 当前已经验证过 `yolov8s + 960` 相比 `yolov8n + 640` 有明显提升

## 3. ONNX / TensorRT 导出

- ONNX / TensorRT 导出脚本：`scripts/export_model.py`
- TensorRT 构建脚本：`scripts/build_tensorrt.py`
- 模型产物目录：`artifacts/models/`

当前支持 4 类部署产物：

- `best.pt`
- `best_960.onnx`
- `best_dynamic.onnx`
- `best.trt`

### 固定 ONNX

固定 ONNX 更适合部署。  
例如导出固定 `960x960` 的 ONNX 后，运行时会始终按模型真实尺寸走。

### 动态 ONNX

动态 ONNX 更适合调参。  
运行时可以通过 `configs/app.yaml -> model.input_size` 控制推理尺寸，但正式部署时一般仍建议回到固定 ONNX。

### 当前后端行为

项目里的 ONNX 后端已经支持：

- 自动读取 ONNX 的真实输入 shape
- 固定 ONNX 优先使用模型真实输入尺寸
- 动态 ONNX 使用 `app.yaml` 中的 `input_size`

这避免了“模型还是固定 640，但配置改成 960 后直接报维度错”的问题。

## 4. 推理与界面

- 桌面端：`scripts/run_desktop.py`
- HTTP 服务：`scripts/run_api.py`
- 推理引擎：`src/weld_inspector/detector.py`
- 多后端推理：`src/weld_inspector/inference/`

灰度增强推理支持独立入口：

- 桌面端：`scripts/run_desktop_grayscale.py`
- HTTP 服务：`scripts/run_api_grayscale.py`

当前支持后端：

- `ultralytics`
- `onnx`
- `tensorrt`

当前支持输入：

- 单张图片
- 视频文件
- USB 摄像头

当前支持输出：

- 结果叠加图
- OK / NG 判定
- 截图保存
- CSV 日志

桌面端当前还新增了“推理模式”切换：

- `彩色原图`
- `灰度增强`

这个开关只作用于 `inference_preprocess`，不会改动原有彩色训练集和训练模型。

## 5. 工业相机扩展点

- 工业相机入口：`src/weld_inspector/camera/hikrobot.py`

当前状态：

- 已预留海康相机扩展类
- 仍未接入真实 MVS SDK
- 当前正式可用输入链路仍是图片、视频和 USB 摄像头

这意味着项目现在已经具备：

- Phase 1 完整能力
- Phase 2 的一部分基础能力

但还不算真正的产线工业版。

## 6. 推荐的部署路径

建议按下面顺序推进：

1. 使用 `best.pt` 完成训练和评估
2. 导出固定 `960x960` 的 ONNX，作为现场联调基线
3. 接入工业相机，先用 ONNX 跑通真实采图链路
4. 在目标机上基于固定 ONNX 构建 TensorRT 引擎
5. 最终用 TensorRT 作为部署后端，保留 ONNX 作为回退路径
