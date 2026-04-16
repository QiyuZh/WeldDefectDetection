# 软件部分整理

本项目的软件实现按原方案拆为 5 个交付模块：

## 1. 数据采集与标注

- 原始数据目录：`data/raw/images`、`data/raw/labels`
- 训练数据目录：`data/dataset/images/{train,val}`、`data/dataset/labels/{train,val}`
- 脚本：`scripts/prepare_dataset.py`
- 目标：
  - 检查图片和标签是否成对存在
  - 按 8:2 划分训练集和验证集
  - 自动生成 `configs/dataset/weld.yaml`

## 2. YOLOv8 训练与评估

- 训练脚本：`scripts/train_yolov8.py`
- 验证脚本：`scripts/evaluate.py`
- 训练配置来源：`configs/app.yaml -> training`
- 保持默认策略：
  - 不改网络结构
  - 不改损失函数
  - 优先通过数据质量、输入尺寸、训练轮数调整效果

## 3. ONNX / TensorRT 导出

- 导出脚本：`scripts/export_model.py`
- TensorRT 构建脚本：`scripts/build_tensorrt.py`
- 产物目录：`artifacts/models/`
- 支持：
  - `best.pt`
  - `best.onnx`
  - `best.trt`

## 4. 检测与部署

- 推理后端：
  - `UltralyticsBackend`
  - `OnnxRuntimeBackend`
  - `TensorRTBackend`
- 桌面端：`src/weld_inspector/ui/main_window.py`
- 服务端：`src/weld_inspector/api.py`
- 支持：
  - 图片检测
  - 视频流检测
  - 摄像头检测
  - 结果框选、FPS、OK/NG 判定
  - 缺陷截图和日志落盘

## 5. 打包与交付

- 桌面端打包：`scripts/package_desktop.ps1`
- 服务端打包：`scripts/package_api.ps1`
- 验收口径：
  - 精度、速度、稳定性、易用性

