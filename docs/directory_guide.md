# 目录与文件作用说明

下面按当前仓库实际结构说明每个目录和文件的职责。

## 根目录

- `.gitignore`
  - 忽略模型、运行日志、打包产物、测试临时文件和缓存目录。
- `README.md`
  - 项目总说明，包含系统能力、目录结构、训练/部署/运行入口。
- `焊缝缺陷2D视觉检测项目实施方案（从0到1落地）.docx`
  - 原始项目方案文档，是本仓库软件实现的需求来源。

## conda

- `conda/environment.cpu.yml`
  - CPU 版 `conda` 环境文件，适合接口联调、代码验证、CPU 推理。
- `conda/environment.gpu.yml`
  - GPU 版 `conda` 环境文件，适合训练、GPU 推理、桌面端演示和打包。

## artifacts

- `artifacts/README.md`
  - 说明模型、报告、日志、运行结果的存放规则。
- `artifacts/models/`
  - 预留给 `best.pt`、`best.onnx`、`best.trt` 等模型文件。
- `artifacts/reports/`
  - 预留给评估报告和验收报告。
- `artifacts/logs/`
  - 预留给训练日志和服务日志。
- `artifacts/runtime/`
  - 预留给运行时缺陷截图和 CSV 日志。

## configs

- `configs/app.yaml`
  - 全局业务配置，定义模型路径、后端类型、阈值、保存策略、API 端口、训练参数。
- `configs/dataset/weld.yaml`
  - YOLOv8 数据集配置文件，定义训练/验证图片位置和类别名。

## data

- `data/README.md`
  - 说明 `raw/` 和 `dataset/` 的目录约定。
- `data/raw/images/`
  - 预留放原始焊缝图片。
- `data/raw/labels/`
  - 预留放 LabelImg 导出的 YOLO 标签。
- `data/dataset/images/train|val/`
  - 预留放切分后的训练集/验证集图片。
- `data/dataset/labels/train|val/`
  - 预留放切分后的训练集/验证集标签。

## docs

- `docs/software_implementation.md`
  - 把原始方案里的软件部分重构成实际交付模块。
- `docs/deployment_guide.md`
  - 部署指南，说明模型文件、部署目录、运行方式和验收步骤。
- `docs/conda_environment.md`
  - 本项目的 `conda` 环境说明、安装顺序和包用途说明。
- `docs/directory_guide.md`
  - 当前这份目录与文件作用说明。

## requirements

- `requirements/base.txt`
  - 所有运行形态共用的最小依赖。
- `requirements/train.txt`
  - 训练、导出 ONNX 时需要的依赖。
- `requirements/runtime-gpu.txt`
  - GPU 运行时依赖，包含 GUI、API 和 ONNXRuntime GPU。
- `requirements/runtime-cpu.txt`
  - CPU 运行时依赖。
- `requirements/dev.txt`
  - 开发辅助依赖，主要是代码检查工具。

## scripts

- `scripts/prepare_dataset.py`
  - 将原始图片和标签切分成 YOLOv8 所需的 `train/val` 目录结构，并生成数据集 yaml。
- `scripts/train_yolov8.py`
  - 读取配置并调用 Ultralytics 训练 YOLOv8 模型。
- `scripts/evaluate.py`
  - 对训练好的模型做验证集评估，并输出 json / markdown 报告。
- `scripts/export_model.py`
  - 将 `.pt` 模型导出为 `onnx` 或 `engine`。
- `scripts/build_tensorrt.py`
  - 从 ONNX 构建 TensorRT 引擎，优先使用 `trtexec`，必要时回退到 Ultralytics。
- `scripts/run_desktop.py`
  - 启动 PySide6 桌面质检端。
- `scripts/run_api.py`
  - 启动 FastAPI 推理服务。
- `scripts/package_desktop.ps1`
  - 将桌面端打包成 exe。
- `scripts/package_api.ps1`
  - 将 API 服务打包成 exe。
- `scripts/__pycache__/`
  - Python 运行后自动生成的字节码缓存，不属于源码交付物。

## src/weld_inspector

- `src/weld_inspector/__init__.py`
  - 包初始化和版本号定义。
- `src/weld_inspector/schemas.py`
  - 定义检测结果数据结构，如 `Detection` 和 `FrameResult`。
- `src/weld_inspector/config.py`
  - 负责加载、保存和解析 `app.yaml` 配置。
- `src/weld_inspector/dataset.py`
  - 负责数据集切分、标签配对和 `weld.yaml` 生成。
- `src/weld_inspector/detector.py`
  - 核心检测引擎，统一封装模型推理、结果保存和 CSV 事件日志。
- `src/weld_inspector/api.py`
  - FastAPI 服务入口，提供健康检查、模型重载和图片推理接口。

## src/weld_inspector/camera

- `camera/__init__.py`
  - 相机模块初始化文件。
- `camera/hikrobot.py`
  - 海康工业相机接入预留扩展点，目前只定义接口说明，方便后续对接 MVS SDK。
- `camera/__pycache__/`
  - 自动生成缓存。

## src/weld_inspector/inference

- `inference/__init__.py`
  - 推理后端子包初始化文件。
- `inference/base.py`
  - 抽象推理后端基类。
- `inference/factory.py`
  - 按模型后缀或配置选择具体推理后端。
- `inference/ultralytics_backend.py`
  - `.pt` 模型推理后端。
- `inference/onnx_backend.py`
  - `.onnx` 模型推理后端。
- `inference/tensorrt_backend.py`
  - `.trt` / `.engine` 模型推理后端。
- `inference/__pycache__/`
  - 自动生成缓存。

## src/weld_inspector/ui

- `ui/__init__.py`
  - UI 子包初始化文件。
- `ui/main_window.py`
  - 桌面端主窗口，包含模型选择、图片/视频/摄像头检测、结果展示和状态更新。
- `ui/__pycache__/`
  - 自动生成缓存。

## src/weld_inspector/utils

- `utils/__init__.py`
  - 工具模块初始化文件。
- `utils/io.py`
  - 路径解析、目录创建、时间戳、文本/json 输出、图片文件遍历。
- `utils/logging.py`
  - 控制台和文件日志封装。
- `utils/vision.py`
  - 视觉预处理和后处理，包括 letterbox、YOLO 输出解码、NMS、绘框、图像编码。
- `utils/__pycache__/`
  - 自动生成缓存。

## tests

- `tests/test_config.py`
  - 测试配置文件加载、保存和后端推断。
- `tests/test_dataset.py`
  - 测试数据集切分和目录生成逻辑。
- `tests/__pycache__/`
  - 自动生成缓存。

## .tmp_tests

- `.tmp_tests/`
  - 我在本地跑单元测试时生成的临时测试数据目录，不属于正式交付物，已经加入忽略列表。
- `.tmp_tests/config_case/app.yaml`
  - 配置测试生成的临时文件。
- `.tmp_tests/dataset_case/...`
  - 数据集测试生成的临时图片、标签和 `weld.yaml`。
- 另外两个以 `tmp` 开头的临时子目录
  - 是 Windows 沙箱环境测试过程中残留的临时目录，也不属于项目源码。

## 如何理解这套结构

可以把它拆成 4 层：

- 文档与配置层：`README.md`、`docs/`、`configs/`
- 训练与导出层：`scripts/train_yolov8.py`、`scripts/export_model.py`、`scripts/build_tensorrt.py`
- 运行与部署层：`src/weld_inspector/`、`scripts/run_desktop.py`、`scripts/run_api.py`
- 数据与产物层：`data/`、`artifacts/`

如果后面你要继续扩展海康工业相机、PLC 报警、检测日志上报，优先改的就是：

- `src/weld_inspector/camera/hikrobot.py`
- `src/weld_inspector/detector.py`
- `src/weld_inspector/ui/main_window.py`
- `src/weld_inspector/api.py`
