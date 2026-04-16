# 新人接手 5 分钟速读版

如果你刚接手这个项目，不要先把整个仓库从头翻到尾。先只看下面这几层。

## 1. 这个项目是干什么的

这是一个基于 YOLOv8 的焊缝缺陷质检系统，目标是完成：

1. 数据整理
2. 模型训练
3. ONNX / TensorRT 导出
4. 桌面端检测
5. API 服务部署
6. exe 打包交付

缺陷类别固定为 4 类：

- `crack`
- `hole`
- `unwelded`
- `offset_weld`

## 2. 先看哪里

你只要先看这 5 个位置：

1. `README.md`
2. `configs/app.yaml`
3. `scripts/`
4. `src/weld_inspector/`
5. `docs/conda_command_checklist.md`

够了。别先钻 `__pycache__`、测试临时目录和一堆细枝末节。

## 3. 最核心的文件

### 配置入口

- `configs/app.yaml`
  - 这是系统总开关。
  - 模型路径、推理后端、置信度阈值、API 端口、训练参数都在这里。

### 训练与导出入口

- `scripts/prepare_dataset.py`
  - 把原始图片和标注整理成 YOLOv8 可训练的数据集。
- `scripts/train_yolov8.py`
  - 训练模型，产出 `best.pt`。
- `scripts/evaluate.py`
  - 评估模型效果。
- `scripts/export_model.py`
  - 导出 `best.onnx` 或 `engine`。
- `scripts/build_tensorrt.py`
  - 从 ONNX 构建 `best.trt`。

### 运行入口

- `scripts/run_desktop.py`
  - 启动桌面端质检程序。
- `scripts/run_api.py`
  - 启动 HTTP 推理服务。

### 打包入口

- `scripts/package_desktop.ps1`
  - 打包桌面端 exe。
- `scripts/package_api.ps1`
  - 打包 API exe。

## 4. 主程序代码怎么分

`src/weld_inspector/` 可以粗暴理解成 6 块：

- `config.py`
  - 负责读写配置。
- `dataset.py`
  - 负责整理数据集。
- `detector.py`
  - 负责统一调用推理、保存结果和写日志。
- `inference/`
  - 真正的推理后端。
  - `ultralytics_backend.py` 处理 `.pt`
  - `onnx_backend.py` 处理 `.onnx`
  - `tensorrt_backend.py` 处理 `.trt` / `.engine`
- `ui/main_window.py`
  - 桌面端主界面。
- `api.py`
  - FastAPI 服务入口。

## 5. 数据、模型、结果分别放哪

### 数据

- `data/raw/images`
  - 原始图片
- `data/raw/labels`
  - 原始标签
- `data/dataset/`
  - 处理后的训练/验证集

### 模型

- `artifacts/models/best.pt`
- `artifacts/models/best.onnx`
- `artifacts/models/best.trt`

### 结果

- `artifacts/reports/`
  - 评估报告
- `artifacts/logs/`
  - 运行日志
- `artifacts/runtime/`
  - 检测截图、CSV 日志

## 6. 一条最短工作链

如果你只是想把项目跑起来，顺序只有这几步：

1. 创建 `conda` 环境
2. 放入 `data/raw/images` 和 `data/raw/labels`
3. 跑 `prepare_dataset.py`
4. 跑 `train_yolov8.py`
5. 跑 `export_model.py`
6. 跑 `build_tensorrt.py`
7. 跑 `run_desktop.py`
8. 最后跑 `package_desktop.ps1`

## 7. 你最容易改的地方

如果后面要扩展业务，通常改这几个点：

- 改模型和阈值：`configs/app.yaml`
- 改训练流程：`scripts/train_yolov8.py`
- 改推理逻辑：`src/weld_inspector/detector.py`
- 改桌面端显示：`src/weld_inspector/ui/main_window.py`
- 改接口协议：`src/weld_inspector/api.py`
- 接海康工业相机：`src/weld_inspector/camera/hikrobot.py`

## 8. 哪些东西先别碰

- `__pycache__/`
  - 这是 Python 自动生成缓存。
- `.tmp_tests/`
  - 这是测试临时目录，不是业务代码。
- `tests/`
  - 有空再看，先把主链跑通。

## 9. 真正接手时的建议顺序

1. 先读 `docs/conda_command_checklist.md`
2. 再看 `configs/app.yaml`
3. 跑一次 `run_desktop.py`
4. 再去看 `src/weld_inspector/`
5. 最后再看详细版 `docs/directory_guide.md`

这样 5 分钟内你至少知道：

- 项目入口在哪
- 主链怎么跑
- 出问题优先查哪里
- 哪些文件重要，哪些文件可以先忽略
