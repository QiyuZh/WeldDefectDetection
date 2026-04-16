# 部署指南

## 1. 目标环境

- Windows 10 / 11
- NVIDIA GPU 工控机
- Python 3.10+
- 已安装显卡驱动

## 2. 目录准备

部署目录建议如下：

```text
weld-qc/
├─ configs/
├─ models/
├─ runtime/
└─ weld-inspector.exe
```

## 3. 模型文件

将以下文件放入 `models/`：

- `best.pt`：训练阶段使用
- `best.onnx`：跨框架验证
- `best.trt`：工控机实时检测首选

## 4. 运行方式

### 桌面端

1. 修改 `configs/app.yaml` 中的模型路径。
2. 双击 `weld-inspector.exe`。
3. 选择图片、视频或摄像头。
4. 观察 `OK/NG`、FPS 和缺陷列表。

### 服务端

1. 启动 `weld-inspector-api.exe`
2. 调用：

```http
GET /health
POST /infer-image
POST /reload-model
```

## 5. TensorRT 说明

- 若 `best.trt` 与目标机器 CUDA / TensorRT 版本不匹配，需要在目标机重新执行 `scripts/build_tensorrt.py`
- TensorRT 引擎建议在最终部署机上生成，不建议跨 GPU 型号直接复用

## 6. 验收步骤

1. 用未参与训练的图片集做离线抽检
2. 用目标工位视频做实时验证
3. 连续运行 1 小时观察稳定性
4. 导出日志和缺陷截图作为验收附件

