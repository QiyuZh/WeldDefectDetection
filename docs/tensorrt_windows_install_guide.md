# Windows 下 TensorRT 安装指南

## 1. 适用场景

这份文档适用于当前项目在 Windows 上部署 TensorRT。

推荐用途：

- 在 Windows 工位机上构建 `.trt` 引擎
- 在 `conda` 环境中运行 Python 版 TensorRT 后端
- 为后续 Phase 2 / Phase 3 部署做环境准备

## 2. 当前项目建议的版本选择

如果你的环境与当前项目一致：

- GPU 驱动支持 CUDA 12.x
- 本机 CUDA Toolkit 为 12.6
- Python 为 3.10

则建议：

- 平台：`Windows x64`
- TensorRT：`10.16.1.x`
- CUDA：选择 `12.x`
- Python：`3.10`

不要选 `CUDA 13.x`。

## 3. 官方入口

- 下载页：[NVIDIA TensorRT Download](https://developer.nvidia.com/tensorrt)
- 安装文档：[TensorRT Installing Guide](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html)

## 4. 安装前检查

先在终端里确认这 3 项：

```powershell
nvidia-smi
python --version
nvcc --version
```

当前项目建议至少满足：

- `nvidia-smi` 正常输出
- Python 为 `3.10`
- CUDA Toolkit 为 `12.x`

## 5. 下载 TensorRT 压缩包

在下载页中按下面思路选择：

1. 进入 TensorRT 下载页
2. 选择 `TensorRT 10.16.1.x`
3. 平台选 `Windows x64`
4. CUDA 选 `12.x`
5. 包类型选 `zip`

## 6. 解压目录建议

建议解压到固定目录，例如：

```text
C:\Tools\TensorRT-10.16.1.11
```

下面命令都以这个目录为示例。

## 7. 当前终端临时加入 PATH

先把 `trtexec` 跑通，再做后面的 Python 安装：

```powershell
$env:TRT_ROOT = "C:\Tools\TensorRT-10.16.1.11"
$env:PATH = "$env:TRT_ROOT\bin;$env:PATH"
Get-Command trtexec
trtexec --help
```

如果 `trtexec --help` 能正常输出，说明 zip 解压和当前终端 PATH 已生效。

## 8. 永久写入用户环境变量

如果你希望重开终端后也能直接用 `trtexec`：

```powershell
[Environment]::SetEnvironmentVariable("TensorRT_HOME", "C:\Tools\TensorRT-10.16.1.11", "User")
$old = [Environment]::GetEnvironmentVariable("Path", "User")
[Environment]::SetEnvironmentVariable("Path", "$old;C:\Tools\TensorRT-10.16.1.11\bin", "User")
```

重开 PowerShell 后验证：

```powershell
trtexec --help
```

## 9. 激活 conda 环境

```powershell
conda activate weld-qc-gpu
python -m pip install --upgrade pip wheel setuptools
python -c "import sys; print(sys.executable)"
python -m pip --version
```

确保当前 Python 和 pip 都属于 `weld-qc-gpu` 环境。

## 10. 安装 TensorRT Python 包

优先使用 zip 包自带的 wheel，这样版本最一致。

```powershell
$env:TRT_ROOT = "C:\Tools\TensorRT-10.16.1.11"
$whl = Get-ChildItem "$env:TRT_ROOT\python\tensorrt-*-cp310-none-win_amd64.whl" | Select-Object -First 1 -ExpandProperty FullName
python -m pip install $whl
```

验证：

```powershell
python -c "import tensorrt as trt; print(trt.__version__); assert trt.Builder(trt.Logger())"
```

如果能打印出类似 `10.16.1.11`，说明 TensorRT Python 包已可用。

## 11. 可选方案：直接用 pip 安装 `tensorrt-cu12`

如果你不想从 zip 目录安装 wheel，也可以走 pip，但要显式指定 `cu12`：

```powershell
conda activate weld-qc-gpu
python -m pip install --upgrade tensorrt-cu12
```

不建议直接无脑安装不带 CUDA 后缀的 `tensorrt` 包。

## 12. 安装 `pycuda`

当前项目里的 Python TensorRT 后端还依赖 `pycuda`。

直接尝试：

```powershell
conda activate weld-qc-gpu
python -m pip install pycuda
```

验证：

```powershell
python -c "import pycuda.driver as cuda; import pycuda.autoinit; print('pycuda ok')"
```

## 13. 如果 `pycuda` 安装失败

Windows 下 `pycuda` 失败通常不是 TensorRT 本身的问题，而是本机编译链问题。

先检查：

```powershell
cl
echo $env:CUDA_PATH
```

### 如果 `cl` 不存在

安装：

- `Visual Studio Build Tools 2022`
- 勾选 `Desktop development with C++`

### 如果 `CUDA_PATH` 为空

按你本机 CUDA 12.6 示例设置：

```powershell
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
```

然后重试：

```powershell
conda activate weld-qc-gpu
python -m pip install pycuda
```

## 14. 最终总验证

全部装完以后，执行：

```powershell
conda activate weld-qc-gpu
trtexec --help
python -c "import tensorrt as trt; print('tensorrt', trt.__version__)"
python -c "import pycuda.driver as cuda; import pycuda.autoinit; print('pycuda ok')"
```

如果这三步都通过，说明：

- TensorRT 命令行工具可用
- TensorRT Python 包可用
- PyCUDA 可用

## 15. 构建当前项目的 TensorRT 引擎

以当前项目推荐的固定 `960x960 ONNX` 为例：

说明：

- 当前项目里的 `scripts/build_tensorrt.py` 已兼容新旧 `trtexec`
- 对 TensorRT 10.x，会自动使用 `--memPoolSize=workspace:...`
- 如果看到 `Unknown option: --workspace`，说明你用的是较新的 `trtexec`，更新脚本后再执行即可

```powershell
conda activate weld-qc-gpu
python scripts\build_tensorrt.py `
  --onnx artifacts\models\best_960.onnx `
  --engine artifacts\models\best.trt `
  --imgsz 960 `
  --fp16
```

## 16. 切换到 TensorRT 运行

在 `configs/app.yaml` 中改成：

```yaml
model:
  backend: tensorrt
  model_path: artifacts/models/best.trt
  input_size: 960
```

然后启动桌面端：

```powershell
conda activate weld-qc-gpu
python scripts\run_desktop.py --config configs\app.yaml
```

## 17. 推荐的部署顺序

建议不要跳步骤，按下面顺序走：

1. 训练并确认 `best.pt`
2. 导出 `best_960.onnx`
3. 验证 ONNX 与 PT 基本一致
4. 安装 TensorRT 环境
5. 构建 `best.trt`
6. 对比 ONNX 与 TensorRT 的结果和速度
7. 再切换正式部署后端

## 18. 一句话总结

Windows 下 TensorRT 的核心步骤是：

下载 zip -> 解压 -> 配 PATH -> 跑通 `trtexec` -> 安装 TensorRT Python wheel -> 安装 `pycuda` -> 构建 `.trt`。
