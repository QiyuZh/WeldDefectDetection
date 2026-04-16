$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$entry = Join-Path $PSScriptRoot "run_desktop.py"
$src = Join-Path $projectRoot "src"
$configs = Join-Path $projectRoot "configs"

python -m PyInstaller `
  --noconfirm `
  --clean `
  --windowed `
  --name weld-inspector `
  --paths $src `
  --add-data "$configs;configs" `
  --hidden-import PySide6.QtSvg `
  --hidden-import onnxruntime `
  --hidden-import tensorrt `
  --hidden-import pycuda.autoinit `
  $entry

Write-Host "桌面端打包完成，输出目录：dist\weld-inspector"

