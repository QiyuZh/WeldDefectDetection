$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$entry = Join-Path $PSScriptRoot "run_api.py"
$src = Join-Path $projectRoot "src"
$configs = Join-Path $projectRoot "configs"

python -m PyInstaller `
  --noconfirm `
  --clean `
  --console `
  --name weld-inspector-api `
  --paths $src `
  --add-data "$configs;configs" `
  --hidden-import onnxruntime `
  --hidden-import tensorrt `
  --hidden-import pycuda.autoinit `
  $entry

Write-Host "服务端打包完成，输出目录：dist\weld-inspector-api"

