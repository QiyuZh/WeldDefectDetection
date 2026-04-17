param(
  [string]$PythonExe
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$entry = Join-Path $PSScriptRoot "run_desktop.py"
$src = Join-Path $projectRoot "src"
$distRoot = Join-Path $projectRoot "dist"
$appName = "weld-inspector"
$appRoot = Join-Path $distRoot $appName

. (Join-Path $PSScriptRoot "package_common.ps1")
$pythonExe = Get-PackagingPython -PreferredPythonExe $PythonExe

$pyInstallerArgs = @(
  "-m", "PyInstaller",
  "--noconfirm",
  "--clean",
  "--windowed",
  "--name", $appName,
  "--paths", $src,
  "--collect-submodules", "onnxruntime",
  "--collect-submodules", "tensorrt",
  "--collect-submodules", "pycuda",
  "--collect-binaries", "onnxruntime",
  "--collect-binaries", "tensorrt",
  "--collect-binaries", "pycuda",
  "--hidden-import", "PySide6.QtSvg",
  "--hidden-import", "tensorrt.plugin",
  "--hidden-import", "pycuda.driver",
  "--hidden-import", "pycuda.autoinit",
  $entry
)

& $pythonExe @pyInstallerArgs
if ($LASTEXITCODE -ne 0) {
  throw "PyInstaller packaging failed with exit code $LASTEXITCODE."
}

Copy-DeployAssets -ProjectRoot $projectRoot -AppRoot $appRoot
Copy-TensorRTRuntime -AppRoot $appRoot

Write-Host "Desktop package created: $appRoot"
Write-Host "Using python: $pythonExe"
Write-Host "Copied configs, deploy models, data/datasets/data.yaml, and TensorRT runtime DLLs."
