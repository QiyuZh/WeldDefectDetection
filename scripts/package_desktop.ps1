param(
  [string]$PythonExe,
  [string]$TensorRtHome
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
$tensorRtHome = Get-TensorRTHome -PreferredTensorRTHome $TensorRtHome

if ($tensorRtHome) {
  $tensorRtBin = Join-Path $tensorRtHome "bin"
  if (Test-Path -LiteralPath $tensorRtBin) {
    $env:TensorRT_HOME = $tensorRtHome
    $env:TRT_ROOT = $tensorRtHome
    $env:PATH = "$tensorRtBin;$env:PATH"
  }
}

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
Copy-CondaRuntime -AppRoot $appRoot -PythonExe $pythonExe
Copy-TensorRTRuntime -AppRoot $appRoot -TensorRTHome $tensorRtHome

Write-Host "Desktop package created: $appRoot"
Write-Host "Using python: $pythonExe"
Write-Host "TensorRT home: $tensorRtHome"
Write-Host "Copied configs, deploy models, data/datasets/data.yaml, conda runtime DLLs, and TensorRT runtime DLLs."
