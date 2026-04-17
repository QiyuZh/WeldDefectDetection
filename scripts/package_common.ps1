Set-StrictMode -Version Latest

function Get-PackagingPython {
    param(
        [string]$PreferredPythonExe
    )

    if ($PreferredPythonExe -and (Test-Path -LiteralPath $PreferredPythonExe)) {
        return $PreferredPythonExe
    }

    if ($env:PACKAGING_PYTHON_EXE -and (Test-Path -LiteralPath $env:PACKAGING_PYTHON_EXE)) {
        return $env:PACKAGING_PYTHON_EXE
    }

    if ($env:CONDA_PREFIX) {
        $condaPython = Join-Path $env:CONDA_PREFIX "python.exe"
        if (Test-Path -LiteralPath $condaPython) {
            return $condaPython
        }
    }

    $wherePython = @(where.exe python 2>$null | Where-Object { $_ -and (Test-Path -LiteralPath $_) })
    $envPython = $wherePython | Where-Object { $_ -match "\\\\.conda\\\\envs\\\\" -or $_ -match "\\\\envs\\\\" } | Select-Object -First 1
    if ($envPython) {
        return $envPython
    }

    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCommand) {
        return $pythonCommand.Source
    }

    throw "python.exe was not found. Activate the target environment first."
}

function Copy-DeployDirectory {
    param(
        [Parameter(Mandatory = $true)][string]$Source,
        [Parameter(Mandatory = $true)][string]$Target
    )

    if (-not (Test-Path -LiteralPath $Source)) {
        return
    }

    if (Test-Path -LiteralPath $Target) {
        Remove-Item -Recurse -Force -LiteralPath $Target
    }

    $parent = Split-Path -Parent $Target
    if ($parent) {
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
    }

    Copy-Item -Recurse -Force -LiteralPath $Source -Destination $Target
}

function Copy-DeployFile {
    param(
        [Parameter(Mandatory = $true)][string]$Source,
        [Parameter(Mandatory = $true)][string]$Target
    )

    if (-not (Test-Path -LiteralPath $Source)) {
        return
    }

    $parent = Split-Path -Parent $Target
    if ($parent) {
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
    }

    Copy-Item -Force -LiteralPath $Source -Destination $Target
}

function Copy-DeployAssets {
    param(
        [Parameter(Mandatory = $true)][string]$ProjectRoot,
        [Parameter(Mandatory = $true)][string]$AppRoot
    )

    Copy-DeployDirectory -Source (Join-Path $ProjectRoot "configs") -Target (Join-Path $AppRoot "configs")
    Copy-DeployFile -Source (Join-Path $ProjectRoot "data\datasets\data.yaml") -Target (Join-Path $AppRoot "data\datasets\data.yaml")

    $sourceModels = Join-Path $ProjectRoot "artifacts\models"
    $targetModels = Join-Path $AppRoot "artifacts\models"
    New-Item -ItemType Directory -Force -Path $targetModels | Out-Null

    if (Test-Path -LiteralPath $sourceModels) {
        foreach ($pattern in @("*.pt", "*.onnx", "*.trt", "*.engine")) {
            Get-ChildItem -LiteralPath $sourceModels -File -Filter $pattern | ForEach-Object {
                Copy-DeployFile -Source $_.FullName -Target (Join-Path $targetModels $_.Name)
            }
        }
    }
}

function Get-TensorRTHome {
    if ($env:TensorRT_HOME -and (Test-Path -LiteralPath $env:TensorRT_HOME)) {
        return $env:TensorRT_HOME
    }

    if ($env:TRT_ROOT -and (Test-Path -LiteralPath $env:TRT_ROOT)) {
        return $env:TRT_ROOT
    }

    $trtexec = Get-Command trtexec -ErrorAction SilentlyContinue
    if ($trtexec) {
        return Split-Path -Parent (Split-Path -Parent $trtexec.Source)
    }

    return $null
}

function Copy-TensorRTRuntime {
    param(
        [Parameter(Mandatory = $true)][string]$AppRoot
    )

    $tensorRtHome = Get-TensorRTHome
    if (-not $tensorRtHome) {
        Write-Warning "TensorRT_HOME is not set and trtexec was not found. Skipping TensorRT runtime DLL copy."
        return
    }

    $binDir = Join-Path $tensorRtHome "bin"
    if (-not (Test-Path -LiteralPath $binDir)) {
        Write-Warning "TensorRT bin directory was not found: $binDir"
        return
    }

    $runtimeDllNames = @(
        "nvinfer_10.dll",
        "nvinfer_dispatch_10.dll",
        "nvinfer_lean_10.dll",
        "nvinfer_plugin_10.dll",
        "nvinfer_vc_plugin_10.dll",
        "nvonnxparser_10.dll"
    )

    foreach ($destination in @($AppRoot, (Join-Path $AppRoot "_internal"))) {
        if (-not (Test-Path -LiteralPath $destination)) {
            continue
        }

        foreach ($dllName in $runtimeDllNames) {
            $sourceDll = Join-Path $binDir $dllName
            if (Test-Path -LiteralPath $sourceDll) {
                Copy-DeployFile -Source $sourceDll -Target (Join-Path $destination $dllName)
            }
        }
    }
}
