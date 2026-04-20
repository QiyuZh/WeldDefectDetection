Set-StrictMode -Version Latest

function Test-PythonModules {
    param(
        [Parameter(Mandatory = $true)][string]$PythonExe,
        [Parameter(Mandatory = $true)][string[]]$Modules
    )

    if (-not (Test-Path -LiteralPath $PythonExe)) {
        return $false
    }

    $script = @"
import importlib.util
import sys
mods = sys.argv[1:]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
raise SystemExit(0 if not missing else 1)
"@
    & $PythonExe -c $script @Modules | Out-Null
    return ($LASTEXITCODE -eq 0)
}

function Get-PackagingPython {
    param(
        [string]$PreferredPythonExe
    )

    $requiredModules = @("PyInstaller", "cv2")
    $preferredCandidates = [System.Collections.Generic.List[string]]::new()
    $fallbackCandidates = [System.Collections.Generic.List[string]]::new()
    $seen = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)

    function Add-Candidate {
        param(
            [string]$Value,
            [bool]$Preferred
        )

        if (-not $Value) {
            return
        }

        if (-not (Test-Path -LiteralPath $Value)) {
            return
        }

        if (-not $seen.Add($Value)) {
            return
        }

        if ($Preferred) {
            $preferredCandidates.Add($Value)
        }
        else {
            $fallbackCandidates.Add($Value)
        }
    }

    Add-Candidate -Value $PreferredPythonExe -Preferred $true
    Add-Candidate -Value $env:PACKAGING_PYTHON_EXE -Preferred $true

    if ($env:CONDA_PREFIX) {
        Add-Candidate -Value (Join-Path $env:CONDA_PREFIX "python.exe") -Preferred $true
    }

    $wherePython = @(where.exe python 2>$null | Where-Object { $_ -and (Test-Path -LiteralPath $_) })
    foreach ($candidate in $wherePython) {
        $isEnvPython = $candidate -match "\\\\.conda\\\\envs\\\\" -or $candidate -match "\\\\envs\\\\"
        Add-Candidate -Value $candidate -Preferred $isEnvPython
    }

    $userCondaEnvs = Join-Path $env:USERPROFILE ".conda\envs"
    if (Test-Path -LiteralPath $userCondaEnvs) {
        Get-ChildItem -LiteralPath $userCondaEnvs -Directory | ForEach-Object {
            Add-Candidate -Value (Join-Path $_.FullName "python.exe") -Preferred $false
        }
    }

    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCommand -and $pythonCommand.Source) {
        Add-Candidate -Value $pythonCommand.Source -Preferred $false
    }

    $preferredMatches = [System.Collections.Generic.List[string]]::new()
    $fallbackMatches = [System.Collections.Generic.List[string]]::new()

    foreach ($candidate in $preferredCandidates) {
        if (Test-PythonModules -PythonExe $candidate -Modules $requiredModules) {
            $preferredMatches.Add($candidate)
        }
    }

    if ($preferredMatches.Count -ge 1) {
        return $preferredMatches[0]
    }

    foreach ($candidate in $fallbackCandidates) {
        if (Test-PythonModules -PythonExe $candidate -Modules $requiredModules) {
            $fallbackMatches.Add($candidate)
        }
    }

    if ($fallbackMatches.Count -eq 1) {
        return $fallbackMatches[0]
    }

    if ($fallbackMatches.Count -gt 1) {
        $candidateList = ($fallbackMatches | ForEach-Object { " - $_" }) -join "`n"
        throw "Multiple Python environments match the packaging requirements.`n$candidateList`nPass -PythonExe explicitly."
    }

    foreach ($candidate in @($preferredCandidates + $fallbackCandidates)) {
        if (Test-Path -LiteralPath $candidate) {
            return $candidate
        }
    }

    throw "python.exe was not found. Activate the target environment first, or pass -PythonExe explicitly."
}

function Get-PythonEnvRoot {
    param(
        [Parameter(Mandatory = $true)][string]$PythonExe
    )

    if (-not (Test-Path -LiteralPath $PythonExe)) {
        return $null
    }

    return Split-Path -Parent $PythonExe
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
    param(
        [string]$PreferredTensorRTHome
    )

    if ($PreferredTensorRTHome -and (Test-Path -LiteralPath $PreferredTensorRTHome)) {
        return $PreferredTensorRTHome
    }

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

function Copy-CondaRuntime {
    param(
        [Parameter(Mandatory = $true)][string]$AppRoot,
        [Parameter(Mandatory = $true)][string]$PythonExe
    )

    $envRoot = Get-PythonEnvRoot -PythonExe $PythonExe
    if (-not $envRoot) {
        Write-Warning "Python environment root could not be resolved from: $PythonExe"
        return
    }

    $libraryBin = Join-Path $envRoot "Library\bin"
    $cv2Root = Join-Path $envRoot "Lib\site-packages\cv2"

    foreach ($destination in @($AppRoot, (Join-Path $AppRoot "_internal"))) {
        if (-not (Test-Path -LiteralPath $destination)) {
            continue
        }

        if (Test-Path -LiteralPath $libraryBin) {
            Get-ChildItem -LiteralPath $libraryBin -File -Filter "*.dll" | ForEach-Object {
                Copy-DeployFile -Source $_.FullName -Target (Join-Path $destination $_.Name)
            }
        }

        if (Test-Path -LiteralPath $cv2Root) {
            Get-ChildItem -LiteralPath $cv2Root -Recurse -File | Where-Object {
                $_.Extension -in @(".dll", ".pyd")
            } | ForEach-Object {
                Copy-DeployFile -Source $_.FullName -Target (Join-Path $destination $_.Name)
            }
        }
    }
}

function Copy-TensorRTRuntime {
    param(
        [Parameter(Mandatory = $true)][string]$AppRoot,
        [string]$TensorRTHome
    )

    $tensorRtHome = Get-TensorRTHome -PreferredTensorRTHome $TensorRTHome
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
