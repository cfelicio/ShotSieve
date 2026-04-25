param(
    [ValidateSet("runtime")]
    [string]$Mode = "runtime",
    [string]$DistRoot = ".\dist",
    [string]$BuildRoot = ".\build\release-targets",
    [string]$PythonExe = "python",
    [string[]]$TargetIds = @(),
    [switch]$PlanOnly,
    [switch]$AsJson
)

$ErrorActionPreference = "Stop"

function Resolve-AbsolutePath {
    param(
        [string]$ProjectRoot,
        [string]$PathValue
    )

    if ([System.IO.Path]::IsPathRooted($PathValue)) {
        return $PathValue
    }

    return Join-Path $ProjectRoot $PathValue
}

function Resolve-PythonCommand {
    param(
        [string]$ProjectRoot,
        [string]$ConfiguredPythonExe
    )

    if (-not [string]::IsNullOrWhiteSpace($ConfiguredPythonExe) -and $ConfiguredPythonExe -ne "python") {
        return $ConfiguredPythonExe
    }

    $venvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
    if (Test-Path $venvPython) {
        return $venvPython
    }

    return $ConfiguredPythonExe
}

function Resolve-TargetPythonCommand {
    param(
        [string]$BasePythonCommand,
        [string]$ResolvedBuildRoot,
        [string]$TargetId
    )

    $targetVenvRoot = Join-Path (Join-Path $ResolvedBuildRoot $TargetId) ".venv"
    $targetPython = Join-Path $targetVenvRoot "Scripts\python.exe"
    $targetPyvenvConfig = Join-Path $targetVenvRoot "pyvenv.cfg"
    $needsRecreate = $false

    if (Test-Path $targetPython) {
        if (-not (Test-Path $targetPyvenvConfig)) {
            Write-Host "Recreating broken build environment for '$TargetId' at '$targetVenvRoot' (missing pyvenv.cfg)..."
            $needsRecreate = $true
        }
        else {
            try {
                & $targetPython -c "import sys" *> $null
            }
            catch {
                $global:LASTEXITCODE = 1
            }

            if ($LASTEXITCODE -ne 0) {
                Write-Host "Recreating broken build environment for '$TargetId' at '$targetVenvRoot' (python startup failed)..."
                $needsRecreate = $true
            }
        }
    }
    else {
        $needsRecreate = $true
    }

    if ($needsRecreate -and (Test-Path $targetVenvRoot)) {
        Remove-Item -Recurse -Force $targetVenvRoot
    }

    if ($needsRecreate) {
        Write-Host "Creating isolated build environment for '$TargetId' at '$targetVenvRoot'..."
        & $BasePythonCommand -m venv $targetVenvRoot
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to create virtual environment for target '$TargetId'."
        }
    }

    return $targetPython
}

function Test-PythonModuleAvailable {
    param(
        [string]$PythonCommand,
        [string]$ModuleName
    )

    & $PythonCommand -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('$ModuleName') else 1)"
    return $LASTEXITCODE -eq 0
}

function Install-PyInstallerIfMissing {
    param(
        [string]$PythonCommand,
        [string]$ConstraintsFile
    )

    if (Test-PythonModuleAvailable -PythonCommand $PythonCommand -ModuleName "PyInstaller") {
        return
    }

    Write-Host "PyInstaller not found for '$PythonCommand'. Installing..."
    & $PythonCommand -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to upgrade pip before installing PyInstaller."
    }

    & $PythonCommand -m pip install pyinstaller -c $ConstraintsFile
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install PyInstaller for '$PythonCommand'."
    }
}

function Install-TorchVariant {
    param(
        [string]$PythonCommand,
        [string]$TorchVariant,
        [string]$ConstraintsFile
    )

    $variant = "default"
    if (-not [string]::IsNullOrWhiteSpace($TorchVariant)) {
        $variant = $TorchVariant.ToLowerInvariant()
    }

    switch ($variant) {
        "cpu" {
            Write-Host "Installing CPU Torch runtime..."
            & $PythonCommand -m pip install --upgrade --force-reinstall --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu -c $ConstraintsFile
            break
        }
        "cuda" {
            Write-Host "Skipping explicit CUDA Torch preinstall for torchless runtime-pack target..."
            break
        }
        "directml" {
            Write-Host "Installing default Torch runtime for DirectML target..."
            & $PythonCommand -m pip install --upgrade --force-reinstall --no-cache-dir torch torchvision -c $ConstraintsFile
            break
        }
        default {
            Write-Host "Installing default Torch runtime..."
            & $PythonCommand -m pip install --upgrade --force-reinstall --no-cache-dir torch torchvision -c $ConstraintsFile
            break
        }
    }

    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install Torch runtime variant '$TorchVariant'."
    }
}

function Install-TargetDependencies {
    param(
        [string]$PythonCommand,
        [string]$ProjectRoot,
        [object]$Target,
        [string]$ConstraintsFile
    )

    Write-Host "Refreshing packaging tools for '$($Target.id)'..."
    & $PythonCommand -m pip install --upgrade pip setuptools wheel -c $ConstraintsFile
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to upgrade packaging tools for target '$($Target.id)'."
    }

    Install-TorchVariant -PythonCommand $PythonCommand -TorchVariant ([string]$Target.torchVariant) -ConstraintsFile $ConstraintsFile

    $extras = @($Target.extras)
    if ($extras.Count -gt 0) {
        $extrasCsv = ($extras -join ",")
        Write-Host "Installing ShotSieve extras for '$($Target.id)': $extrasCsv"
        Push-Location $ProjectRoot
        try {
            & $PythonCommand -m pip install -e ".[${extrasCsv}]" -c $ConstraintsFile
            if ($LASTEXITCODE -ne 0) {
                throw "Failed to install ShotSieve extras for target '$($Target.id)'."
            }
        }
        finally {
            Pop-Location
        }
    }

    Install-PyInstallerIfMissing -PythonCommand $PythonCommand -ConstraintsFile $ConstraintsFile
}

function Get-WindowsTargets {
    param(
        [string]$PythonCommand,
        [string]$ProjectRoot
    )

    $jsonLines = & $PythonCommand (Join-Path $ProjectRoot "scripts\release_target_matrix.py") --kind runtime
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to resolve release target matrix for kind 'runtime'."
    }

    $json = ($jsonLines -join "`n")
    $targets = $json | ConvertFrom-Json
    return @($targets | Where-Object { $_.id -like "windows-*" })
}

function New-WindowsTargetBundle {
    param(
        [string]$PythonCommand,
        [string]$ProjectRoot,
        [object]$Target,
        [string]$ResolvedDistRoot,
        [string]$ResolvedBuildRoot
    )

    $cudaTarget = ([string]$Target.torchVariant).ToLowerInvariant() -eq "cuda"
    $hadSkipEnv = Test-Path Env:SHOTSIEVE_SKIP_BUNDLED_TORCH
    $previousSkipEnv = $env:SHOTSIEVE_SKIP_BUNDLED_TORCH

    if ($cudaTarget) {
        Write-Host "Configuring torchless CUDA bundle packaging for '$($Target.id)'..."
        $env:SHOTSIEVE_SKIP_BUNDLED_TORCH = "1"
    }

    try {
        & $PythonCommand (Join-Path $ProjectRoot "scripts\build_portable_bundle.py") --target $Target.id --dist-root $ResolvedDistRoot --build-root $ResolvedBuildRoot
        if ($LASTEXITCODE -ne 0) {
            throw "Build failed for target '$($Target.id)'."
        }
    }
    finally {
        if ($hadSkipEnv) {
            $env:SHOTSIEVE_SKIP_BUNDLED_TORCH = $previousSkipEnv
        }
        else {
            Remove-Item Env:SHOTSIEVE_SKIP_BUNDLED_TORCH -ErrorAction SilentlyContinue
        }
    }

    $archivePath = Join-Path $ResolvedDistRoot $Target.archiveName
    if (-not (Test-Path $archivePath)) {
        throw "Build finished but expected archive was not found: '$archivePath'"
    }

    return (Resolve-Path $archivePath).Path
}

if ($MyInvocation.InvocationName -eq '.') {
    return
}

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
$resolvedPythonCommand = Resolve-PythonCommand -ProjectRoot $projectRoot -ConfiguredPythonExe $PythonExe
$resolvedDistRoot = Resolve-AbsolutePath -ProjectRoot $projectRoot -PathValue $DistRoot
$resolvedBuildRoot = Resolve-AbsolutePath -ProjectRoot $projectRoot -PathValue $BuildRoot
$constraintsFile = Join-Path $projectRoot "scripts\release-constraints.txt"

if (-not (Test-Path $constraintsFile)) {
    throw "Release constraints file not found: '$constraintsFile'"
}

if (-not (Test-Path $resolvedDistRoot)) {
    New-Item -ItemType Directory -Path $resolvedDistRoot | Out-Null
}

if (-not (Test-Path $resolvedBuildRoot)) {
    New-Item -ItemType Directory -Path $resolvedBuildRoot | Out-Null
}

$selectedTargets = Get-WindowsTargets -PythonCommand $resolvedPythonCommand -ProjectRoot $projectRoot

if ($TargetIds.Count -gt 0) {
    $targetIdLookup = @{}
    foreach ($targetId in $TargetIds) {
        if ([string]::IsNullOrWhiteSpace($targetId)) {
            continue
        }
        $targetIdLookup[$targetId.Trim().ToLowerInvariant()] = $true
    }

    $selectedTargets = @(
        $selectedTargets | Where-Object {
            $targetIdLookup.ContainsKey(([string]$_.id).ToLowerInvariant())
        }
    )
}

if ($selectedTargets.Count -eq 0) {
    if ($TargetIds.Count -gt 0) {
        throw "No Windows targets matched the requested target ids: $($TargetIds -join ', ')."
    }
    throw "No Windows targets found for mode '$Mode'."
}

if ($PlanOnly) {
    if ($AsJson) {
        $selectedTargets | ConvertTo-Json -Depth 8
    }
    else {
        $selectedTargets | Format-Table id, buildProfile, runtime, executableName, archiveName
    }
    return
}

foreach ($target in $selectedTargets) {
    Write-Host "Building Windows release target '$($target.id)'..."
    $targetPythonCommand = Resolve-TargetPythonCommand -BasePythonCommand $resolvedPythonCommand -ResolvedBuildRoot $resolvedBuildRoot -TargetId $target.id
    Install-TargetDependencies -PythonCommand $targetPythonCommand -Target $target -ProjectRoot $projectRoot -ConstraintsFile $constraintsFile
    $archivePath = New-WindowsTargetBundle -PythonCommand $targetPythonCommand -ProjectRoot $projectRoot -Target $target -ResolvedDistRoot $resolvedDistRoot -ResolvedBuildRoot $resolvedBuildRoot
    Write-Host "Built archive: $archivePath"
}

Write-Host ""
Write-Host "Done. Built Windows targets:" -ForegroundColor Green
foreach ($target in $selectedTargets) {
    $archivePath = Join-Path $resolvedDistRoot $target.archiveName
    Write-Host " - $($target.id): $archivePath"
}
Write-Host "Build root: $resolvedBuildRoot"
