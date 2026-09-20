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

function ConvertTo-CanonicalTargetId {
    param([string]$TargetId)

    return $TargetId.Trim().ToLowerInvariant()
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

function Get-PythonVersion {
    param(
        [string]$PythonCommand
    )

    $versionOutput = & $PythonCommand -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to determine the Python version for '$PythonCommand'."
    }

    $version = ($versionOutput | Select-Object -Last 1).Trim()
    if ([string]::IsNullOrWhiteSpace($version)) {
        throw "Python '$PythonCommand' did not report a version."
    }

    return $version
}

function Resolve-PythonCommandForTarget {
    param(
        [string]$BasePythonCommand,
        [string]$ConfiguredPythonExe,
        [string]$ExpectedPythonVersion
    )

    $baseVersion = Get-PythonVersion -PythonCommand $BasePythonCommand
    if ($baseVersion -eq $ExpectedPythonVersion) {
        return $BasePythonCommand
    }

    if ($ConfiguredPythonExe -ne "python") {
        throw "Configured Python $baseVersion does not match target requirement Python $ExpectedPythonVersion. Re-run with -PythonExe pointing to a matching interpreter."
    }

    $pythonLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($null -eq $pythonLauncher) {
        throw "Target requires Python $ExpectedPythonVersion, but the configured Python is $baseVersion and the Windows Python launcher could not be found. Re-run with -PythonExe pointing to a matching interpreter."
    }

    $candidate = & $pythonLauncher.Source "-$ExpectedPythonVersion" -c "import sys; print(sys.executable)"
    if ($LASTEXITCODE -ne 0) {
        throw "Target requires Python $ExpectedPythonVersion, but the Windows Python launcher could not resolve that interpreter. Re-run with -PythonExe pointing to a matching interpreter."
    }

    $candidatePath = ($candidate | Select-Object -Last 1).Trim()
    if ([string]::IsNullOrWhiteSpace($candidatePath) -or -not (Test-Path $candidatePath)) {
        throw "Target requires Python $ExpectedPythonVersion, but the Windows Python launcher returned no usable interpreter. Re-run with -PythonExe pointing to a matching interpreter."
    }

    return $candidatePath
}

function Resolve-TargetPythonCommand {
    param(
        [string]$BasePythonCommand,
        [string]$ResolvedBuildRoot,
        [string]$TargetId,
        [string]$ExpectedPythonVersion = ""
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
            elseif (-not [string]::IsNullOrWhiteSpace($ExpectedPythonVersion)) {
                $actualPythonVersion = Get-PythonVersion -PythonCommand $targetPython
                if ($actualPythonVersion -ne $ExpectedPythonVersion) {
                    Write-Host "Recreating build environment for '$TargetId' at '$targetVenvRoot' (requires Python $ExpectedPythonVersion, found $actualPythonVersion)..."
                    $needsRecreate = $true
                }
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

    if (-not [string]::IsNullOrWhiteSpace($ExpectedPythonVersion)) {
        $actualPythonVersion = Get-PythonVersion -PythonCommand $targetPython
        if ($actualPythonVersion -ne $ExpectedPythonVersion) {
            throw "Build environment for '$TargetId' uses Python $actualPythonVersion, but the target requires Python $ExpectedPythonVersion. Re-run with -PythonExe pointing to a matching interpreter."
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
        [string]$ConstraintsFile,
        [string]$AdditionalConstraintsFile = ""
    )

    if (Test-PythonModuleAvailable -PythonCommand $PythonCommand -ModuleName "PyInstaller") {
        return
    }

    Write-Host "PyInstaller not found for '$PythonCommand'. Installing..."
    & $PythonCommand -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to upgrade pip before installing PyInstaller."
    }

    $constraintArgs = @("-c", $ConstraintsFile)
    if (-not [string]::IsNullOrWhiteSpace($AdditionalConstraintsFile)) {
        $constraintArgs += @("-c", $AdditionalConstraintsFile)
    }
    & $PythonCommand -m pip install pyinstaller @constraintArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install PyInstaller for '$PythonCommand'."
    }
}

function Install-TorchVariant {
    param(
        [string]$PythonCommand,
        [string]$TorchVariant,
        [string]$TargetId,
        [string]$BuildRoot,
        [string]$ConstraintsFile,
        [string]$AdditionalConstraintsFile = ""
    )

    $variant = "default"
    if (-not [string]::IsNullOrWhiteSpace($TorchVariant)) {
        $variant = $TorchVariant.ToLowerInvariant()
    }
    $constraintArgs = @("-c", $ConstraintsFile)
    if (-not [string]::IsNullOrWhiteSpace($AdditionalConstraintsFile)) {
        $constraintArgs += @("-c", $AdditionalConstraintsFile)
    }

    switch ($variant) {
        "cpu" {
            Write-Host "Installing CPU Torch runtime..."
            & $PythonCommand -m pip install --upgrade --force-reinstall --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu @constraintArgs
            break
        }
        "cuda" {
            Write-Host "Installing the pinned CUDA Torch/Torchvision pair for torchless runtime-pack target..."
            & $PythonCommand -m pip install --upgrade --force-reinstall --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu130 --trusted-host download.pytorch.org @constraintArgs
            break
        }
        "xpu" {
            Write-Host "Installing the pinned Intel XPU Torch/Torchvision pair..."
            & $PythonCommand -m pip install --upgrade --force-reinstall --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/xpu --extra-index-url https://pypi.org/simple @constraintArgs
            break
        }
        "rocm" {
            Write-Host "Installing the AMD ROCm 7.2.1 Torch runtime for the Windows release target..."
            $rocmPackages = @(
                "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_core-7.2.1-py3-none-win_amd64.whl",
                "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl",
                "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl",
                "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm-7.2.1.tar.gz",
                "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl",
                "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torchaudio-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl",
                "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torchvision-0.24.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl"
            )
            & $PythonCommand -m pip install --upgrade --force-reinstall --no-cache-dir @rocmPackages --trusted-host repo.radeon.com @constraintArgs
            break
        }
        "rocm10-gfx1103" {
            Write-Host "Preparing the stable AMD ROCm 10.0 gfx1103 candidate..."
            $selectorWheelDirectory = Join-Path (Join-Path $BuildRoot $TargetId) "rocm-selector-wheel"
            New-Item -ItemType Directory -Force -Path $selectorWheelDirectory | Out-Null
            & $PythonCommand -m pip wheel --no-deps --wheel-dir $selectorWheelDirectory "rocm==10.0.0" --index-url https://stable.repo.amd.com/rocm/whl-next/ --extra-index-url https://pypi.org/simple
            if ($LASTEXITCODE -ne 0) {
                throw "Failed to build the ROCm 10 selector wheel for target '$TargetId'."
            }
            $rocm10Packages = @(
                "rocm==10.0.0",
                "torch[device-gfx1103]==2.13.0+rocm10.0.0",
                "torchvision[device-gfx1103]==0.28.0+rocm10.0.0"
            )
            & $PythonCommand -m pip install --upgrade --force-reinstall --no-cache-dir @rocm10Packages --index-url https://stable.repo.amd.com/rocm/whl-next/ --extra-index-url https://pypi.org/simple --find-links $selectorWheelDirectory --only-binary=rocm @constraintArgs
            break
        }
        default {
            Write-Host "Installing default Torch runtime..."
            & $PythonCommand -m pip install --upgrade --force-reinstall --no-cache-dir torch torchvision @constraintArgs
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
        [string]$BuildRoot,
        [object]$Target,
        [string]$ConstraintsFile
    )

    $targetConstraintsFile = $ConstraintsFile
    $targetConstraintRelativePath = [string]$Target.constraintsFile
    if (-not [string]::IsNullOrWhiteSpace($targetConstraintRelativePath)) {
        $targetConstraintsFile = Join-Path $ProjectRoot $targetConstraintRelativePath
    }
    if (-not (Test-Path $targetConstraintsFile)) {
        throw "Target constraints file not found for '$($Target.id)': '$targetConstraintsFile'"
    }

    Write-Host "Refreshing packaging tools for '$($Target.id)'..."
    & $PythonCommand -m pip install --upgrade pip setuptools wheel packaging -c $targetConstraintsFile -c $ConstraintsFile
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to upgrade packaging tools for target '$($Target.id)'."
    }

    $additionalConstraintsFile = if ($targetConstraintsFile -ne $ConstraintsFile) { $ConstraintsFile } else { "" }
    Install-TorchVariant -PythonCommand $PythonCommand -TorchVariant ([string]$Target.torchVariant) -TargetId ([string]$Target.id) -BuildRoot $BuildRoot -ConstraintsFile $targetConstraintsFile -AdditionalConstraintsFile $additionalConstraintsFile

    $extras = @($Target.extras)
    if ($extras.Count -gt 0) {
        $extrasCsv = ($extras -join ",")
        Write-Host "Installing ShotSieve extras for '$($Target.id)': $extrasCsv"
        Push-Location $ProjectRoot
        try {
            & $PythonCommand -m pip install -e ".[${extrasCsv}]" -c $targetConstraintsFile -c $ConstraintsFile
            if ($LASTEXITCODE -ne 0) {
                throw "Failed to install ShotSieve extras for target '$($Target.id)'."
            }
        }
        finally {
            Pop-Location
        }
    }

    Install-PyInstallerIfMissing -PythonCommand $PythonCommand -ConstraintsFile $targetConstraintsFile -AdditionalConstraintsFile $additionalConstraintsFile

    Write-Host "Checking resolved requirements for '$($Target.id)'..."
    & $PythonCommand -m pip check
    if ($LASTEXITCODE -ne 0) {
        throw "Resolved requirements are inconsistent for target '$($Target.id)'."
    }
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

    $hadSkipEnv = Test-Path Env:SHOTSIEVE_SKIP_BUNDLED_TORCH
    $previousSkipEnv = $env:SHOTSIEVE_SKIP_BUNDLED_TORCH

    # Every portable target is a torchless runtime pack.  The target-specific
    # build environment still installs Torch for PyInstaller analysis, but the
    # final archive must load it from data/runtime/site-packages/<target-id>.
    Write-Host "Configuring torchless runtime-pack packaging for '$($Target.id)'..."
    $env:SHOTSIEVE_SKIP_BUNDLED_TORCH = "1"

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
        $targetIdLookup[(ConvertTo-CanonicalTargetId $targetId)] = $true
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
    $targetBasePythonCommand = Resolve-PythonCommandForTarget -BasePythonCommand $resolvedPythonCommand -ConfiguredPythonExe $PythonExe -ExpectedPythonVersion ([string]$target.pythonVersion)
    $targetPythonCommand = Resolve-TargetPythonCommand -BasePythonCommand $targetBasePythonCommand -ResolvedBuildRoot $resolvedBuildRoot -TargetId $target.id -ExpectedPythonVersion ([string]$target.pythonVersion)
    Install-TargetDependencies -PythonCommand $targetPythonCommand -Target $target -ProjectRoot $projectRoot -BuildRoot $resolvedBuildRoot -ConstraintsFile $constraintsFile
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
