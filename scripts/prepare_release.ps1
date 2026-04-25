# Prepare release references before tagging a new GitHub release.
param(
    [string]$Version,
    [string]$ReleaseDate = (Get-Date -Format "yyyy-MM-dd"),
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

function Read-PrepareReleaseVersion {
    $selected = Read-Host "Release version to prepare (example: 1.2.3 or v1.2.3)"
    if ([string]::IsNullOrWhiteSpace($selected)) {
        throw "Release version cannot be empty."
    }

    return $selected.Trim()
}

function ConvertTo-ReleaseVersion {
    param([string]$RawVersion)

    $normalized = $RawVersion.Trim()
    if ($normalized.StartsWith("v")) {
        $normalized = $normalized.Substring(1)
    }

    if ($normalized -notmatch '^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$') {
        throw "Release version '$RawVersion' must look like 1.2.3 or v1.2.3."
    }

    return $normalized
}

function Resolve-ProjectRoot {
    return (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

function Get-FileText {
    param([string]$Path)

    if (-not (Test-Path $Path)) {
        throw "Required file was not found: '$Path'"
    }

    return [System.IO.File]::ReadAllText($Path)
}

function Set-FileText {
    param(
        [string]$Path,
        [string]$Content
    )

    [System.IO.File]::WriteAllText($Path, $Content, [System.Text.UTF8Encoding]::new($false))
}

function Update-PatternValue {
    param(
        [string]$Content,
        [string]$Pattern,
        [string]$Replacement,
        [string]$Description
    )

    $match = [regex]::Match($Content, $Pattern, [System.Text.RegularExpressions.RegexOptions]::Multiline)
    if (-not $match.Success) {
        throw "Could not find $Description to update."
    }

    return [regex]::Replace($Content, $Pattern, $Replacement, [System.Text.RegularExpressions.RegexOptions]::Multiline)
}

function Update-LineValue {
    param(
        [string]$Content,
        [string]$Prefix,
        [string]$Replacement,
        [string]$Description
    )

    $newline = "`r`n"
    if ($Content -notmatch "`r`n") {
        $newline = "`n"
    }

    $lines = $Content -split '\r?\n'
    for ($index = 0; $index -lt $lines.Count; $index++) {
        if ($lines[$index].StartsWith($Prefix, [System.StringComparison]::Ordinal)) {
            $lines[$index] = $Replacement
            return ($lines -join $newline)
        }
    }

    throw "Could not find $Description to update."
}

function Add-ChangelogEntry {
    param(
        [string]$Content,
        [string]$Version,
        [string]$ReleaseDate
    )

    $existingHeadingPattern = "(?m)^## \[$([regex]::Escape($Version))\] - "
    if ([regex]::IsMatch($Content, $existingHeadingPattern)) {
        return $Content
    }

    $newline = "`r`n"
    if ($Content -notmatch "`r`n") {
        $newline = "`n"
    }

    $entryLines = @(
        "## [$Version] - $ReleaseDate",
        "",
        "### Added",
        "",
        "- Release notes pending.",
        ""
    )
    $entry = ($entryLines -join $newline) + $newline

    $firstReleaseHeading = [regex]::Match($Content, "(?m)^## \[")
    if ($firstReleaseHeading.Success) {
        return $Content.Insert($firstReleaseHeading.Index, $entry)
    }

    $trimmed = $Content.TrimEnd("`r", "`n")
    if ([string]::IsNullOrWhiteSpace($trimmed)) {
        return $entry
    }

    return $trimmed + $newline + $newline + $entry
}

function Update-ReleaseFiles {
    param(
        [string]$ProjectRoot,
        [string]$Version,
        [string]$ReleaseDate,
        [switch]$DryRun
    )

    $filePlans = @(
        @{
            Path = Join-Path $ProjectRoot "pyproject.toml"
            Description = "project version in pyproject.toml"
            Transform = {
                param([string]$Content, [string]$ResolvedVersion, [string]$ResolvedReleaseDate)
                $replacement = 'version = "{0}"' -f $ResolvedVersion
                Update-LineValue -Content $Content -Prefix 'version = "' -Replacement $replacement -Description "the [project] version in pyproject.toml"
            }
        },
        @{
            Path = Join-Path $ProjectRoot "src\shotsieve\__init__.py"
            Description = "runtime __version__ in src/shotsieve/__init__.py"
            Transform = {
                param([string]$Content, [string]$ResolvedVersion, [string]$ResolvedReleaseDate)
                $replacement = '__version__ = "{0}"' -f $ResolvedVersion
                Update-LineValue -Content $Content -Prefix '__version__ = "' -Replacement $replacement -Description "__version__ in src/shotsieve/__init__.py"
            }
        },
        @{
            Path = Join-Path $ProjectRoot "CHANGELOG.md"
            Description = "release entry in CHANGELOG.md"
            Transform = {
                param([string]$Content, [string]$ResolvedVersion, [string]$ResolvedReleaseDate)
                Add-ChangelogEntry -Content $Content -Version $ResolvedVersion -ReleaseDate $ResolvedReleaseDate
            }
        }
    )

    $pkgInfoPath = Join-Path $ProjectRoot "src\shotsieve.egg-info\PKG-INFO"
    if (Test-Path $pkgInfoPath) {
        $filePlans += @{
            Path = $pkgInfoPath
            Description = "Version field in src/shotsieve.egg-info/PKG-INFO"
            Transform = {
                param([string]$Content, [string]$ResolvedVersion, [string]$ResolvedReleaseDate)
                Update-LineValue -Content $Content -Prefix 'Version: ' -Replacement "Version: $ResolvedVersion" -Description "Version field in src/shotsieve.egg-info/PKG-INFO"
            }
        }
    }

    $changes = New-Object System.Collections.Generic.List[string]
    foreach ($filePlan in $filePlans) {
        $path = [string]$filePlan.Path
        $description = [string]$filePlan.Description
        $transform = $filePlan.Transform

        $currentContent = Get-FileText -Path $path
        $updatedContent = & $transform $currentContent $Version $ReleaseDate
        if ($updatedContent -ne $currentContent) {
            $changes.Add($description)
            if (-not $DryRun) {
                Set-FileText -Path $path -Content $updatedContent
            }
        }
    }

    return $changes
}

$ProjectRoot = Resolve-ProjectRoot

if ([string]::IsNullOrWhiteSpace($Version)) {
    $Version = Read-PrepareReleaseVersion
}

$normalizedVersion = ConvertTo-ReleaseVersion -RawVersion $Version
$changes = Update-ReleaseFiles -ProjectRoot $ProjectRoot -Version $normalizedVersion -ReleaseDate $ReleaseDate -DryRun:$DryRun

if ($DryRun) {
    Write-Host "Dry run complete for release $normalizedVersion."
    if ($changes.Count -eq 0) {
        Write-Host "  No file updates are needed."
    }
    else {
        Write-Host "  The following updates would be applied:"
        foreach ($change in $changes) {
            Write-Host "  - $change"
        }
    }
    Write-Host "  Next step after review/commit: ./scripts/create_github_release.ps1 -Version v$normalizedVersion"
    exit 0
}

if ($changes.Count -eq 0) {
    Write-Host "Release references already match $normalizedVersion."
}
else {
    Write-Host "Prepared release references for $normalizedVersion."
    foreach ($change in $changes) {
        Write-Host "- Updated $change"
    }
}

Write-Host "Next step: review and commit the changes, then run ./scripts/create_github_release.ps1 -Version v$normalizedVersion"