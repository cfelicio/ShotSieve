# Tag-push helper for the GitHub Actions release flow.
# Use scripts/prepare_release.ps1 first when you need to update version files and changelog entries.
param(
    [string]$Version,
    [switch]$PreRelease,
    [switch]$SkipFetch,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

function Invoke-CommandChecked {
    param(
        [string]$Command,
        [string[]]$Arguments,
        [switch]$CaptureOutput
    )

    if ($CaptureOutput) {
        $output = & $Command @Arguments 2>&1
        if ($LASTEXITCODE -ne 0) {
            $joinedOutput = ($output -join "`n")
            throw "Command failed: $Command $($Arguments -join ' ')`n$joinedOutput"
        }

        return ($output -join "`n").Trim()
    }

    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed: $Command $($Arguments -join ' ')"
    }
}

function Resolve-CommandPath {
    param([string]$CommandName)

    $resolved = Get-Command $CommandName -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($resolved) {
        $resolvedPath = $resolved.Source
        if ([string]::IsNullOrWhiteSpace($resolvedPath)) {
            $resolvedPath = $resolved.Definition
        }
        if ([string]::IsNullOrWhiteSpace($resolvedPath)) {
            $resolvedPath = $CommandName
        }

        return $resolvedPath
    }

    throw "Required command '$CommandName' was not found in PATH."
}

function Test-CleanGitState {
    param([string]$GitCommand)

    $porcelain = Invoke-CommandChecked -Command $GitCommand -Arguments @("status", "--porcelain") -CaptureOutput
    if (-not [string]::IsNullOrWhiteSpace($porcelain)) {
        throw "Working tree is not clean. Commit, stash, or discard changes before creating a release."
    }

    $branch = Invoke-CommandChecked -Command $GitCommand -Arguments @("rev-parse", "--abbrev-ref", "HEAD") -CaptureOutput
    if ($branch -eq "HEAD") {
        throw "You are in detached HEAD state. Check out a branch before creating a release."
    }

    if (-not $SkipFetch) {
        Invoke-CommandChecked -Command $GitCommand -Arguments @("fetch", "--tags", "origin")
    }

    $aheadBehind = Invoke-CommandChecked -Command $GitCommand -Arguments @("rev-list", "--left-right", "--count", "origin/$branch...HEAD") -CaptureOutput
    $parts = $aheadBehind -split "\s+"
    if ($parts.Count -lt 2) {
        throw "Unable to parse branch divergence for origin/$branch."
    }

    $behindCount = [int]$parts[0]
    $aheadCount = [int]$parts[1]

    if ($behindCount -gt 0 -or $aheadCount -gt 0) {
        throw "Branch '$branch' is not in sync with origin/$branch (behind=$behindCount, ahead=$aheadCount)."
    }
}

function Get-LatestReleaseTag {
    param([string]$GitCommand)

    try {
        $latestTag = Invoke-CommandChecked -Command $GitCommand -Arguments @("describe", "--tags", "--abbrev=0") -CaptureOutput
        return $latestTag
    }
    catch {
        return ""
    }
}

function Read-ReleaseVersion {
    param([string]$LatestTag)

    $prompt = "Version tag to release (example: v1.2.3)"
    if (-not [string]::IsNullOrWhiteSpace($LatestTag)) {
        $prompt = "$prompt. Latest tag: $LatestTag"
    }

    $selected = Read-Host $prompt
    if ([string]::IsNullOrWhiteSpace($selected)) {
        throw "Version tag cannot be empty."
    }

    return $selected.Trim()
}

function Test-VersionNotAlreadyPublished {
    param(
        [string]$GitCommand,
        [string]$VersionTag
    )

    $existingLocalTag = Invoke-CommandChecked -Command $GitCommand -Arguments @("tag", "--list", $VersionTag) -CaptureOutput
    if ($existingLocalTag -eq $VersionTag) {
        throw "Tag '$VersionTag' already exists locally."
    }

    $remoteTag = Invoke-CommandChecked -Command $GitCommand -Arguments @("ls-remote", "--tags", "--refs", "origin", $VersionTag) -CaptureOutput
    if (-not [string]::IsNullOrWhiteSpace($remoteTag)) {
        throw "Tag '$VersionTag' already exists on origin."
    }
}

$GitCommand = Resolve-CommandPath -CommandName "git"

if ($PreRelease) {
    throw "The tag-push release flow does not support -PreRelease. Push the tag first, then mark the GitHub release as a pre-release manually or update .github/workflows/release.yml to support it."
}

Test-CleanGitState -GitCommand $GitCommand

$latestRelease = Get-LatestReleaseTag -GitCommand $GitCommand
if ([string]::IsNullOrWhiteSpace($Version)) {
    $Version = Read-ReleaseVersion -LatestTag $latestRelease
}

if ($Version -notmatch "^v") {
    throw "Version tag '$Version' must start with 'v' (for example: v1.2.3)."
}

Test-VersionNotAlreadyPublished -GitCommand $GitCommand -VersionTag $Version

if ($DryRun) {
    Write-Host "Dry run complete. The following commands would be executed:"
    Write-Host "  git tag -a $Version -m 'Release $Version'"
    Write-Host "  git push origin $Version"
    Write-Host "  (This tag helper does not edit version files; run ./scripts/prepare_release.ps1 first if needed.)"
    Write-Host "  GitHub Actions release workflow will publish the release after the tag push."
    exit 0
}

Invoke-CommandChecked -Command $GitCommand -Arguments @("tag", "-a", $Version, "-m", "Release $Version")
Invoke-CommandChecked -Command $GitCommand -Arguments @("push", "origin", $Version)

Write-Host "Tag '$Version' pushed successfully. GitHub Actions will publish the release from .github/workflows/release.yml."
