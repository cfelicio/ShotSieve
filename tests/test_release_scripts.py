from __future__ import annotations

import importlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import typing
import zipfile
import tomllib
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "build_windows_releases.ps1"
GITHUB_RELEASE_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "create_github_release.ps1"
PREPARE_RELEASE_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "prepare_release.ps1"
MATRIX_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "release_target_matrix.py"
BUNDLE_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "build_portable_bundle.py"
BOOTSTRAP_MANIFEST_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "generate_bootstrap_manifest.py"
EMBED_BOOTSTRAP_MANIFEST_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "embed_manifest_in_bootstrap_archives.py"
SPEC_PATH = PROJECT_ROOT / "shotsieve.spec"
RELEASE_CONSTRAINTS_PATH = PROJECT_ROOT / "scripts" / "release-constraints.txt"
LOCAL_ONLY_REPO_RELATIVE_PATHS = (
    "blog.md",
    ".github/agents/anvil.agent.md",
)


def _dict_value(value: object) -> dict[str, object]:
    return cast(dict[str, object], value)


def _string_value(value: object) -> str:
    return str(value)


def powershell_executable() -> str:
    for candidate in ("powershell", "pwsh"):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved

    pytest.skip("PowerShell executable is not available in this test environment")


def git_executable() -> str:
    executable = shutil.which("git")
    if executable:
        return executable

    pytest.skip("git is not available in this test environment")


def create_temp_release_repo(tmp_path: Path) -> Path:
    git = git_executable()
    origin = tmp_path / "origin.git"
    repo = tmp_path / "repo"

    subprocess.run([git, "init", "--bare", str(origin)], check=True, capture_output=True, text=True)
    subprocess.run([git, "clone", str(origin), str(repo)], check=True, capture_output=True, text=True)
    subprocess.run([git, "-C", str(repo), "config", "user.name", "ShotSieve Tests"], check=True)
    subprocess.run([git, "-C", str(repo), "config", "user.email", "shotsieve-tests@example.com"], check=True)

    (repo / "README.md").write_text("temporary release repo\n", encoding="utf-8")
    subprocess.run([git, "-C", str(repo), "add", "README.md"], check=True)
    subprocess.run([git, "-C", str(repo), "commit", "-m", "Initial commit"], check=True)

    branch = subprocess.run(
        [git, "-C", str(repo), "branch", "--show-current"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run([git, "-C", str(repo), "push", "-u", "origin", branch], check=True, capture_output=True, text=True)
    return repo


def create_git_wrapper(bin_dir: Path) -> None:
    bin_dir.mkdir(parents=True, exist_ok=True)
    wrapper = bin_dir / "git.cmd"
    wrapper.write_text(
        "@echo off\r\n"
        f'"{git_executable()}" %*\r\n',
        encoding="utf-8",
    )


def build_release_script_env(tmp_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    bin_dir = tmp_path / "bin"
    create_git_wrapper(bin_dir)
    existing_path = env.get("PATH", "")
    env["PATH"] = str(bin_dir) if not existing_path else os.pathsep.join((str(bin_dir), existing_path))
    return env


def create_temp_prepare_release_workspace(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "prepare-workspace"
    script_dir = repo / "scripts"
    shotsieve_dir = repo / "src" / "shotsieve"
    egg_info_dir = repo / "src" / "shotsieve.egg-info"

    script_dir.mkdir(parents=True, exist_ok=True)
    shotsieve_dir.mkdir(parents=True, exist_ok=True)
    egg_info_dir.mkdir(parents=True, exist_ok=True)

    (repo / "pyproject.toml").write_text(
        "[project]\n"
        'name = "shotsieve"\n'
        'version = "0.1.0"\n',
        encoding="utf-8",
    )
    (shotsieve_dir / "__init__.py").write_text(
        '__all__ = ["__version__"]\n\n__version__ = "0.1.0"\n',
        encoding="utf-8",
    )
    (repo / "CHANGELOG.md").write_text(
        "# Changelog\n\n"
        "All notable changes to this project will be documented in this file.\n\n"
        "## [0.1.0] - 2026-04-25\n\n"
        "### Added\n\n"
        "- Initial release of ShotSieve.\n",
        encoding="utf-8",
    )
    (egg_info_dir / "PKG-INFO").write_text(
        "Metadata-Version: 2.4\n"
        "Name: shotsieve\n"
        "Version: 0.1.0\n",
        encoding="utf-8",
    )

    prepare_script_copy = script_dir / "prepare_release.ps1"
    prepare_script_copy.write_text(PREPARE_RELEASE_SCRIPT_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    return repo, prepare_script_copy


def run_release_plan(mode: str = "runtime") -> list[dict[str, object]]:
    completed = subprocess.run(
        [
            powershell_executable(),
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT_PATH),
            "-Mode",
            mode,
            "-PlanOnly",
            "-AsJson",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    payload = json.loads(completed.stdout)
    if isinstance(payload, dict):
        return [payload]
    return payload


def run_release_matrix(kind: str = "runtime") -> list[dict[str, object]]:
    completed = subprocess.run(
        [
            "python",
            str(MATRIX_SCRIPT_PATH),
            "--kind",
            kind,
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    return json.loads(completed.stdout)


def run_bundle_plan(target: str) -> dict[str, object]:
    completed = subprocess.run(
        [
            "python",
            str(BUNDLE_SCRIPT_PATH),
            "--target",
            target,
            "--plan",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    return json.loads(completed.stdout)


def test_integrated_release_script_defaults_to_windows_runtime_pack_plan() -> None:
    assert SCRIPT_PATH.exists()

    plan = run_release_plan()
    targets = {target["id"]: target for target in plan}

    assert set(targets) == {"windows-cpu", "windows-nvidia", "windows-dml"}
    assert all(target["buildProfile"] == "runtime-pack" for target in targets.values())
    assert targets["windows-cpu"]["runtime"] == "cpu"
    assert targets["windows-cpu"]["archiveName"] == "ShotSieve-windows-cpu-x64.zip"


def test_github_release_script_exists_and_checks_repo_cleanliness() -> None:
    script_text = GITHUB_RELEASE_SCRIPT_PATH.read_text(encoding="utf-8")

    assert '"status", "--porcelain"' in script_text
    assert '"rev-list", "--left-right", "--count"' in script_text
    assert "Test-CleanGitState" in script_text


def test_github_release_script_looks_up_latest_tag_and_prompts_for_version() -> None:
    script_text = GITHUB_RELEASE_SCRIPT_PATH.read_text(encoding="utf-8")

    assert '"describe", "--tags", "--abbrev=0"' in script_text
    assert "Read-Host" in script_text
    assert "Read-ReleaseVersion" in script_text
    assert "Latest tag" in script_text


def test_github_release_script_avoids_dead_hints_and_unapproved_verbs() -> None:
    script_text = GITHUB_RELEASE_SCRIPT_PATH.read_text(encoding="utf-8")

    assert "$GitStatusCommandHint" not in script_text
    assert "$GitDivergenceCommandHint" not in script_text
    assert "$LatestReleaseCommandHint" not in script_text
    assert "function Ensure-CleanGitState" not in script_text
    assert "function Prompt-Version" not in script_text
    assert "function Ensure-VersionNotAlreadyPublished" not in script_text
    assert "function Test-CleanGitState" in script_text
    assert "function Read-ReleaseVersion" in script_text
    assert "function Test-VersionNotAlreadyPublished" in script_text


def test_github_release_script_creates_and_pushes_tag_for_actions_release() -> None:
    script_text = GITHUB_RELEASE_SCRIPT_PATH.read_text(encoding="utf-8")

    assert "git tag" in script_text
    assert "git push origin" in script_text
    assert "GitHub Actions" in script_text
    assert "gh release create" not in script_text


def test_prepare_release_script_exists_and_updates_version_files(tmp_path: Path) -> None:
    repo, prepare_script = create_temp_prepare_release_workspace(tmp_path)

    completed = subprocess.run(
        [
            powershell_executable(),
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(prepare_script),
            "-Version",
            "v1.2.3",
            "-ReleaseDate",
            "2026-05-01",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=repo,
    )

    assert "Prepared release references for 1.2.3." in completed.stdout
    assert "./scripts/create_github_release.ps1 -Version v1.2.3" in completed.stdout
    assert 'version = "1.2.3"' in (repo / "pyproject.toml").read_text(encoding="utf-8")
    assert '__version__ = "1.2.3"' in (repo / "src" / "shotsieve" / "__init__.py").read_text(encoding="utf-8")
    assert "Version: 1.2.3" in (repo / "src" / "shotsieve.egg-info" / "PKG-INFO").read_text(encoding="utf-8")

    changelog_text = (repo / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "## [1.2.3] - 2026-05-01" in changelog_text
    assert "- Release notes pending." in changelog_text
    assert changelog_text.index("## [1.2.3] - 2026-05-01") < changelog_text.index("## [0.1.0] - 2026-04-25")


def test_prepare_release_script_dry_run_does_not_modify_files(tmp_path: Path) -> None:
    repo, prepare_script = create_temp_prepare_release_workspace(tmp_path)
    before_pyproject = (repo / "pyproject.toml").read_text(encoding="utf-8")
    before_changelog = (repo / "CHANGELOG.md").read_text(encoding="utf-8")

    completed = subprocess.run(
        [
            powershell_executable(),
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(prepare_script),
            "-Version",
            "2.0.0",
            "-ReleaseDate",
            "2026-06-01",
            "-DryRun",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=repo,
    )

    assert "Dry run complete for release 2.0.0." in completed.stdout
    assert "project version in pyproject.toml" in completed.stdout
    assert "./scripts/create_github_release.ps1 -Version v2.0.0" in completed.stdout
    assert (repo / "pyproject.toml").read_text(encoding="utf-8") == before_pyproject
    assert (repo / "CHANGELOG.md").read_text(encoding="utf-8") == before_changelog


def test_prepare_release_script_uses_approved_powershell_verbs() -> None:
    script_text = PREPARE_RELEASE_SCRIPT_PATH.read_text(encoding="utf-8")

    assert "function Read-PrepareReleaseVersion" in script_text
    assert "function ConvertTo-ReleaseVersion" in script_text
    assert "function Resolve-ProjectRoot" in script_text
    assert "function Update-ReleaseFiles" in script_text
    assert "function Ensure-ReleaseFiles" not in script_text
    assert "function Prompt-ReleaseVersion" not in script_text


def test_build_release_script_env_keeps_git_discoverable(tmp_path: Path) -> None:
    env = build_release_script_env(tmp_path)
    path_parts = env["PATH"].split(os.pathsep)

    assert path_parts[0] == str(tmp_path / "bin")
    assert shutil.which("git", path=env["PATH"])


def test_powershell_executable_returns_resolved_path() -> None:
    executable = Path(powershell_executable())

    assert executable.is_absolute()
    assert executable.exists()


def test_github_release_script_dry_run_can_succeed_without_gh_when_version_is_provided(tmp_path: Path) -> None:
    repo = create_temp_release_repo(tmp_path)
    env = build_release_script_env(tmp_path)

    completed = subprocess.run(
        [
            powershell_executable(),
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(GITHUB_RELEASE_SCRIPT_PATH),
            "-Version",
            "v9.9.9",
            "-DryRun",
            "-SkipFetch",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=repo,
        env=env,
    )

    assert "Dry run complete" in completed.stdout
    assert "git tag -a v9.9.9 -m 'Release v9.9.9'" in completed.stdout
    assert "git push origin v9.9.9" in completed.stdout
    assert "GitHub Actions release workflow will publish the release" in completed.stdout


def test_github_release_script_uses_windows_default_gh_install_when_not_on_path() -> None:
    script_text = GITHUB_RELEASE_SCRIPT_PATH.read_text(encoding="utf-8")

    assert "Resolve-CommonWindowsGhCommandPath" not in script_text
    assert "GitHub CLI\\\\gh.exe" not in script_text
    assert "GitHub CLI\\gh.exe" not in script_text


def test_github_release_script_rejects_existing_remote_tag(tmp_path: Path) -> None:
    repo = create_temp_release_repo(tmp_path)
    env = build_release_script_env(tmp_path)
    git = git_executable()
    subprocess.run([git, "-C", str(repo), "tag", "-a", "v9.9.9", "-m", "Remote tag"], check=True)
    subprocess.run([git, "-C", str(repo), "push", "origin", "v9.9.9"], check=True, capture_output=True, text=True)
    subprocess.run([git, "-C", str(repo), "tag", "-d", "v9.9.9"], check=True, capture_output=True, text=True)

    completed = subprocess.run(
        [
            powershell_executable(),
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(GITHUB_RELEASE_SCRIPT_PATH),
            "-Version",
            "v9.9.9",
            "-DryRun",
            "-SkipFetch",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=repo,
        env=env,
    )

    assert completed.returncode != 0
    assert "already exists on origin" in completed.stderr


def test_windows_nvidia_bootstrap_script_removed() -> None:
    assert not (PROJECT_ROOT / "scripts" / "build_windows_bootstrap_nvidia.ps1").exists()


def test_integrated_release_script_can_plan_single_windows_nvidia_target() -> None:
    completed = subprocess.run(
        [
            powershell_executable(),
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT_PATH),
            "-Mode",
            "runtime",
            "-TargetIds",
            "windows-nvidia",
            "-PlanOnly",
            "-AsJson",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )

    payload = json.loads(completed.stdout)
    plan = payload if isinstance(payload, list) else [payload]
    assert len(plan) == 1
    target = plan[0]
    assert target["id"] == "windows-nvidia"
    assert target["runtime"] == "cuda"
    assert target["buildProfile"] == "runtime-pack"
    assert target["executableName"] == "ShotSieve-NVIDIA.exe"
    assert target["archiveName"] == "ShotSieve-windows-nvidia-x64.zip"


def test_integrated_release_script_runtime_mode_plans_windows_runtime_packs_only() -> None:
    plan = run_release_plan("runtime")
    targets = {target["id"]: target for target in plan}

    assert set(targets) == {"windows-cpu", "windows-nvidia", "windows-dml"}
    assert all(target["buildProfile"] == "runtime-pack" for target in targets.values())


def test_integrated_release_script_uses_runtime_orchestrator_toolchain() -> None:
    script_text = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "scripts\\release_target_matrix.py" in script_text
    assert "scripts\\build_portable_bundle.py" in script_text
    assert "release-constraints.txt" in script_text
    assert "-m pip install pyinstaller -c" in script_text
    assert "Install-PyInstallerIfMissing" in script_text
    assert "Ensure-PyInstaller" not in script_text
    assert "generate_bootstrap_manifest.py" not in script_text
    assert "New-TargetPlan" not in script_text
    assert "ForceRecreateEnvs" not in script_text


def test_integrated_release_script_is_runtime_only_mode() -> None:
    script_text = SCRIPT_PATH.read_text(encoding="utf-8")

    assert '[ValidateSet("runtime")]' in script_text
    assert '[ValidateSet("runtime", "all")]' not in script_text
    assert "Resolve-MatrixKind" not in script_text
    assert "--kind runtime" in script_text


def test_release_matrix_script_is_runtime_only_interface() -> None:
    script_text = MATRIX_SCRIPT_PATH.read_text(encoding="utf-8")

    assert 'choices=("runtime",)' in script_text
    assert 'choices=("runtime", "all")' not in script_text


def test_release_matrix_script_rejects_removed_all_kind() -> None:
    completed = subprocess.run(
        [
            "python",
            str(MATRIX_SCRIPT_PATH),
            "--kind",
            "all",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )

    assert completed.returncode != 0
    assert "invalid choice" in completed.stderr


def test_integrated_release_script_installs_target_runtime_dependencies() -> None:
    script_text = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "function Install-TorchVariant" in script_text
    assert "--index-url https://download.pytorch.org/whl/cpu" in script_text
    assert "--index-url https://download.pytorch.org/whl/cu126" not in script_text
    assert "function Install-TargetDependencies" in script_text
    assert "-c $ConstraintsFile" in script_text
    assert "pip install -e \".[" in script_text


def test_integrated_release_script_marks_cuda_targets_to_skip_bundled_torch() -> None:
    script_text = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "SHOTSIEVE_SKIP_BUNDLED_TORCH" in script_text
    assert "torchVariant" in script_text
    assert "TargetIds" in script_text


def test_integrated_release_script_does_not_include_directml_torch26_override_switches() -> None:
    script_text = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "[switch]$ForceDirectMLTorch26" not in script_text
    assert "[switch]$DisableDirectMLTorch26Override" not in script_text
    assert "function Install-DirectMLTorch26Override" not in script_text
    assert "function Test-DirectMLRuntimeAvailable" not in script_text
    assert "function Install-DirectMLStableRuntime" not in script_text


def test_integrated_release_script_uses_stable_directml_runtime_path_without_torch26_overrides() -> None:
    script_text = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "Installing default Torch runtime for DirectML target" in script_text
    assert "torch==2.6.*" not in script_text
    assert "torchvision==0.21.*" not in script_text


def test_integrated_release_script_forces_torch_variant_reinstall_between_targets() -> None:
    script_text = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "--upgrade --force-reinstall --no-cache-dir" in script_text


def test_integrated_release_script_installs_dependencies_before_each_target_build() -> None:
    script_text = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "function Resolve-TargetPythonCommand" in script_text
    assert "Install-TargetDependencies -PythonCommand $targetPythonCommand -Target $target" in script_text
    assert "New-WindowsTargetBundle -PythonCommand $targetPythonCommand" in script_text
    assert "Build-WindowsTarget -PythonCommand $targetPythonCommand" not in script_text
    assert "Creating isolated build environment" in script_text


def test_integrated_release_script_recreates_broken_target_virtualenv(tmp_path: Path) -> None:
    target_id = "windows-nvidia"
    build_root = tmp_path / "build-root"
    target_venv_root = build_root / target_id / ".venv"
    target_python = target_venv_root / "Scripts" / "python.exe"
    pyvenv_cfg = target_venv_root / "pyvenv.cfg"

    target_python.parent.mkdir(parents=True, exist_ok=True)
    target_python.write_text("broken-venv", encoding="utf-8")
    assert not pyvenv_cfg.exists()

    completed = subprocess.run(
        [
            powershell_executable(),
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            (
                "$ErrorActionPreference='Stop'; "
                f". '{SCRIPT_PATH}'; "
                f"$python = Resolve-TargetPythonCommand -BasePythonCommand '{sys.executable}' "
                f"-ResolvedBuildRoot '{build_root}' -TargetId '{target_id}'; "
                "Write-Output $python; "
                f"Write-Output ('HAS_PYVENV=' + (Test-Path '{pyvenv_cfg}'))"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )

    assert pyvenv_cfg.exists()
    assert "HAS_PYVENV=True" in completed.stdout


def test_integrated_release_script_recreates_target_virtualenv_when_python_startup_fails(
    tmp_path: Path,
) -> None:
    target_id = "windows-nvidia"
    build_root = tmp_path / "build-root"
    target_venv_root = build_root / target_id / ".venv"
    target_python = target_venv_root / "Scripts" / "python.exe"
    pyvenv_cfg = target_venv_root / "pyvenv.cfg"

    target_python.parent.mkdir(parents=True, exist_ok=True)
    target_python.write_text("broken-venv", encoding="utf-8")
    pyvenv_cfg.write_text("home = C:/broken", encoding="utf-8")

    completed = subprocess.run(
        [
            powershell_executable(),
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            (
                "$ErrorActionPreference='Stop'; "
                f". '{SCRIPT_PATH}'; "
                f"$python = Resolve-TargetPythonCommand -BasePythonCommand '{sys.executable}' "
                f"-ResolvedBuildRoot '{build_root}' -TargetId '{target_id}'; "
                "Write-Output $python; "
                f"Write-Output ('HAS_PYVENV=' + (Test-Path '{pyvenv_cfg}'))"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )

    assert pyvenv_cfg.exists()
    assert "HAS_PYVENV=True" in completed.stdout


def test_integrated_release_script_uses_approved_powershell_verbs_for_custom_functions() -> None:
    script_text = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "function Ensure-PyInstaller" not in script_text
    assert "function Build-WindowsTarget" not in script_text
    assert "function Install-PyInstallerIfMissing" in script_text
    assert "function New-WindowsTargetBundle" in script_text


def test_integrated_release_script_installs_editable_project_from_repo_root() -> None:
    script_text = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "Push-Location $ProjectRoot" in script_text
    assert "pip install -e \".[" in script_text
    assert "Pop-Location" in script_text


def test_dead_import_validation_dependencies_include_ruff() -> None:
    pyproject_text = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    pyproject = tomllib.loads(pyproject_text)

    test_dependencies = pyproject["project"]["optional-dependencies"]["test"]
    lint_dependencies = pyproject["project"]["optional-dependencies"]["lint"]

    assert any(entry.startswith("ruff") for entry in test_dependencies)
    assert any(entry.startswith("ruff") for entry in lint_dependencies)


def test_local_only_blog_and_github_automation_paths_are_ignored() -> None:
    completed = subprocess.run(
        [
            "git",
            "check-ignore",
            "--verbose",
            "--no-index",
            *LOCAL_ONLY_REPO_RELATIVE_PATHS,
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )

    normalized_lines = [line.replace("\\", "/") for line in completed.stdout.splitlines() if line.strip()]

    assert any(".gitignore:" in line and ":blog.md" in line and line.endswith("blog.md") for line in normalized_lines)
    assert any(
        ".gitignore:" in line and ":.github/agents/" in line and line.endswith(".github/agents/anvil.agent.md")
        for line in normalized_lines
    )


def test_release_workflow_file_is_tracked_for_automation() -> None:
    completed = subprocess.run(
        ["git", "ls-files", "--error-unmatch", ".github/workflows/release.yml"],
        check=False,
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )

    assert completed.returncode == 0


def test_release_workflow_file_is_not_ignored() -> None:
    completed = subprocess.run(
        ["git", "check-ignore", "--verbose", ".github/workflows/release.yml"],
        check=False,
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )

    assert completed.returncode != 0


def test_local_only_blog_and_github_automation_paths_are_not_tracked() -> None:
    for relative_path in LOCAL_ONLY_REPO_RELATIVE_PATHS:
        completed = subprocess.run(
            ["git", "ls-files", "--error-unmatch", relative_path],
            check=False,
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )

        assert completed.returncode != 0, f"Expected {relative_path} to stay local-only, but it is tracked"


def test_build_guide_documents_repo_wide_dead_import_validation() -> None:
    build_doc_text = (PROJECT_ROOT / "docs" / "building.md").read_text(encoding="utf-8")

    assert 'python -m pip install -e .[lint]' in build_doc_text
    assert 'python -m ruff check --select F401 src/shotsieve' in build_doc_text


def test_build_guide_documents_release_builds_and_windows_runtime_script() -> None:
    build_doc_text = (PROJECT_ROOT / "docs" / "building.md").read_text(encoding="utf-8")

    assert "## Release builds and portable bundles" in build_doc_text
    assert "./scripts/build_windows_releases.ps1" in build_doc_text


def test_build_guide_documents_prepare_then_publish_release_flow() -> None:
    build_doc_text = (PROJECT_ROOT / "docs" / "building.md").read_text(encoding="utf-8")

    assert "./scripts/prepare_release.ps1 -Version 0.2.0" in build_doc_text
    assert "./scripts/create_github_release.ps1 -Version v0.2.0" in build_doc_text
    assert "does **not** edit version files" in build_doc_text


def test_build_guide_clarifies_xpu_is_source_only_not_packaged() -> None:
    build_doc_text = (PROJECT_ROOT / "docs" / "building.md").read_text(encoding="utf-8")

    assert "Intel XPU remains a source-only runtime path today" in build_doc_text
    assert "there is no prebuilt XPU runtime-pack target" in build_doc_text


def test_pyproject_does_not_expose_removed_cli_entry_point() -> None:
    pyproject_text = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    pyproject = tomllib.loads(pyproject_text)
    scripts = pyproject.get("project", {}).get("scripts", {})

    assert "shotsieve" not in scripts
    assert scripts.get("shotsieve-desktop") == "shotsieve.desktop:main"


def test_cli_module_file_has_been_removed() -> None:
    assert not (PROJECT_ROOT / "src" / "shotsieve" / "cli.py").exists()


def test_cli_module_is_no_longer_importable() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("shotsieve.cli")


def test_target_modules_do_not_keep_dead_imports() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "--select",
            "F401",
            "src/shotsieve",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_directml_extra_accepts_available_prerelease_torch_directml() -> None:
    pyproject_text = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    pyproject = tomllib.loads(pyproject_text)
    directml_dependencies = pyproject["project"]["optional-dependencies"]["learned-iqa-directml"]

    torch_directml_dependency = next(
        (entry for entry in directml_dependencies if entry.startswith("torch-directml")),
        "",
    )

    assert torch_directml_dependency
    assert ">=0.2.5.dev0" in torch_directml_dependency
    assert "python_version < '3.13'" in torch_directml_dependency


def test_learned_iqa_extras_include_icecream_dependency() -> None:
    pyproject_text = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    pyproject = tomllib.loads(pyproject_text)

    learned_iqa_dependencies = pyproject["project"]["optional-dependencies"]["learned-iqa"]
    directml_dependencies = pyproject["project"]["optional-dependencies"]["learned-iqa-directml"]

    assert any(entry.startswith("icecream") for entry in learned_iqa_dependencies)
    assert any(entry.startswith("icecream") for entry in directml_dependencies)


def test_windows_build_dependencies_pin_setuptools_with_pkg_resources() -> None:
    pyproject_text = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    pyproject = tomllib.loads(pyproject_text)

    windows_build_dependencies = pyproject["project"]["optional-dependencies"]["windows-build"]

    assert any(entry.startswith("setuptools<81") for entry in windows_build_dependencies)


def test_tier1_release_matrix_covers_all_runtime_pack_targets() -> None:
    assert MATRIX_SCRIPT_PATH.exists()

    matrix = run_release_matrix("runtime")
    targets = {target["id"]: target for target in matrix}

    assert set(targets) == {
        "windows-cpu",
        "windows-nvidia",
        "windows-dml",
        "linux-cpu",
        "linux-nvidia",
        "macos-cpu",
        "macos-mps",
    }

    assert targets["windows-cpu"]["runsOn"] == "windows-latest"
    assert targets["linux-cpu"]["runsOn"] == "ubuntu-latest"
    assert targets["macos-mps"]["runsOn"] == "macos-latest"
    assert targets["windows-dml"]["extras"] == ["format-loaders", "learned-iqa-directml", "windows-build"]
    assert targets["linux-nvidia"]["torchVariant"] == "cuda"
    assert targets["macos-mps"]["runtime"] == "mps"
    assert _string_value(targets["windows-cpu"]["archiveName"]).endswith(".zip")
    assert _string_value(targets["linux-cpu"]["archiveName"]).endswith(".tar.gz")
    assert _string_value(targets["macos-cpu"]["archiveName"]).endswith(".tar.gz")


def test_release_matrix_script_only_emits_runtime_targets() -> None:
    matrix = run_release_matrix("runtime")
    target_ids = {target["id"] for target in matrix}
    assert all("bootstrap" not in _string_value(target_id) for target_id in target_ids)


def test_portable_bundle_builder_exposes_runtime_pack_target_plan() -> None:
    assert BUNDLE_SCRIPT_PATH.exists()

    plan = run_bundle_plan("linux-nvidia")
    target = _dict_value(plan["target"])

    assert target["id"] == "linux-nvidia"
    assert target["runtime"] == "cuda"
    assert target["variantFolderName"] == "ShotSieve-linux-nvidia"
    assert _string_value(plan["archivePath"]).endswith("ShotSieve-linux-nvidia-x64.tar.gz")
    assert _string_value(plan["distPath"]).endswith("ShotSieve-linux-nvidia")


def test_portable_bundle_target_plan_uses_typed_plan_contract() -> None:
    module_name = "build_portable_bundle_script_annotations"
    spec = importlib.util.spec_from_file_location(module_name, BUNDLE_SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    type_hints = typing.get_type_hints(module.target_plan)

    assert type_hints["return"] is module.BundlePlan


def test_portable_bundle_builder_rejects_removed_bootstrap_target() -> None:
    completed = subprocess.run(
        [
            "python",
            str(BUNDLE_SCRIPT_PATH),
            "--target",
            "windows-bootstrap",
            "--plan",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )

    assert completed.returncode != 0
    assert "Unknown release target 'windows-bootstrap'" in (completed.stderr + completed.stdout)


def test_release_constraints_file_exists_with_required_pins() -> None:
    assert RELEASE_CONSTRAINTS_PATH.exists()

    constraints_text = RELEASE_CONSTRAINTS_PATH.read_text(encoding="utf-8")
    assert "setuptools<81" in constraints_text
    assert "pyinstaller>=6.19,<7" in constraints_text


def test_release_targets_module_does_not_define_bootstrap_matrix_helpers() -> None:
    module_text = (PROJECT_ROOT / "src" / "shotsieve" / "release_targets.py").read_text(encoding="utf-8")

    assert "def bootstrap_release_targets" not in module_text
    assert "def bootstrap_release_matrix" not in module_text


def test_scripts_folder_no_longer_contains_bootstrap_release_helpers() -> None:
    assert not BOOTSTRAP_MANIFEST_SCRIPT_PATH.exists()
    assert not EMBED_BOOTSTRAP_MANIFEST_SCRIPT_PATH.exists()


def test_bootstrap_pyinstaller_spec_has_been_removed() -> None:
    assert not (PROJECT_ROOT / "shotsieve_bootstrap.spec").exists()


def test_pyinstaller_spec_collects_clip_vocabulary_data_files() -> None:
    spec_text = SPEC_PATH.read_text(encoding="utf-8")

    assert "collect_data_files(\"clip\")" in spec_text


def test_pyinstaller_spec_collects_icecream_package() -> None:
    spec_text = SPEC_PATH.read_text(encoding="utf-8")

    assert '"icecream"' in spec_text


def test_pyinstaller_spec_collects_pkg_resources_for_clipiqa_runtime() -> None:
    spec_text = SPEC_PATH.read_text(encoding="utf-8")

    assert '"pkg_resources"' in spec_text


def test_pyinstaller_spec_collects_modulefinder_for_learned_iqa_runtime() -> None:
    spec_text = SPEC_PATH.read_text(encoding="utf-8")

    assert '"modulefinder"' in spec_text


def test_pyinstaller_spec_hard_excludes_torch_when_skip_bundled_torch_enabled() -> None:
    spec_text = SPEC_PATH.read_text(encoding="utf-8")

    assert "skip_bundled_torch" in spec_text
    assert "analysis_excludes" in spec_text
    assert "if skip_bundled_torch:" in spec_text
    assert "_is_torch_related" in spec_text
    assert "hiddenimports = [entry for entry in hiddenimports if not _is_torch_related(entry)]" in spec_text
    assert "datas = [entry for entry in datas if not _is_torch_related(entry)]" in spec_text
    assert "binaries = [entry for entry in binaries if not _is_torch_related(entry)]" in spec_text
    assert "excludes=analysis_excludes" in spec_text


def test_portable_bundle_builder_preserves_target_build_root_and_venv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module_name = "build_portable_bundle_script"
    spec = importlib.util.spec_from_file_location(module_name, BUNDLE_SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "shotsieve.spec").write_text("# stub spec", encoding="utf-8")

    target = SimpleNamespace(
        id="windows-cpu",
        platform="windows",
        specPath="shotsieve.spec",
        variantFolderName="ShotSieve-windows-cpu",
        archiveName="ShotSieve-windows-cpu-x64.zip",
        executableName="ShotSieve-windows-cpu.exe",
        to_json=lambda: {"id": "windows-cpu"},
    )

    dist_root = tmp_path / "dist"
    build_root = tmp_path / "build"
    target_build_root = build_root / target.id
    pyinstaller_dist_root = target_build_root / "dist"
    pyinstaller_work_root = target_build_root / "work"
    venv_python = target_build_root / ".venv" / "Scripts" / "python.exe"

    pyinstaller_dist_root.mkdir(parents=True, exist_ok=True)
    pyinstaller_work_root.mkdir(parents=True, exist_ok=True)
    venv_python.parent.mkdir(parents=True, exist_ok=True)
    venv_python.write_text("placeholder", encoding="utf-8")

    removed_paths: list[Path] = []
    real_rmtree = module.shutil.rmtree

    def fake_rmtree(path: str | Path) -> None:
        removed_paths.append(Path(path).resolve())
        if Path(path).exists():
            real_rmtree(path)

    def fake_pyinstaller_run(cmd: list[str], check: bool, cwd: Path) -> subprocess.CompletedProcess[str]:
        dist_idx = cmd.index("--distpath") + 1
        generated_dist = Path(cmd[dist_idx])
        bundle_dir = generated_dist / "ShotSieve"
        bundle_dir.mkdir(parents=True, exist_ok=True)
        (bundle_dir / "ShotSieve.exe").write_text("launcher", encoding="utf-8")
        return subprocess.CompletedProcess(args=cmd, returncode=0)

    monkeypatch.setattr(module.shutil, "rmtree", fake_rmtree)
    monkeypatch.setattr(module.subprocess, "run", fake_pyinstaller_run)

    plan = module.build_bundle(
        target,
        project_root=project_root,
        dist_root=dist_root,
        build_root=build_root,
    )

    assert Path(plan["archivePath"]).exists()
    assert venv_python.exists()
    assert target_build_root.resolve() not in removed_paths


def test_portable_bundle_zip_is_flat_without_variant_folder_prefix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module_name = "build_portable_bundle_script_flat_zip"
    spec = importlib.util.spec_from_file_location(module_name, BUNDLE_SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "shotsieve.spec").write_text("# stub spec", encoding="utf-8")

    target = SimpleNamespace(
        id="windows-cpu",
        platform="windows",
        specPath="shotsieve.spec",
        variantFolderName="ShotSieve-windows-cpu",
        archiveName="ShotSieve-windows-cpu-x64.zip",
        executableName="ShotSieve-CPU.exe",
        to_json=lambda: {"id": "windows-cpu"},
    )

    dist_root = tmp_path / "dist"
    build_root = tmp_path / "build"

    def fake_pyinstaller_run(cmd: list[str], check: bool, cwd: Path) -> subprocess.CompletedProcess[str]:
        dist_idx = cmd.index("--distpath") + 1
        generated_dist = Path(cmd[dist_idx])
        bundle_dir = generated_dist / "ShotSieve"
        bundle_dir.mkdir(parents=True, exist_ok=True)
        (bundle_dir / "ShotSieve.exe").write_text("launcher", encoding="utf-8")
        (bundle_dir / "_internal").mkdir(parents=True, exist_ok=True)
        (bundle_dir / "_internal" / "dummy.txt").write_text("payload", encoding="utf-8")
        return subprocess.CompletedProcess(args=cmd, returncode=0)

    monkeypatch.setattr(module.subprocess, "run", fake_pyinstaller_run)

    plan = module.build_bundle(
        target,
        project_root=project_root,
        dist_root=dist_root,
        build_root=build_root,
    )

    archive_path = Path(plan["archivePath"])
    assert archive_path.exists()

    with zipfile.ZipFile(archive_path, "r") as archive:
        names = archive.namelist()

    assert "ShotSieve-CPU.exe" in names
    assert "_internal/dummy.txt" in names
    assert all(not name.startswith("ShotSieve-windows-cpu/") for name in names)