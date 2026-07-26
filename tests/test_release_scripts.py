from __future__ import annotations

import importlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import cast

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "build_windows_releases.ps1"
GITHUB_RELEASE_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "create_github_release.ps1"
PREPARE_RELEASE_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "prepare_release.ps1"
MATRIX_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "release_target_matrix.py"
BUNDLE_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "build_portable_bundle.py"
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


def run_release_matrix(kind: str) -> list[dict[str, object]]:
    completed = subprocess.run(
        [
            sys.executable,
            str(MATRIX_SCRIPT_PATH),
            "--kind",
            kind,
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    return cast(list[dict[str, object]], json.loads(completed.stdout))


def test_windows_release_script_defines_all_runtime_targets() -> None:
    script_text = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "function Get-WindowsTargets" in script_text
    assert 'Where-Object { $_.id -like "windows-*" }' in script_text


def test_windows_release_script_validates_target_arguments() -> None:
    completed = subprocess.run(
        [
            powershell_executable(),
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT_PATH),
            "-TargetIds",
            "invalid-target",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )

    assert completed.returncode != 0
    assert "Failed to install" in completed.stderr or "invalid" in completed.stderr.lower()


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


def test_windows_build_dependencies_pin_pip_below_26_for_embedded_runtime_installs() -> None:
    pyproject_text = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    pyproject = tomllib.loads(pyproject_text)

    windows_build_dependencies = pyproject["project"]["optional-dependencies"]["windows-build"]

    assert any(entry.startswith("pip<26") for entry in windows_build_dependencies)


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