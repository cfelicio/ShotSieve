from __future__ import annotations

import importlib
import importlib.util
import json
import subprocess
import sys
import typing
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "build_windows_releases.ps1"
MATRIX_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "release_target_matrix.py"
BUNDLE_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "build_portable_bundle.py"
BOOTSTRAP_MANIFEST_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "generate_bootstrap_manifest.py"
EMBED_BOOTSTRAP_MANIFEST_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "embed_manifest_in_bootstrap_archives.py"
SPEC_PATH = PROJECT_ROOT / "shotsieve.spec"
RELEASE_CONSTRAINTS_PATH = PROJECT_ROOT / "scripts" / "release-constraints.txt"


def _dict_value(value: object) -> dict[str, object]:
    return cast(dict[str, object], value)


def _string_value(value: object) -> str:
    return str(value)


def run_bundle_plan(target_id: str) -> dict[str, object]:
    completed = subprocess.run(
        [
            sys.executable,
            str(BUNDLE_SCRIPT_PATH),
            "--target",
            target_id,
            "--plan",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    return cast(dict[str, object], json.loads(completed.stdout))


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
            sys.executable,
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
    assert "pip<26" in constraints_text
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


def test_portable_bundle_builder_falls_back_when_existing_staged_bundle_is_locked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "build_portable_bundle_script_locked_staging"
    spec = importlib.util.spec_from_file_location(module_name, BUNDLE_SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "shotsieve.spec").write_text("# stub spec", encoding="utf-8")

    target = SimpleNamespace(
        id="windows-nvidia",
        platform="windows",
        specPath="shotsieve.spec",
        variantFolderName="ShotSieve-windows-nvidia",
        archiveName="ShotSieve-windows-nvidia-x64.zip",
        executableName="ShotSieve-NVIDIA.exe",
        to_json=lambda: {"id": "windows-nvidia"},
    )

    dist_root = tmp_path / "dist"
    build_root = tmp_path / "build"
    locked_staged_bundle = dist_root / target.variantFolderName
    locked_payload = locked_staged_bundle / "data" / "runtime" / "site-packages" / "windows-nvidia" / "torch" / "lib"
    locked_payload.mkdir(parents=True, exist_ok=True)
    (locked_payload / "c10.dll").write_text("locked", encoding="utf-8")

    real_rmtree = module.shutil.rmtree

    def fake_rmtree(path: str | Path) -> None:
        resolved = Path(path).resolve()
        if resolved == locked_staged_bundle.resolve():
            raise PermissionError("locked c10.dll")
        if Path(path).exists():
            real_rmtree(path)

    def fake_pyinstaller_run(cmd: list[str], check: bool, cwd: Path) -> subprocess.CompletedProcess[str]:
        dist_idx = cmd.index("--distpath") + 1
        generated_dist = Path(cmd[dist_idx])
        bundle_dir = generated_dist / "ShotSieve"
        bundle_dir.mkdir(parents=True, exist_ok=True)
        (bundle_dir / "ShotSieve.exe").write_text("launcher", encoding="utf-8")
        (bundle_dir / "_internal").mkdir(parents=True, exist_ok=True)
        (bundle_dir / "_internal" / "dummy.txt").write_text("payload", encoding="utf-8")
        return subprocess.CompletedProcess(args=cmd, returncode=0)

    monkeypatch.setattr(module.shutil, "rmtree", fake_rmtree)
    monkeypatch.setattr(module.subprocess, "run", fake_pyinstaller_run)

    plan = module.build_bundle(
        target,
        project_root=project_root,
        dist_root=dist_root,
        build_root=build_root,
    )

    rebuilt_bundle = Path(plan["distPath"])
    archive_path = Path(plan["archivePath"])

    assert rebuilt_bundle.exists()
    assert rebuilt_bundle != locked_staged_bundle
    assert rebuilt_bundle.name.startswith("ShotSieve-windows-nvidia-rebuilt")
    assert (rebuilt_bundle / "ShotSieve-NVIDIA.exe").exists()
    assert locked_staged_bundle.exists()
    assert archive_path.exists()

    with zipfile.ZipFile(archive_path, "r") as archive:
        names = archive.namelist()

    assert "ShotSieve-NVIDIA.exe" in names
    assert "_internal/dummy.txt" in names


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
