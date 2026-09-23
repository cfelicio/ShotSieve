from __future__ import annotations

import json
from pathlib import Path

import pytest

from shotsieve.bootstrap_sidecar import (
    ROCM_PYTHON_INDEX_URL,
    ROCM_TORCH_REQUIREMENTS,
    SIDECAR_STATE_VERSION,
    sidecar_site_packages_dir,
    torch_install_plan,
    torch_sidecar_is_valid,
)
from shotsieve.desktop import runtime_target_id_from_executable_name
from shotsieve.release_targets import runtime_pack_release_targets


@pytest.mark.parametrize("target", runtime_pack_release_targets())
def test_every_runtime_pack_has_a_torchless_target_plan(target) -> None:
    plan = torch_install_plan(target_id=target.id)

    assert plan.target_id == target.id
    assert plan.runtime == target.runtime
    assert "torch" in " ".join(plan.packages).casefold()
    if target.torchVariant == "xpu":
        assert "https://download.pytorch.org/whl/xpu" in plan.index_args
        assert any("+xpu" in package for package in plan.packages)
    elif target.torchVariant == "rocm":
        assert plan.packages == ROCM_TORCH_REQUIREMENTS
        assert ROCM_PYTHON_INDEX_URL in plan.index_args
        assert "https://pypi.org/simple" in plan.index_args
    elif target.platform == "macos":
        assert plan.index_args == ()


@pytest.mark.parametrize("platform", ("windows", "linux"))
def test_amd_plan_uses_the_same_stable_rocm10_stack(platform: str) -> None:
    plan = torch_install_plan(target_id=f"{platform}-amd-rocm")

    assert plan.runtime == "rocm"
    assert plan.packages == ROCM_TORCH_REQUIREMENTS
    assert ROCM_PYTHON_INDEX_URL in plan.index_args
    assert "https://pypi.org/simple" in plan.index_args


def test_rocm_is_not_offered_for_unsupported_platform() -> None:
    with pytest.raises(ValueError, match="ROCm is not supported"):
        torch_install_plan(target_id="macos-amd-rocm")


def test_sidecar_uses_only_the_current_target_location(tmp_path: Path) -> None:
    current = sidecar_site_packages_dir(tmp_path, "windows-nvidia-cuda")
    legacy = sidecar_site_packages_dir(tmp_path, "windows-nvidia")
    assert current != legacy
    assert current.name == "windows-nvidia-cuda"

    (legacy / "torch").mkdir(parents=True)
    (legacy / "torch" / "__init__.py").write_text("# old target\n", encoding="utf-8")
    assert not torch_sidecar_is_valid(legacy, target_id="windows-nvidia-cuda", runtime="cuda")


def test_partial_or_mismatched_sidecar_is_not_valid(tmp_path: Path) -> None:
    site_packages = tmp_path / "windows-cpu"
    (site_packages / "torch").mkdir(parents=True)
    assert not torch_sidecar_is_valid(site_packages, target_id="windows-cpu", runtime="cpu")


def test_previous_sidecar_schema_is_invalidated_after_dependency_fix(tmp_path: Path) -> None:
    site_packages = tmp_path / "windows-intel-xpu"
    (site_packages / "torch").mkdir(parents=True)
    (site_packages / "torch" / "__init__.py").write_text("# torch\n", encoding="utf-8")
    (site_packages / ".shotsieve-runtime.json").write_text(
        json.dumps(
            {
                "schema": SIDECAR_STATE_VERSION - 1,
                "kind": "runtime",
                "complete": True,
                "torch_complete": True,
                "plan": torch_install_plan(target_id=site_packages.name, runtime="xpu").to_json(),
            }
        ),
        encoding="utf-8",
    )

    assert not torch_sidecar_is_valid(site_packages, target_id=site_packages.name, runtime="xpu")

    (site_packages / "torch" / "__init__.py").write_text("# torch\n", encoding="utf-8")
    (site_packages / ".shotsieve-runtime.json").write_text(
        json.dumps(
            {
                "schema": SIDECAR_STATE_VERSION,
                "kind": "runtime",
                "complete": True,
                "torch_complete": True,
                "plan": {"target_id": "linux-cpu", "source_fingerprint": "wrong"},
            }
        ),
        encoding="utf-8",
    )
    assert not torch_sidecar_is_valid(site_packages, target_id="windows-cpu", runtime="cpu")


@pytest.mark.parametrize(
    ("launcher", "system", "target"),
    (
        ("ShotSieve-CPU.exe", "Windows", "windows-cpu"),
        ("ShotSieve-NVIDIA-CUDA.exe", "Windows", "windows-nvidia-cuda"),
        ("ShotSieve-Intel-XPU", "Linux", "linux-intel-xpu"),
        ("ShotSieve-AMD-ROCm", "Linux", "linux-amd-rocm"),
        ("ShotSieve-Apple-MPS", "Darwin", "macos-apple-mps"),
    ),
)
def test_launcher_names_select_target_specific_sidecars(
    launcher: str,
    system: str,
    target: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sys.frozen", True, raising=False)
    monkeypatch.setattr("sys.executable", str(Path("runtime") / launcher), raising=False)
    assert runtime_target_id_from_executable_name(system_name=system) == target
