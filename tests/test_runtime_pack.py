from __future__ import annotations

import json
from pathlib import Path

import pytest

from shotsieve.bootstrap_sidecar import (
    ROCM_LINUX_PACKAGE_URLS,
    ROCM_WINDOWS_PACKAGE_URLS,
    sidecar_site_packages_candidates,
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
        assert all("repo.radeon.com" in package for package in plan.packages)
        assert "https://pypi.org/simple" not in plan.index_args
    elif target.platform == "macos":
        assert plan.index_args == ()


def test_rocm_plans_use_the_exact_platform_wheels() -> None:
    assert torch_install_plan(target_id="linux-amd-rocm").packages == ROCM_LINUX_PACKAGE_URLS
    assert torch_install_plan(target_id="windows-amd-rocm").packages == ROCM_WINDOWS_PACKAGE_URLS


def test_legacy_sidecar_aliases_are_read_but_new_target_is_canonical(tmp_path: Path) -> None:
    candidates = sidecar_site_packages_candidates(tmp_path, "windows-nvidia-cuda")
    assert candidates[0].name == "windows-nvidia-cuda"
    assert candidates[1].name == "windows-nvidia"

    legacy = candidates[1]
    (legacy / "torch").mkdir(parents=True)
    (legacy / "torch" / "__init__.py").write_text("# legacy\n", encoding="utf-8")
    assert torch_sidecar_is_valid(legacy, target_id="windows-nvidia-cuda", runtime="cuda")


def test_partial_or_mismatched_sidecar_is_not_valid(tmp_path: Path) -> None:
    site_packages = tmp_path / "windows-cpu"
    (site_packages / "torch").mkdir(parents=True)
    assert not torch_sidecar_is_valid(site_packages, target_id="windows-cpu", runtime="cpu")

    (site_packages / "torch" / "__init__.py").write_text("# torch\n", encoding="utf-8")
    (site_packages / ".shotsieve-runtime.json").write_text(
        json.dumps(
            {
                "schema": 1,
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
