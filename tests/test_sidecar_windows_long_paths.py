from __future__ import annotations

from pathlib import Path

import pytest

from shotsieve import bootstrap_sidecar as sidecar_module


@pytest.mark.parametrize(
    "entry_name",
    (
        "torch-2.14.0+xpu.dist-info",
        "torch-2.13.0+rocm10.0.0.dist-info",
        "torchvision-0.29.0+xpu.dist-info",
        "rocm-10.0.0.dist-info",
        "_rocm_sdk_core-7.1.0.dist-info",
        "nvidia_cuda_runtime_cu13-13.0.96.dist-info",
        "intel_sycl_rt-2026.1.0.dist-info",
        "onemkl_sycl_blas-2026.0.0.dist-info",
    ),
)
def test_runtime_metadata_entries_are_preserved_during_learned_repair(entry_name: str) -> None:
    assert sidecar_module._is_preserved_runtime_metadata_entry(entry_name)
    assert sidecar_module._is_preserved_runtime_sidecar_entry(entry_name)


@pytest.mark.parametrize(
    "entry_name",
    (
        "pyiqa-0.1.16.dist-info",
        "transformers-5.17.0.dist-info",
        "timm-1.0.30.dist-info",
    ),
)
def test_learned_metadata_entries_remain_replaceable(entry_name: str) -> None:
    assert not sidecar_module._is_preserved_runtime_metadata_entry(entry_name)
    assert not sidecar_module._is_preserved_runtime_sidecar_entry(entry_name)


@pytest.mark.parametrize(
    ("runtime", "metadata_name", "deep_relative_path"),
    (
        (
            "xpu",
            "torch-2.14.0+xpu.dist-info",
            Path("licenses/third_party/flash-attention/csrc/composable_kernel/docs/LICENSE.rst"),
        ),
        (
            "rocm",
            "torch-2.13.0+rocm10.0.0.dist-info",
            Path(
                "licenses/third_party/kineto/libkineto/third_party/dynolog/"
                "third_party/DCGM/testing/python3/marker.txt"
            ),
        ),
    ),
)
def test_learned_staging_skips_deep_runtime_metadata_before_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runtime: str,
    metadata_name: str,
    deep_relative_path: Path,
) -> None:
    source_dir = tmp_path / f"windows-{runtime}"
    staging_dir = tmp_path / ".s123456"
    (source_dir / "torch").mkdir(parents=True)
    (source_dir / "torch" / "__init__.py").write_text("torch", encoding="utf-8")
    deep_marker = source_dir / metadata_name / deep_relative_path
    deep_marker.parent.mkdir(parents=True)
    deep_marker.write_text("runtime metadata", encoding="utf-8")
    staging_dir.mkdir()

    real_copytree = sidecar_module.shutil.copytree
    attempted_runtime_metadata_copies: list[str] = []

    def guarded_copytree(source, destination, *args, **kwargs):
        source_path = Path(source)
        if sidecar_module._is_preserved_runtime_metadata_entry(source_path.name):
            attempted_runtime_metadata_copies.append(source_path.name)
            raise AssertionError(f"runtime metadata should not be copied: {source_path}")
        return real_copytree(source, destination, *args, **kwargs)

    monkeypatch.setattr(sidecar_module.shutil, "copytree", guarded_copytree)

    sidecar_module._prepare_learned_iqa_staging(
        source_dir=source_dir,
        staging_dir=staging_dir,
        runtime=runtime,
    )

    assert attempted_runtime_metadata_copies == []
    assert (staging_dir / "torch" / "__init__.py").read_text(encoding="utf-8") == "torch"
    assert not (staging_dir / metadata_name).exists()
    assert deep_marker.exists()


@pytest.mark.parametrize(
    "metadata_name",
    (
        "torch-2.14.0+xpu.dist-info",
        "torch-2.13.0+rocm10.0.0.dist-info",
    ),
)
def test_learned_backup_and_cleanup_leave_runtime_metadata_in_place(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    metadata_name: str,
) -> None:
    site_packages = tmp_path / "windows-runtime"
    rollback_dir = tmp_path / ".rollback"
    runtime_marker = (
        site_packages
        / metadata_name
        / "licenses"
        / "third_party"
        / "flash-attention"
        / "third_party"
        / "aiter"
        / "3rdparty"
        / "composable_kernel"
        / "docs"
        / "LICENSE.rst"
    )
    runtime_marker.parent.mkdir(parents=True)
    runtime_marker.write_text("runtime license", encoding="utf-8")
    (site_packages / "pyiqa").mkdir()
    (site_packages / "pyiqa" / "old.py").write_text("old learned package", encoding="utf-8")

    real_copytree = sidecar_module.shutil.copytree
    attempted_runtime_metadata_copies: list[str] = []

    def guarded_copytree(source, destination, *args, **kwargs):
        source_path = Path(source)
        if sidecar_module._is_preserved_runtime_metadata_entry(source_path.name):
            attempted_runtime_metadata_copies.append(source_path.name)
            raise AssertionError(f"runtime metadata should not be copied: {source_path}")
        return real_copytree(source, destination, *args, **kwargs)

    monkeypatch.setattr(sidecar_module.shutil, "copytree", guarded_copytree)

    sidecar_module._copy_learned_sidecar_entries(site_packages, rollback_dir)

    assert attempted_runtime_metadata_copies == []
    assert not (rollback_dir / metadata_name).exists()
    assert (rollback_dir / "pyiqa" / "old.py").read_text(encoding="utf-8") == "old learned package"

    sidecar_module._remove_learned_sidecar_entries(site_packages)

    assert runtime_marker.read_text(encoding="utf-8") == "runtime license"
    assert not (site_packages / "pyiqa").exists()
