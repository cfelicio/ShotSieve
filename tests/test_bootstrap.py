from __future__ import annotations

from email.message import Message
import io
import json
import os
import sys
import types
from pathlib import Path
import urllib.error
from typing import Any
import warnings

import pytest

from shotsieve import bootstrap_assets as bootstrap_module
from shotsieve import bootstrap_sidecar
from shotsieve import runtime_support


def _new_module(name: str) -> Any:
    return types.ModuleType(name)


def _not_found_http_error(url: str) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(url=url, code=404, msg="Not Found", hdrs=Message(), fp=None)


def test_coerce_pip_main_return_code_handles_none_int_and_unexpected_values() -> None:
    assert bootstrap_sidecar._coerce_pip_main_return_code(None) == 0
    assert bootstrap_sidecar._coerce_pip_main_return_code(3) == 3
    assert bootstrap_sidecar._coerce_pip_main_return_code(object()) == 1


def test_runtime_sidecar_dll_paths_include_target_library_bin_and_torch_lib(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sidecar = tmp_path / "windows-intel-xpu"
    library_bin = sidecar / "Library" / "bin"
    torch_lib = sidecar / "torch" / "lib"
    rocm_core_bin = sidecar / "_rocm_sdk_core" / "bin"
    rocm_libraries_bin = sidecar / "_rocm_sdk_libraries" / "bin"
    library_bin.mkdir(parents=True)
    torch_lib.mkdir(parents=True)
    rocm_core_bin.mkdir(parents=True)
    rocm_libraries_bin.mkdir(parents=True)
    monkeypatch.setattr(runtime_support.sys, "platform", "win32")

    existing = os.pathsep.join(["existing-a", str(torch_lib), "existing-b"])
    composed = runtime_support.compose_runtime_dll_path(existing=existing, sidecar_path=sidecar)

    parts = composed.split(os.pathsep)
    assert parts[:3] == [str(sidecar.resolve()), str(library_bin.resolve()), str(torch_lib.resolve())]
    assert parts.count(str(torch_lib)) == 1
    assert str(rocm_core_bin.resolve()) in parts
    assert str(rocm_libraries_bin.resolve()) in parts
    assert parts[-2:] == ["existing-a", "existing-b"]


@pytest.mark.parametrize("supports_dll_directory", [False, True])
def test_direct_sidecar_activation_updates_path_for_native_loadlibrary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, supports_dll_directory: bool,
) -> None:
    sidecar = tmp_path / "windows-intel-xpu"
    native = sidecar / "Library" / "bin"
    native.mkdir(parents=True)
    (sidecar / "torch" / "lib").mkdir(parents=True)
    calls = []
    monkeypatch.setattr(runtime_support.sys, "platform", "win32")
    monkeypatch.setattr(runtime_support, "_WINDOWS_DLL_DIRECTORY_HANDLES", {})
    monkeypatch.setenv("PATH", "original-search-path")
    monkeypatch.setattr(
        runtime_support.os, "add_dll_directory",
        (lambda path: calls.append(path) or object()) if supports_dll_directory else None,
        raising=False,
    )

    runtime_support.prepare_runtime_dll_search_path(sidecar)
    runtime_support.prepare_runtime_dll_search_path(sidecar)

    parts = os.environ["PATH"].split(os.pathsep)
    assert parts.count(str(native.resolve())) == 1
    assert parts[-1] == "original-search-path"
    assert calls.count(str(native.resolve())) == int(supports_dll_directory)


def test_select_runtime_target_prefers_nvidia_cuda_on_windows_and_linux() -> None:
    assert bootstrap_module.select_runtime_target(system_name="Windows", machine_name="AMD64", has_nvidia=True) == "windows-nvidia-cuda"
    assert bootstrap_module.select_runtime_target(system_name="Linux", machine_name="x86_64", has_nvidia=True) == "linux-nvidia-cuda"


def test_select_runtime_target_prefers_mps_on_apple_silicon() -> None:
    assert bootstrap_module.select_runtime_target(system_name="Darwin", machine_name="arm64", has_nvidia=False) == "macos-apple-mps"


def test_select_runtime_target_falls_back_to_cpu_when_no_accelerator() -> None:
    assert bootstrap_module.select_runtime_target(system_name="Windows", machine_name="AMD64", has_nvidia=False) == "windows-cpu"
    assert bootstrap_module.select_runtime_target(system_name="Linux", machine_name="x86_64", has_nvidia=False) == "linux-cpu"
    assert bootstrap_module.select_runtime_target(system_name="Darwin", machine_name="x86_64", has_nvidia=False) == "macos-cpu"


def test_select_manifest_asset_returns_target_entry() -> None:
    manifest = {
        "assets": [
            {"id": "windows-cpu", "archive_name": "ShotSieve-windows-cpu-x64.zip"},
            {"id": "linux-nvidia-cuda", "archive_name": "ShotSieve-linux-nvidia-cuda-x64.tar.gz"},
        ]
    }

    asset = bootstrap_module.select_manifest_asset(manifest, "linux-nvidia-cuda")

    assert asset["id"] == "linux-nvidia-cuda"
    assert asset["archive_name"] == "ShotSieve-linux-nvidia-cuda-x64.tar.gz"


def test_select_manifest_asset_rejects_a_legacy_target_manifest() -> None:
    manifest = {
        "assets": [
            {"id": "linux-nvidia", "archive_name": "ShotSieve-linux-nvidia-x64.tar.gz"},
        ]
    }

    with pytest.raises(SystemExit, match="does not include target"):
        bootstrap_module.select_manifest_asset(manifest, "linux-nvidia-cuda")


def test_parse_runtime_asset_accepts_verified_split_parts() -> None:
    entry = {
        "id": "linux-amd-rocm",
        "platform": "linux",
        "runtime": "rocm",
        "archive_name": "ShotSieve-linux-amd-rocm-x64.tar.gz",
        "executable_name": "ShotSieve-AMD-ROCm",
        "variant_folder_name": "ShotSieve-linux-amd-rocm",
        "sha256": "a" * 64,
        "parts": [
            {
                "archive_name": "ShotSieve-linux-amd-rocm-x64.tar.gz.part-000",
                "url": "https://example.invalid/part-000",
                "sha256": "b" * 64,
            }
        ],
    }

    asset = bootstrap_module.parse_runtime_asset(entry)

    assert asset.url is None
    assert len(asset.parts) == 1
    assert asset.parts[0].archive_name.endswith("part-000")


def test_default_manifest_url_targets_latest_release_asset() -> None:
    assert bootstrap_module.DEFAULT_MANIFEST_URL.endswith("/releases/latest/download/bootstrap-manifest.json")


def test_default_runtime_root_uses_internal_folder_for_frozen_launcher(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    appdata_runtime = tmp_path / "appdata" / "ShotSieve" / "runtime"
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "appdata"))

    launcher_dir = tmp_path / "bootstrap-launcher-nvidia" / "ShotSieve"
    launcher_dir.mkdir(parents=True)
    launcher_exe = launcher_dir / "ShotSieve.exe"
    launcher_exe.write_bytes(b"launcher")

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(launcher_exe), raising=False)

    runtime_root = bootstrap_module.default_runtime_root()

    assert runtime_root == (launcher_dir / "runtime").resolve()
    assert runtime_root != appdata_runtime.resolve()


def test_default_runtime_root_prefers_repo_local_data_runtime_for_source_checkout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    appdata_runtime = tmp_path / "appdata" / "ShotSieve" / "runtime"
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "appdata"))

    project_root = tmp_path / "shotsieve-src"
    module_path = project_root / "src" / "shotsieve" / "bootstrap_assets.py"
    module_path.parent.mkdir(parents=True)
    module_path.write_text("# source module\n", encoding="utf-8")
    (project_root / "src" / "shotsieve" / "__init__.py").write_text("", encoding="utf-8")
    (project_root / "pyproject.toml").write_text("[project]\nname='shotsieve'\n", encoding="utf-8")
    monkeypatch.setattr(bootstrap_module, "__file__", str(module_path))

    runtime_root = bootstrap_module.default_runtime_root()

    assert runtime_root == (project_root / "data" / "runtime").resolve()
    assert runtime_root != appdata_runtime.resolve()


def test_resolve_manifest_url_defaults_to_github_even_when_local_manifest_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "bootstrap-manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SHOTSIEVE_BOOTSTRAP_MANIFEST_URL", raising=False)

    resolved = bootstrap_module.resolve_manifest_url(None)

    assert resolved == bootstrap_module.DEFAULT_MANIFEST_URL


def test_resolve_manifest_url_prefers_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHOTSIEVE_BOOTSTRAP_MANIFEST_URL", "https://example.invalid/custom.json")

    resolved = bootstrap_module.resolve_manifest_url(None)

    assert resolved == "https://example.invalid/custom.json"


def test_fetch_manifest_raises_friendly_error_for_non_default_manifest(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request, timeout: int):
        url = request.full_url if hasattr(request, "full_url") else str(request)
        raise _not_found_http_error(url)

    monkeypatch.setattr(bootstrap_module.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(SystemExit, match="Failed to load bootstrap manifest"):
        bootstrap_module.fetch_manifest("https://example.invalid/bootstrap-manifest.json")


def test_fetch_manifest_rejects_unverified_static_fallback_when_default_manifest_unreachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_urlopen(request, timeout: int):
        url = request.full_url if hasattr(request, "full_url") else str(request)
        raise urllib.error.URLError(f"offline: {url}")

    monkeypatch.setattr(bootstrap_module.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(SystemExit, match="checksummed release manifest.*valid release manifest.*manual package"):
        bootstrap_module.fetch_manifest(bootstrap_module.DEFAULT_MANIFEST_URL)


def test_parse_runtime_asset_requires_a_valid_sha256_digest() -> None:
    entry = {
        "id": "windows-cpu",
        "platform": "windows",
        "runtime": "cpu",
        "url": "https://example.invalid/runtime.zip",
        "archive_name": "runtime.zip",
        "executable_name": "ShotSieve-CPU.exe",
        "variant_folder_name": "ShotSieve-windows-cpu",
    }

    with pytest.raises(SystemExit, match="missing keys: sha256"):
        bootstrap_module.parse_runtime_asset(entry)

    entry["sha256"] = "z" * 64
    with pytest.raises(SystemExit, match="valid 64-digit SHA-256"):
        bootstrap_module.parse_runtime_asset(entry)


def test_fetch_manifest_uses_release_api_digest_for_404_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release_tag = "v9.9.9"
    release_payload = {
        "tag_name": release_tag,
        "assets": [
            {
                "name": "ShotSieve-windows-cpu-x64.zip",
                "browser_download_url": "https://github.com/example/ShotSieve/releases/download/v9.9.9/ShotSieve-windows-cpu-x64.zip",
                "digest": "sha256:" + "a" * 64,
            }
        ],
    }

    def fake_open_url(url: str):
        if url == bootstrap_module.DEFAULT_MANIFEST_URL:
            raise _not_found_http_error(url)
        assert url == "https://api.github.com/repos/cfelicio/ShotSieve/releases/latest"
        return io.BytesIO(json.dumps(release_payload).encode("utf-8"))

    monkeypatch.setattr(bootstrap_module, "open_url", fake_open_url)

    manifest = bootstrap_module.fetch_manifest(bootstrap_module.DEFAULT_MANIFEST_URL)

    assert manifest["release_tag"] == release_tag
    asset = bootstrap_module.select_manifest_asset(manifest, "windows-cpu")
    assert asset["sha256"] == "a" * 64


def test_find_local_runtime_archive_in_parent_dist_root_for_frozen_launcher_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_name = "ShotSieve-windows-nvidia-cuda-x64.zip"
    dist_root = tmp_path / "dist"
    dist_root.mkdir(parents=True)
    local_archive = dist_root / archive_name
    local_archive.write_bytes(b"runtime-archive")

    launcher_dir = dist_root / "bootstrap-launcher-nvidia" / "ShotSieve"
    launcher_dir.mkdir(parents=True)
    fake_executable = launcher_dir / "ShotSieve.exe"
    fake_executable.write_bytes(b"fake-launcher")

    unrelated_cwd = tmp_path / "elsewhere"
    unrelated_cwd.mkdir(parents=True)
    monkeypatch.chdir(unrelated_cwd)

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(fake_executable), raising=False)

    found = bootstrap_module.find_local_runtime_archive(archive_name)

    assert found is not None
    assert found.resolve() == local_archive.resolve()


def test_maybe_prepare_torch_runtime_skips_when_bundle_already_contains_torch(tmp_path: Path) -> None:
    install_dir = tmp_path / "install"
    bundled_torch = install_dir / "_internal" / "torch"
    bundled_torch.mkdir(parents=True)
    (bundled_torch / "__init__.py").write_text("", encoding="utf-8")

    asset = bootstrap_module.RuntimeAsset(
        id="windows-nvidia-cuda",
        platform="windows",
        runtime="cuda",
        url="https://example.invalid/runtime.zip",
        archive_name="runtime.zip",
        executable_name="ShotSieve-NVIDIA-CUDA.exe",
        variant_folder_name="ShotSieve-windows-nvidia-cuda",
        sha256=None,
    )

    def fail_input(_: str) -> str:
        raise AssertionError("input() should not be called when torch is bundled")

    env_updates = bootstrap_sidecar.maybe_prepare_torch_runtime(
        asset,
        install_dir=install_dir,
        runtime_root=tmp_path / "runtime",
        input_func=fail_input,
    )

    assert env_updates == {}


def test_maybe_prepare_torch_runtime_uses_existing_sidecar_site_packages(tmp_path: Path) -> None:
    runtime_root = tmp_path / "runtime"
    install_dir = tmp_path / "install"
    install_dir.mkdir(parents=True)
    site_packages = bootstrap_sidecar.sidecar_site_packages_dir(runtime_root, "windows-nvidia-cuda")
    (site_packages / "torch").mkdir(parents=True)
    (site_packages / "torch" / "__init__.py").write_text("", encoding="utf-8")
    (site_packages / ".shotsieve-runtime.json").write_text(
        json.dumps(
            {
                "schema": bootstrap_sidecar.SIDECAR_STATE_VERSION,
                "kind": "runtime",
                "complete": True,
                "torch_complete": True,
                "plan": bootstrap_sidecar.torch_install_plan(
                    target_id="windows-nvidia-cuda", runtime="cuda"
                ).to_json(),
            }
        ),
        encoding="utf-8",
    )

    asset = bootstrap_module.RuntimeAsset(
        id="windows-nvidia-cuda",
        platform="windows",
        runtime="cuda",
        url="https://example.invalid/runtime.zip",
        archive_name="runtime.zip",
        executable_name="ShotSieve-NVIDIA-CUDA.exe",
        variant_folder_name="ShotSieve-windows-nvidia-cuda",
        sha256=None,
    )

    def fail_input(_: str) -> str:
        raise AssertionError("input() should not be called when sidecar torch exists")

    env_updates = bootstrap_sidecar.maybe_prepare_torch_runtime(
        asset,
        install_dir=install_dir,
        runtime_root=runtime_root,
        input_func=fail_input,
    )

    assert "PYTHONPATH" in env_updates
    assert str(site_packages) in env_updates["PYTHONPATH"]


def test_maybe_prepare_torch_runtime_auto_installs_when_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_root = tmp_path / "runtime"
    install_dir = tmp_path / "install"
    install_dir.mkdir(parents=True)
    monkeypatch.setenv("SHOTSIEVE_BOOTSTRAP_AUTO_INSTALL_TORCH", "1")

    calls: list[tuple[str, Path]] = []

    def fake_install_torch_sidecar(*, runtime: str, site_packages: Path) -> bool:
        calls.append((runtime, site_packages))
        (site_packages / "torch").mkdir(parents=True)
        (site_packages / "torch" / "__init__.py").write_text("", encoding="utf-8")
        return True

    monkeypatch.setattr(bootstrap_sidecar, "install_torch_sidecar", fake_install_torch_sidecar)

    asset = bootstrap_module.RuntimeAsset(
        id="windows-nvidia-cuda",
        platform="windows",
        runtime="cuda",
        url="https://example.invalid/runtime.zip",
        archive_name="runtime.zip",
        executable_name="ShotSieve-NVIDIA-CUDA.exe",
        variant_folder_name="ShotSieve-windows-nvidia-cuda",
        sha256=None,
    )

    env_updates = bootstrap_sidecar.maybe_prepare_torch_runtime(
        asset,
        install_dir=install_dir,
        runtime_root=runtime_root,
    )

    assert len(calls) == 1
    assert calls[0][0] == "cuda"
    assert "PYTHONPATH" in env_updates


def test_load_embedded_pip_main_suppresses_distutils_replacement_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_pip_main_module = _new_module("pip._internal.cli.main")

    def fake_main(args):
        return 0

    fake_pip_main_module.main = fake_main

    def fake_import_module(name: str):
        if name != "pip._internal.cli.main":
            raise ImportError(name)
        warnings.warn(
            "Setuptools is replacing distutils. Support for replacing an already imported distutils is deprecated.",
            UserWarning,
            stacklevel=1,
        )
        return fake_pip_main_module

    monkeypatch.setattr(bootstrap_sidecar.importlib, "import_module", fake_import_module)

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        pip_main = bootstrap_sidecar._load_embedded_pip_main()

    assert pip_main is fake_main
    assert recorded == []
