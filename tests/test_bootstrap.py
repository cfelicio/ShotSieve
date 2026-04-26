from __future__ import annotations

from email.message import Message
import io
import sys
import types
from pathlib import Path
import tarfile
import urllib.error
from typing import Any
import warnings
import zipfile

import pytest

from shotsieve import bootstrap as bootstrap_module
from shotsieve import runtime_support


def _new_module(name: str) -> Any:
    return types.ModuleType(name)


def _not_found_http_error(url: str) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(url=url, code=404, msg="Not Found", hdrs=Message(), fp=None)


def test_coerce_pip_main_return_code_handles_none_int_and_unexpected_values() -> None:
    assert bootstrap_module._coerce_pip_main_return_code(None) == 0
    assert bootstrap_module._coerce_pip_main_return_code(3) == 3
    assert bootstrap_module._coerce_pip_main_return_code(object()) == 1


def test_select_runtime_target_prefers_nvidia_on_windows_and_linux() -> None:
    assert bootstrap_module.select_runtime_target(system_name="Windows", machine_name="AMD64", has_nvidia=True) == "windows-nvidia"
    assert bootstrap_module.select_runtime_target(system_name="Linux", machine_name="x86_64", has_nvidia=True) == "linux-nvidia"


def test_select_runtime_target_prefers_mps_on_apple_silicon() -> None:
    assert bootstrap_module.select_runtime_target(system_name="Darwin", machine_name="arm64", has_nvidia=False) == "macos-mps"


def test_select_runtime_target_falls_back_to_cpu_when_no_accelerator() -> None:
    assert bootstrap_module.select_runtime_target(system_name="Windows", machine_name="AMD64", has_nvidia=False) == "windows-cpu"
    assert bootstrap_module.select_runtime_target(system_name="Linux", machine_name="x86_64", has_nvidia=False) == "linux-cpu"
    assert bootstrap_module.select_runtime_target(system_name="Darwin", machine_name="x86_64", has_nvidia=False) == "macos-cpu"


def test_select_manifest_asset_returns_target_entry() -> None:
    manifest = {
        "assets": [
            {"id": "windows-cpu", "archive_name": "ShotSieve-windows-cpu-x64.zip"},
            {"id": "linux-nvidia", "archive_name": "ShotSieve-linux-nvidia-x64.tar.gz"},
        ]
    }

    asset = bootstrap_module.select_manifest_asset(manifest, "linux-nvidia")

    assert asset["id"] == "linux-nvidia"
    assert asset["archive_name"] == "ShotSieve-linux-nvidia-x64.tar.gz"


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
    module_path = project_root / "src" / "shotsieve" / "bootstrap.py"
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


def test_fetch_manifest_uses_static_default_fallback_when_default_manifest_unreachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_urlopen(request, timeout: int):
        url = request.full_url if hasattr(request, "full_url") else str(request)
        raise urllib.error.URLError(f"offline: {url}")

    monkeypatch.setattr(bootstrap_module.urllib.request, "urlopen", fake_urlopen)

    manifest = bootstrap_module.fetch_manifest(bootstrap_module.DEFAULT_MANIFEST_URL)

    assert manifest["release_tag"] == "latest"
    assert manifest["repo"] == "cfelicio/ShotSieve"
    assert any(asset["id"] == "windows-cpu" for asset in manifest["assets"])


def test_ensure_runtime_asset_falls_back_to_local_archive_when_download_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build_root = tmp_path / "local-build"
    build_root.mkdir(parents=True)
    archive_name = "ShotSieve-windows-nvidia-x64.zip"
    executable_name = "ShotSieve-NVIDIA.exe"
    variant_folder = "ShotSieve-windows-nvidia"

    local_archive = build_root / archive_name
    with zipfile.ZipFile(local_archive, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{variant_folder}/{executable_name}", "fake-binary")

    monkeypatch.chdir(build_root)

    attempted_urls: list[str] = []

    def fake_open_url(url: str):
        attempted_urls.append(url)
        raise _not_found_http_error(url)

    monkeypatch.setattr(bootstrap_module, "open_url", fake_open_url)

    asset = bootstrap_module.RuntimeAsset(
        id="windows-nvidia",
        platform="windows",
        runtime="cuda",
        url="https://github.com/cfelicio/ShotSieve/releases/latest/download/ShotSieve-windows-nvidia-x64.zip",
        archive_name=archive_name,
        executable_name=executable_name,
        variant_folder_name=variant_folder,
        sha256=None,
    )

    executable = bootstrap_module.ensure_runtime_asset(asset, runtime_root=tmp_path / "runtime")

    assert attempted_urls == [asset.url]
    assert executable.exists()
    assert executable.name == executable_name
    assert not (tmp_path / "runtime" / "downloads" / archive_name).exists()


def test_find_local_runtime_archive_in_parent_dist_root_for_frozen_launcher_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_name = "ShotSieve-windows-nvidia-x64.zip"
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


def test_ensure_runtime_asset_raises_when_download_fails_and_no_local_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    asset = bootstrap_module.RuntimeAsset(
        id="windows-cpu",
        platform="windows",
        runtime="cpu",
        url="https://github.com/cfelicio/ShotSieve/releases/download/v0.1.0/ShotSieve-windows-cpu-x64.zip",
        archive_name="ShotSieve-windows-cpu-x64.zip",
        executable_name="ShotSieve-CPU.exe",
        variant_folder_name="ShotSieve-windows-cpu",
        sha256=None,
    )

    def fake_open_url(url: str):
        raise _not_found_http_error(url)

    monkeypatch.setattr(bootstrap_module, "open_url", fake_open_url)

    with pytest.raises(SystemExit, match="Failed to download runtime archive"):
        bootstrap_module.ensure_runtime_asset(asset, runtime_root=tmp_path)


def test_ensure_runtime_asset_reuses_existing_install_without_sha_and_without_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_root = tmp_path / "runtime"
    install_dir = runtime_root / "installs" / "windows-nvidia"
    executable = install_dir / "ShotSieve-windows-nvidia" / "ShotSieve-NVIDIA.exe"
    executable.parent.mkdir(parents=True, exist_ok=True)
    executable.write_bytes(b"fake-runtime")
    (install_dir / ".asset-sha256").write_text("", encoding="utf-8")

    called_urls: list[str] = []

    def fake_open_url(url: str):
        called_urls.append(url)
        raise _not_found_http_error(url)

    monkeypatch.setattr(bootstrap_module, "open_url", fake_open_url)

    asset = bootstrap_module.RuntimeAsset(
        id="windows-nvidia",
        platform="windows",
        runtime="cuda",
        url="https://example.invalid/ShotSieve-windows-nvidia-x64.zip",
        archive_name="ShotSieve-windows-nvidia-x64.zip",
        executable_name="ShotSieve-NVIDIA.exe",
        variant_folder_name="ShotSieve-windows-nvidia",
        sha256=None,
    )

    resolved = bootstrap_module.ensure_runtime_asset(asset, runtime_root=runtime_root)

    assert resolved.resolve() == executable.resolve()
    assert called_urls == []


def test_ensure_runtime_asset_prefers_colocated_frozen_runtime_executable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher_dir = tmp_path / "bootstrap-launcher-nvidia" / "ShotSieve"
    launcher_dir.mkdir(parents=True)
    launcher_exe = launcher_dir / "ShotSieve.exe"
    launcher_exe.write_bytes(b"bootstrap")

    runtime_exe = launcher_dir / "ShotSieve-NVIDIA.exe"
    runtime_exe.write_bytes(b"runtime")

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(launcher_exe), raising=False)

    called_urls: list[str] = []

    def fake_open_url(url: str):
        called_urls.append(url)
        raise _not_found_http_error(url)

    monkeypatch.setattr(bootstrap_module, "open_url", fake_open_url)

    asset = bootstrap_module.RuntimeAsset(
        id="windows-nvidia",
        platform="windows",
        runtime="cuda",
        url="https://example.invalid/ShotSieve-windows-nvidia-x64.zip",
        archive_name="ShotSieve-windows-nvidia-x64.zip",
        executable_name="ShotSieve-NVIDIA.exe",
        variant_folder_name="ShotSieve-windows-nvidia",
        sha256=None,
    )

    resolved = bootstrap_module.ensure_runtime_asset(asset, runtime_root=tmp_path / "runtime")

    assert resolved.resolve() == runtime_exe.resolve()
    assert called_urls == []


def test_extract_archive_rejects_tar_symlink_members(tmp_path: Path) -> None:
    archive_path = tmp_path / "malicious.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        link_info = tarfile.TarInfo("runtime/link")
        link_info.type = tarfile.SYMTYPE
        link_info.linkname = "../escape"
        archive.addfile(link_info)

    with pytest.raises(SystemExit, match="Unsupported archive member type"):
        bootstrap_module.extract_archive(archive_path, tmp_path / "out")


def test_extract_archive_accepts_regular_tar_files(tmp_path: Path) -> None:
    archive_path = tmp_path / "runtime.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        data = b"runtime-binary"
        file_info = tarfile.TarInfo("runtime/bin/shotsieve")
        file_info.size = len(data)
        archive.addfile(file_info, io.BytesIO(data))

    destination = tmp_path / "out"
    bootstrap_module.extract_archive(archive_path, destination)

    extracted = destination / "runtime" / "bin" / "shotsieve"
    assert extracted.exists()
    assert extracted.read_bytes() == b"runtime-binary"


def test_maybe_prepare_torch_runtime_skips_when_bundle_already_contains_torch(tmp_path: Path) -> None:
    install_dir = tmp_path / "install"
    bundled_torch = install_dir / "_internal" / "torch"
    bundled_torch.mkdir(parents=True)
    (bundled_torch / "__init__.py").write_text("", encoding="utf-8")

    asset = bootstrap_module.RuntimeAsset(
        id="windows-nvidia",
        platform="windows",
        runtime="cuda",
        url="https://example.invalid/runtime.zip",
        archive_name="runtime.zip",
        executable_name="ShotSieve-NVIDIA.exe",
        variant_folder_name="ShotSieve-windows-nvidia",
        sha256=None,
    )

    def fail_input(_: str) -> str:
        raise AssertionError("input() should not be called when torch is bundled")

    env_updates = bootstrap_module.maybe_prepare_torch_runtime(
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
    site_packages = bootstrap_module.sidecar_site_packages_dir(runtime_root, "windows-nvidia")
    (site_packages / "torch").mkdir(parents=True)
    (site_packages / "torch" / "__init__.py").write_text("", encoding="utf-8")

    asset = bootstrap_module.RuntimeAsset(
        id="windows-nvidia",
        platform="windows",
        runtime="cuda",
        url="https://example.invalid/runtime.zip",
        archive_name="runtime.zip",
        executable_name="ShotSieve-NVIDIA.exe",
        variant_folder_name="ShotSieve-windows-nvidia",
        sha256=None,
    )

    def fail_input(_: str) -> str:
        raise AssertionError("input() should not be called when sidecar torch exists")

    env_updates = bootstrap_module.maybe_prepare_torch_runtime(
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

    monkeypatch.setattr(bootstrap_module, "install_torch_sidecar", fake_install_torch_sidecar)

    asset = bootstrap_module.RuntimeAsset(
        id="windows-nvidia",
        platform="windows",
        runtime="cuda",
        url="https://example.invalid/runtime.zip",
        archive_name="runtime.zip",
        executable_name="ShotSieve-NVIDIA.exe",
        variant_folder_name="ShotSieve-windows-nvidia",
        sha256=None,
    )

    env_updates = bootstrap_module.maybe_prepare_torch_runtime(
        asset,
        install_dir=install_dir,
        runtime_root=runtime_root,
    )

    assert len(calls) == 1
    assert calls[0][0] == "cuda"
    assert "PYTHONPATH" in env_updates


def test_bootstrap_runtime_support_wrappers_delegate_to_shared_module(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(runtime_support, "path_has_torch", lambda path: calls.append(("torch", path)) or True)
    monkeypatch.setattr(runtime_support, "path_has_pyiqa", lambda path: calls.append(("pyiqa", path)) or False)
    monkeypatch.setattr(runtime_support, "parse_env_bool", lambda value: calls.append(("bool", value)) or True)
    monkeypatch.setattr(runtime_support, "is_interactive_console", lambda: calls.append(("interactive", None)) or False)

    def fake_confirm(prompt: str, *, input_func=input) -> bool:
        calls.append(("confirm", prompt))
        return True

    monkeypatch.setattr(runtime_support, "confirm", fake_confirm)
    monkeypatch.setattr(
        runtime_support,
        "compose_pythonpath",
        lambda *, existing, prepend_path: calls.append(("pythonpath", (existing, prepend_path))) or "shared-pythonpath",
    )

    assert bootstrap_module._path_has_torch(tmp_path) is True
    assert bootstrap_module._path_has_pyiqa(tmp_path) is False
    assert bootstrap_module._parse_env_bool("yes") is True
    assert bootstrap_module._is_interactive_console() is False
    assert bootstrap_module._confirm("Install now?") is True
    assert bootstrap_module._compose_pythonpath(existing="existing", prepend_path=tmp_path) == "shared-pythonpath"

    assert calls == [
        ("torch", tmp_path),
        ("pyiqa", tmp_path),
        ("bool", "yes"),
        ("interactive", None),
        ("confirm", "Install now?"),
        ("pythonpath", ("existing", tmp_path)),
    ]


def test_install_torch_sidecar_uses_embedded_installer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    site_packages = tmp_path / "site-packages"
    embedded_calls: list[tuple[str, Path, bool]] = []

    def fake_embedded_install(*, runtime: str, site_packages: Path, force_reinstall: bool = False, output_func=print):
        embedded_calls.append((runtime, site_packages, force_reinstall))
        return True

    monkeypatch.setattr(bootstrap_module, "_install_torch_sidecar_with_embedded_pip", fake_embedded_install)

    installed = bootstrap_module.install_torch_sidecar(runtime="cuda", site_packages=site_packages)

    assert installed is True
    assert embedded_calls == [("cuda", site_packages, False)]


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

    monkeypatch.setattr(bootstrap_module.importlib, "import_module", fake_import_module)

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        pip_main = bootstrap_module._load_embedded_pip_main()

    assert pip_main is fake_main
    assert recorded == []


def test_install_learned_iqa_sidecar_uses_embedded_installer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    site_packages = tmp_path / "site-packages"
    embedded_calls: list[tuple[str, Path, bool]] = []

    def fake_embedded_install(*, runtime: str, site_packages: Path, force_reinstall: bool = False, output_func=print):
        embedded_calls.append((runtime, site_packages, force_reinstall))
        return True

    monkeypatch.setattr(bootstrap_module, "_install_learned_iqa_sidecar_with_embedded_pip", fake_embedded_install, raising=False)

    installed = bootstrap_module.install_learned_iqa_sidecar(runtime="cuda", site_packages=site_packages)

    assert installed is True
    assert embedded_calls == [("cuda", site_packages, False)]


def test_embedded_install_learned_iqa_sidecar_installs_expected_packages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    site_packages = tmp_path / "site-packages"
    captured_args: list[list[str]] = []

    fake_pip_main_module = _new_module("pip._internal.cli.main")

    def fake_main(args):
        captured_args.append(list(args))
        return 0

    fake_pip_main_module.main = fake_main
    monkeypatch.setitem(sys.modules, "pip._internal.cli.main", fake_pip_main_module)
    monkeypatch.setattr(bootstrap_module, "_path_has_pyiqa", lambda path: True, raising=False)

    installed = bootstrap_module._install_learned_iqa_sidecar_with_embedded_pip(
        runtime="cuda",
        site_packages=site_packages,
    )

    assert installed is True
    assert captured_args
    flattened = " ".join(" ".join(args) for args in captured_args)
    assert "pyiqa" in flattened
    assert "icecream" in flattened
    assert "--index-url" not in flattened
    assert "download.pytorch.org" not in flattened


def test_embedded_install_learned_iqa_sidecar_installs_pyiqa_without_deps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    site_packages = tmp_path / "site-packages"
    captured_args: list[list[str]] = []

    fake_pip_main_module = _new_module("pip._internal.cli.main")

    def fake_main(args):
        captured_args.append(list(args))
        return 0

    fake_pip_main_module.main = fake_main
    monkeypatch.setitem(sys.modules, "pip._internal.cli.main", fake_pip_main_module)
    monkeypatch.setattr(bootstrap_module, "_path_has_pyiqa", lambda path: True, raising=False)

    installed = bootstrap_module._install_learned_iqa_sidecar_with_embedded_pip(
        runtime="cuda",
        site_packages=site_packages,
    )

    assert installed is True
    pyiqa_args = next(args for args in captured_args if "pyiqa" in args)
    assert "--no-deps" in pyiqa_args


def test_embedded_install_learned_iqa_sidecar_installs_opencv_headless(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    site_packages = tmp_path / "site-packages"
    captured_args: list[list[str]] = []

    fake_pip_main_module = _new_module("pip._internal.cli.main")

    def fake_main(args):
        captured_args.append(list(args))
        return 0

    fake_pip_main_module.main = fake_main
    monkeypatch.setitem(sys.modules, "pip._internal.cli.main", fake_pip_main_module)
    monkeypatch.setattr(bootstrap_module, "_path_has_pyiqa", lambda path: True, raising=False)

    installed = bootstrap_module._install_learned_iqa_sidecar_with_embedded_pip(
        runtime="cuda",
        site_packages=site_packages,
    )

    assert installed is True
    flattened = " ".join(" ".join(args) for args in captured_args)
    assert "opencv-python-headless" in flattened


def test_embedded_install_learned_iqa_sidecar_installs_pyyaml(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    site_packages = tmp_path / "site-packages"
    captured_args: list[list[str]] = []

    fake_pip_main_module = _new_module("pip._internal.cli.main")

    def fake_main(args):
        captured_args.append(list(args))
        return 0

    fake_pip_main_module.main = fake_main
    monkeypatch.setitem(sys.modules, "pip._internal.cli.main", fake_pip_main_module)
    monkeypatch.setattr(bootstrap_module, "_path_has_pyiqa", lambda path: True, raising=False)

    installed = bootstrap_module._install_learned_iqa_sidecar_with_embedded_pip(
        runtime="cuda",
        site_packages=site_packages,
    )

    assert installed is True
    flattened = " ".join(" ".join(args) for args in captured_args)
    assert "pyyaml" in flattened


def test_embedded_install_learned_iqa_sidecar_installs_sympy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    site_packages = tmp_path / "site-packages"
    captured_args: list[list[str]] = []

    fake_pip_main_module = _new_module("pip._internal.cli.main")

    def fake_main(args):
        captured_args.append(list(args))
        return 0

    fake_pip_main_module.main = fake_main
    monkeypatch.setitem(sys.modules, "pip._internal.cli.main", fake_pip_main_module)
    monkeypatch.setattr(bootstrap_module, "_path_has_pyiqa", lambda path: True, raising=False)

    installed = bootstrap_module._install_learned_iqa_sidecar_with_embedded_pip(
        runtime="cuda",
        site_packages=site_packages,
    )

    assert installed is True
    flattened = " ".join(" ".join(args) for args in captured_args)
    assert "sympy" in flattened


def test_patch_distlib_finder_for_frozen_registers_loader_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_resources_module = _new_module("pip._vendor.distlib.resources")
    fake_resources_module._finder_registry = {}

    class FakeResourceFinder:
        pass

    fake_resources_module.ResourceFinder = FakeResourceFinder
    register_calls: list[tuple[type, object]] = []

    def fake_register_finder(loader_type: type, finder: object) -> None:
        register_calls.append((loader_type, finder))

    fake_resources_module.register_finder = fake_register_finder

    class FakeLoader:
        pass

    fake_distlib_module = _new_module("pip._vendor.distlib")
    fake_distlib_module.__loader__ = FakeLoader()

    def fake_import_module(name: str):
        if name == "pip._vendor.distlib.resources":
            return fake_resources_module
        if name == "pip._vendor.distlib":
            return fake_distlib_module
        raise ImportError(name)

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(bootstrap_module.importlib, "import_module", fake_import_module)

    bootstrap_module._patch_distlib_finder_for_frozen()

    assert register_calls
    registered_types = {loader_type for loader_type, _ in register_calls}
    assert FakeLoader in registered_types
    assert all(finder is FakeResourceFinder for _loader_type, finder in register_calls)


def test_patch_distlib_finder_for_frozen_suppresses_distutils_warning_during_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_resources_module = _new_module("pip._vendor.distlib.resources")
    fake_resources_module._finder_registry = {}

    class FakeResourceFinder:
        pass

    fake_resources_module.ResourceFinder = FakeResourceFinder
    fake_resources_module.register_finder = lambda *_args, **_kwargs: None

    class FakeLoader:
        pass

    fake_distlib_module = _new_module("pip._vendor.distlib")
    fake_distlib_module.__loader__ = FakeLoader()

    def fake_import_module(name: str):
        if name in {"pip._vendor.distlib.resources", "pip._vendor.distlib"}:
            warnings.warn(
                "Setuptools is replacing distutils. Support for replacing an already imported distutils is deprecated.",
                UserWarning,
                stacklevel=1,
            )
            return fake_resources_module if name.endswith("resources") else fake_distlib_module
        raise ImportError(name)

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(bootstrap_module.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(bootstrap_module.pkgutil, "get_loader", lambda name: None, raising=False)

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        bootstrap_module._patch_distlib_finder_for_frozen()

    assert recorded == []


def test_patch_distlib_finder_for_frozen_registers_pkgutil_loader_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_resources_module = _new_module("pip._vendor.distlib.resources")
    fake_resources_module._finder_registry = {}

    class FakeResourceFinder:
        pass

    fake_resources_module.ResourceFinder = FakeResourceFinder
    register_calls: list[tuple[type, object]] = []

    def fake_register_finder(loader_type: type, finder: object) -> None:
        register_calls.append((loader_type, finder))

    fake_resources_module.register_finder = fake_register_finder

    class DistlibLoader:
        pass

    class PkgutilLoader:
        pass

    fake_distlib_module = _new_module("pip._vendor.distlib")
    fake_distlib_module.__loader__ = DistlibLoader()

    def fake_import_module(name: str):
        if name == "pip._vendor.distlib.resources":
            return fake_resources_module
        if name == "pip._vendor.distlib":
            return fake_distlib_module
        raise ImportError(name)

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(bootstrap_module.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(bootstrap_module.pkgutil, "get_loader", lambda name: PkgutilLoader(), raising=False)

    bootstrap_module._patch_distlib_finder_for_frozen()

    registered_types = {loader_type for loader_type, _ in register_calls}
    assert DistlibLoader in registered_types
    assert PkgutilLoader in registered_types


def test_patch_distlib_finder_for_frozen_wraps_finder_with_resource_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_resources_module = _new_module("pip._vendor.distlib.resources")
    fake_resources_module._finder_registry = {}

    class FakeDistlibException(Exception):
        pass

    class FakeResourceFinder:
        def __init__(self, package: str):
            self.package = package

    def fake_finder(_package: str):
        raise FakeDistlibException("missing finder")

    fake_resources_module.DistlibException = FakeDistlibException
    fake_resources_module.ResourceFinder = FakeResourceFinder
    fake_resources_module.finder = fake_finder
    fake_resources_module.register_finder = lambda *_args, **_kwargs: None

    class DistlibLoader:
        pass

    fake_distlib_module = _new_module("pip._vendor.distlib")
    fake_distlib_module.__loader__ = DistlibLoader()

    def fake_import_module(name: str):
        if name == "pip._vendor.distlib.resources":
            return fake_resources_module
        if name == "pip._vendor.distlib":
            return fake_distlib_module
        raise ImportError(name)

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(bootstrap_module.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(bootstrap_module.pkgutil, "get_loader", lambda name: DistlibLoader(), raising=False)

    bootstrap_module._patch_distlib_finder_for_frozen()

    resolved = fake_resources_module.finder("pip._vendor.distlib")
    assert isinstance(resolved, FakeResourceFinder)
    assert resolved.package == "pip._vendor.distlib"


def test_embedded_install_torch_sidecar_calls_distlib_patch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    site_packages = tmp_path / "site-packages"
    patch_calls: list[bool] = []

    monkeypatch.setattr(bootstrap_module, "_patch_distlib_finder_for_frozen", lambda: patch_calls.append(True))

    fake_pip_main_module = _new_module("pip._internal.cli.main")

    def fake_main(args):
        return 0

    fake_pip_main_module.main = fake_main
    monkeypatch.setitem(sys.modules, "pip._internal.cli.main", fake_pip_main_module)
    monkeypatch.setattr(bootstrap_module, "_path_has_torch", lambda path: True)

    installed = bootstrap_module._install_torch_sidecar_with_embedded_pip(
        runtime="cuda",
        site_packages=site_packages,
    )

    assert installed is True
    assert patch_calls == [True]


def test_patch_pip_scriptmaker_for_embedded_install_disables_launchers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_wheel_module = _new_module("pip._internal.operations.install.wheel")
    original_import_module = bootstrap_module.importlib.import_module

    class FakePipScriptMaker:
        def __init__(self, *args, **kwargs):
            self.add_launchers = True

    fake_wheel_module.PipScriptMaker = FakePipScriptMaker

    def fake_import_module(name: str):
        if name == "pip._internal.operations.install.wheel":
            return fake_wheel_module
        return original_import_module(name)

    monkeypatch.setattr(bootstrap_module.importlib, "import_module", fake_import_module)

    bootstrap_module._patch_pip_scriptmaker_for_embedded_install()

    maker = FakePipScriptMaker()
    assert maker.add_launchers is False


def test_patch_pip_scriptmaker_for_embedded_install_suppresses_distutils_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_wheel_module = _new_module("pip._internal.operations.install.wheel")

    class FakePipScriptMaker:
        def __init__(self, *args, **kwargs):
            self.add_launchers = True

    fake_wheel_module.PipScriptMaker = FakePipScriptMaker

    def fake_import_module(name: str):
        if name != "pip._internal.operations.install.wheel":
            raise ImportError(name)
        warnings.warn(
            "Setuptools is replacing distutils. Support for replacing an already imported distutils is deprecated.",
            UserWarning,
            stacklevel=1,
        )
        return fake_wheel_module

    monkeypatch.setattr(bootstrap_module.importlib, "import_module", fake_import_module)

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        bootstrap_module._patch_pip_scriptmaker_for_embedded_install()

    assert recorded == []


def test_embedded_install_torch_sidecar_calls_scriptmaker_patch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    site_packages = tmp_path / "site-packages"
    patch_calls: list[bool] = []

    monkeypatch.setattr(
        bootstrap_module,
        "_patch_pip_scriptmaker_for_embedded_install",
        lambda: patch_calls.append(True),
    )

    fake_pip_main_module = _new_module("pip._internal.cli.main")

    def fake_main(args):
        return 0

    fake_pip_main_module.main = fake_main
    monkeypatch.setitem(sys.modules, "pip._internal.cli.main", fake_pip_main_module)
    monkeypatch.setattr(bootstrap_module, "_path_has_torch", lambda path: True)

    installed = bootstrap_module._install_torch_sidecar_with_embedded_pip(
        runtime="cuda",
        site_packages=site_packages,
    )

    assert installed is True
    assert patch_calls == [True]


def test_embedded_install_torch_sidecar_uses_no_deps_and_supports_force_reinstall(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    site_packages = tmp_path / "site-packages"
    captured_args: list[list[str]] = []

    fake_pip_main_module = _new_module("pip._internal.cli.main")

    def fake_main(args):
        captured_args.append(list(args))
        return 0

    fake_pip_main_module.main = fake_main
    monkeypatch.setitem(sys.modules, "pip._internal.cli.main", fake_pip_main_module)
    monkeypatch.setattr(bootstrap_module, "_path_has_torch", lambda path: True)

    installed = bootstrap_module._install_torch_sidecar_with_embedded_pip(
        runtime="cuda",
        site_packages=site_packages,
        force_reinstall=True,
    )

    assert installed is True
    assert len(captured_args) == 2
    assert "--no-deps" in captured_args[0]
    assert "--force-reinstall" in captured_args[0]
    assert "--disable-pip-version-check" in captured_args[0]
    assert "--trusted-host" in captured_args[0]
    assert "download.pytorch.org" in captured_args[0]
    assert "--log" in captured_args[0]
    assert "torch" in captured_args[0]
    assert "torchvision" in captured_args[1]


def test_embedded_install_torch_sidecar_continues_when_torchvision_fails_but_torch_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    site_packages = tmp_path / "site-packages"
    captured_args: list[list[str]] = []

    fake_pip_main_module = _new_module("pip._internal.cli.main")

    def fake_main(args):
        captured_args.append(list(args))
        package_name = "torchvision" if "torchvision" in args else "torch"
        return 0 if package_name == "torch" else 1

    fake_pip_main_module.main = fake_main
    monkeypatch.setitem(sys.modules, "pip._internal.cli.main", fake_pip_main_module)
    monkeypatch.setattr(bootstrap_module, "_path_has_torch", lambda path: True)

    messages: list[str] = []
    installed = bootstrap_module._install_torch_sidecar_with_embedded_pip(
        runtime="cuda",
        site_packages=site_packages,
        output_func=messages.append,
    )

    assert installed is True
    assert len(captured_args) == 2
    assert "torch" in captured_args[0]
    assert "torchvision" in captured_args[1]
    assert any("torchvision installation failed" in message.casefold() for message in messages)


def test_install_torch_sidecar_reports_missing_runtime_installer_when_embedded_pip_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    site_packages = tmp_path / "site-packages"
    monkeypatch.setattr(bootstrap_module, "_install_torch_sidecar_with_embedded_pip", lambda **kwargs: None)

    messages: list[str] = []
    installed = bootstrap_module.install_torch_sidecar(runtime="cuda", site_packages=site_packages, output_func=messages.append)

    assert installed is False
    assert any("Bundled pip runtime installer is unavailable" in message for message in messages)


def test_install_torch_sidecar_suppresses_distutils_warning_from_bootstrap_prep(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    site_packages = tmp_path / "site-packages"

    def emit_distutils_warning() -> None:
        warnings.warn(
            "Setuptools is replacing distutils. Support for replacing an already imported distutils is deprecated.",
            UserWarning,
            stacklevel=1,
        )

    monkeypatch.setattr(bootstrap_module, "_patch_distlib_finder_for_frozen", emit_distutils_warning)
    monkeypatch.setattr(bootstrap_module, "_patch_pip_scriptmaker_for_embedded_install", lambda: None)
    monkeypatch.setattr(bootstrap_module, "_load_embedded_pip_main", lambda: None)

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        installed = bootstrap_module.install_torch_sidecar(runtime="cuda", site_packages=site_packages)

    assert installed is False
    assert recorded == []


def test_embedded_install_torch_sidecar_reports_pip_log_path_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    site_packages = tmp_path / "site-packages"
    pip_log = site_packages / "pip-install.log"

    fake_pip_main_module = _new_module("pip._internal.cli.main")

    def fake_main(args):
        return 1

    fake_pip_main_module.main = fake_main
    monkeypatch.setitem(sys.modules, "pip._internal.cli.main", fake_pip_main_module)
    monkeypatch.setattr(bootstrap_module, "_path_has_torch", lambda path: False)

    messages: list[str] = []
    installed = bootstrap_module._install_torch_sidecar_with_embedded_pip(
        runtime="cuda",
        site_packages=site_packages,
        output_func=messages.append,
    )

    assert installed is False
    assert any("pip-install.log" in message for message in messages)
    assert any("exit code" in message.casefold() for message in messages)
    assert pip_log.exists()
    assert "torch" in pip_log.read_text(encoding="utf-8")


def test_embedded_install_torch_sidecar_suppresses_distutils_warning_during_pip_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    site_packages = tmp_path / "site-packages"

    fake_pip_main_module = _new_module("pip._internal.cli.main")

    def fake_main(args):
        warnings.warn(
            "Setuptools is replacing distutils. Support for replacing an already imported distutils is deprecated.",
            UserWarning,
            stacklevel=1,
        )
        return 0

    fake_pip_main_module.main = fake_main
    monkeypatch.setitem(sys.modules, "pip._internal.cli.main", fake_pip_main_module)
    monkeypatch.setattr(bootstrap_module, "_path_has_torch", lambda path: True)

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        installed = bootstrap_module._install_torch_sidecar_with_embedded_pip(
            runtime="cuda",
            site_packages=site_packages,
        )

    assert installed is True
    assert recorded == []


def test_embedded_install_torch_sidecar_suppresses_pip_unexpected_import_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    site_packages = tmp_path / "site-packages"

    fake_pip_main_module = _new_module("pip._internal.cli.main")

    class FakePipDeprecationWarning(Warning):
        pass

    def fake_main(args):
        warnings.warn(
            "DEPRECATION: Unexpected import of 'torch' after pip install started. pip 26.3 will enforce this behaviour change.",
            FakePipDeprecationWarning,
            stacklevel=1,
        )
        return 0

    fake_pip_main_module.main = fake_main
    monkeypatch.setitem(sys.modules, "pip._internal.cli.main", fake_pip_main_module)
    monkeypatch.setattr(bootstrap_module, "_path_has_torch", lambda path: True)

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        installed = bootstrap_module._install_torch_sidecar_with_embedded_pip(
            runtime="cuda",
            site_packages=site_packages,
        )

    assert installed is True
    assert recorded == []


def test_embedded_install_learned_iqa_sidecar_suppresses_pip_unexpected_import_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    site_packages = tmp_path / "site-packages"

    fake_pip_main_module = _new_module("pip._internal.cli.main")

    class FakePipDeprecationWarning(Warning):
        pass

    def fake_main(args):
        warnings.warn(
            "DEPRECATION: Unexpected import of 'pyiqa' after pip install started. pip 26.3 will enforce this behaviour change.",
            FakePipDeprecationWarning,
            stacklevel=1,
        )
        return 0

    fake_pip_main_module.main = fake_main
    monkeypatch.setitem(sys.modules, "pip._internal.cli.main", fake_pip_main_module)
    monkeypatch.setattr(bootstrap_module, "_path_has_pyiqa", lambda path: True, raising=False)

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        installed = bootstrap_module._install_learned_iqa_sidecar_with_embedded_pip(
            runtime="cuda",
            site_packages=site_packages,
        )

    assert installed is True
    assert recorded == []

