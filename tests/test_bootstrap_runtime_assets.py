from __future__ import annotations

from email.message import Message
import hashlib
import io
import json
import sys
import types
from pathlib import Path
import tarfile
import urllib.error
from typing import Any
import warnings
import zipfile

import pytest

from shotsieve import bootstrap_assets as bootstrap_module
from shotsieve import bootstrap_sidecar as sidecar_module


def _new_module(name: str) -> Any:
    return types.ModuleType(name)


def _not_found_http_error(url: str) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(url=url, code=404, msg="Not Found", hdrs=Message(), fp=None)


def test_ensure_runtime_asset_falls_back_to_local_archive_when_download_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build_root = tmp_path / "local-build"
    build_root.mkdir(parents=True)
    archive_name = "ShotSieve-windows-nvidia-cuda-x64.zip"
    executable_name = "ShotSieve-NVIDIA-CUDA.exe"
    variant_folder = "ShotSieve-windows-nvidia-cuda"

    local_archive = build_root / archive_name
    with zipfile.ZipFile(local_archive, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{variant_folder}/{executable_name}", "fake-binary")
    archive_sha256 = bootstrap_module.sha256_file(local_archive)

    monkeypatch.chdir(build_root)

    attempted_urls: list[str] = []

    def fake_open_url(url: str):
        attempted_urls.append(url)
        raise _not_found_http_error(url)

    monkeypatch.setattr(bootstrap_module, "open_url", fake_open_url)

    asset = bootstrap_module.RuntimeAsset(
        id="windows-nvidia-cuda",
        platform="windows",
        runtime="cuda",
        url="https://github.com/cfelicio/ShotSieve/releases/latest/download/ShotSieve-windows-nvidia-cuda-x64.zip",
        archive_name=archive_name,
        executable_name=executable_name,
        variant_folder_name=variant_folder,
        sha256=archive_sha256,
    )

    executable = bootstrap_module.ensure_runtime_asset(asset, runtime_root=tmp_path / "runtime")

    assert attempted_urls == [asset.url]
    assert executable.exists()
    assert executable.name == executable_name
    assert not (tmp_path / "runtime" / "downloads" / archive_name).exists()


def test_download_archive_reassembles_verified_split_parts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    part_bytes = (b"first-part-", b"second-part")
    asset = bootstrap_module.RuntimeAsset(
        id="linux-amd-rocm",
        platform="linux",
        runtime="rocm",
        url=None,
        archive_name="ShotSieve-linux-amd-rocm-x64.tar.gz",
        executable_name="ShotSieve-AMD-ROCm",
        variant_folder_name="ShotSieve-linux-amd-rocm",
        sha256=hashlib.sha256(b"".join(part_bytes)).hexdigest(),
        parts=tuple(
            bootstrap_module.RuntimeAssetPart(
                archive_name=f"archive.part-{index:03d}",
                url=f"https://example.invalid/part-{index}",
                sha256=hashlib.sha256(part).hexdigest(),
            )
            for index, part in enumerate(part_bytes)
        ),
    )

    def fake_open_url(url: str):
        index = int(url.rsplit("-", 1)[-1])
        return io.BytesIO(part_bytes[index])

    monkeypatch.setattr(bootstrap_module, "open_url", fake_open_url)
    archive_path = tmp_path / "downloads" / asset.archive_name
    archive_path.parent.mkdir()

    bootstrap_module._download_archive_with_local_fallback(asset=asset, archive_path=archive_path)

    assert archive_path.read_bytes() == b"".join(part_bytes)


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
        sha256="a" * 64,
    )

    def fake_open_url(url: str):
        raise _not_found_http_error(url)

    monkeypatch.setattr(bootstrap_module, "open_url", fake_open_url)

    with pytest.raises(SystemExit, match="Failed to download runtime archive"):
        bootstrap_module.ensure_runtime_asset(asset, runtime_root=tmp_path)


def test_ensure_runtime_asset_does_not_reuse_old_empty_marker_without_verified_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_root = tmp_path / "runtime"
    install_dir = runtime_root / "installs" / "windows-nvidia-cuda"
    executable = install_dir / "ShotSieve-windows-nvidia-cuda" / "ShotSieve-NVIDIA-CUDA.exe"
    executable.parent.mkdir(parents=True, exist_ok=True)
    executable.write_bytes(b"fake-runtime")
    (install_dir / ".asset-sha256").write_text("", encoding="utf-8")

    called_urls: list[str] = []

    def fake_open_url(url: str):
        called_urls.append(url)
        raise _not_found_http_error(url)

    monkeypatch.setattr(bootstrap_module, "open_url", fake_open_url)
    monkeypatch.setattr(bootstrap_module, "find_local_runtime_archive", lambda archive_name: None)

    asset = bootstrap_module.RuntimeAsset(
        id="windows-nvidia-cuda",
        platform="windows",
        runtime="cuda",
        url="https://example.invalid/ShotSieve-windows-nvidia-cuda-x64.zip",
        archive_name="ShotSieve-windows-nvidia-cuda-x64.zip",
        executable_name="ShotSieve-NVIDIA-CUDA.exe",
        variant_folder_name="ShotSieve-windows-nvidia-cuda",
        sha256="a" * 64,
    )

    with pytest.raises(SystemExit, match="Failed to download runtime archive"):
        bootstrap_module.ensure_runtime_asset(asset, runtime_root=runtime_root)

    assert called_urls == [asset.url]


def test_ensure_runtime_asset_rejects_malformed_digest_before_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asset = bootstrap_module.RuntimeAsset(
        id="windows-cpu",
        platform="windows",
        runtime="cpu",
        url="https://example.invalid/runtime.zip",
        archive_name="runtime.zip",
        executable_name="ShotSieve-CPU.exe",
        variant_folder_name="ShotSieve-windows-cpu",
        sha256="not-a-sha256",
    )
    monkeypatch.setattr(bootstrap_module, "open_url", lambda url: pytest.fail(f"downloaded {url}"))

    with pytest.raises(SystemExit, match="valid 64-digit SHA-256"):
        bootstrap_module.ensure_runtime_asset(asset, runtime_root=tmp_path)


def test_ensure_runtime_asset_rejects_mismatched_archive_before_extraction(
    tmp_path: Path,
) -> None:
    archive_name = "ShotSieve-windows-cpu-x64.zip"
    archive_path = tmp_path / "downloads" / archive_name
    archive_path.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("ShotSieve-windows-cpu/ShotSieve-CPU.exe", "fake-binary")

    asset = bootstrap_module.RuntimeAsset(
        id="windows-cpu",
        platform="windows",
        runtime="cpu",
        url="https://example.invalid/runtime.zip",
        archive_name=archive_name,
        executable_name="ShotSieve-CPU.exe",
        variant_folder_name="ShotSieve-windows-cpu",
        sha256="b" * 64,
    )

    with pytest.raises(SystemExit, match="Downloaded archive hash mismatch"):
        bootstrap_module.ensure_runtime_asset(asset, runtime_root=tmp_path)

    assert not (tmp_path / "installs" / asset.id).exists()


def test_ensure_runtime_asset_does_not_publish_valid_archive_with_missing_executable(
    tmp_path: Path,
) -> None:
    archive_name = "ShotSieve-windows-cpu-x64.zip"
    archive_path = tmp_path / "downloads" / archive_name
    archive_path.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("ShotSieve-windows-cpu/readme.txt", "not a launcher")

    asset = bootstrap_module.RuntimeAsset(
        id="windows-cpu",
        platform="windows",
        runtime="cpu",
        url="https://example.invalid/runtime.zip",
        archive_name=archive_name,
        executable_name="ShotSieve-CPU.exe",
        variant_folder_name="ShotSieve-windows-cpu",
        sha256=bootstrap_module.sha256_file(archive_path),
    )

    with pytest.raises(SystemExit, match="executable 'ShotSieve-CPU.exe' was not found"):
        bootstrap_module.ensure_runtime_asset(asset, runtime_root=tmp_path)

    assert not (tmp_path / "installs" / asset.id).exists()


def test_ensure_runtime_asset_prefers_colocated_frozen_runtime_executable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher_dir = tmp_path / "bootstrap-launcher-nvidia" / "ShotSieve"
    launcher_dir.mkdir(parents=True)
    launcher_exe = launcher_dir / "ShotSieve.exe"
    launcher_exe.write_bytes(b"bootstrap")

    runtime_exe = launcher_dir / "ShotSieve-NVIDIA-CUDA.exe"
    runtime_exe.write_bytes(b"runtime")

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(launcher_exe), raising=False)

    called_urls: list[str] = []

    def fake_open_url(url: str):
        called_urls.append(url)
        raise _not_found_http_error(url)

    monkeypatch.setattr(bootstrap_module, "open_url", fake_open_url)

    asset = bootstrap_module.RuntimeAsset(
        id="windows-nvidia-cuda",
        platform="windows",
        runtime="cuda",
        url="https://example.invalid/ShotSieve-windows-nvidia-cuda-x64.zip",
        archive_name="ShotSieve-windows-nvidia-cuda-x64.zip",
        executable_name="ShotSieve-NVIDIA-CUDA.exe",
        variant_folder_name="ShotSieve-windows-nvidia-cuda",
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


def test_install_torch_sidecar_uses_helper_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    site_packages = tmp_path / "site-packages"
    installer_calls: list[tuple[str, Path, bool]] = []

    def fake_subprocess_install(*, operation: str, runtime: str, site_packages: Path, force_reinstall: bool, output_func=print):
        installer_calls.append((operation, site_packages, force_reinstall))
        return True

    monkeypatch.setattr(sidecar_module, "_run_sidecar_install_subprocess", fake_subprocess_install)

    installed = sidecar_module.install_torch_sidecar(runtime="cuda", site_packages=site_packages)

    assert installed is True
    assert installer_calls == [("torch", site_packages, False)]


def test_install_learned_iqa_sidecar_uses_helper_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    site_packages = tmp_path / "site-packages"
    installer_calls: list[tuple[str, str, Path, bool]] = []

    def fake_subprocess_install(*, operation: str, runtime: str, site_packages: Path, force_reinstall: bool, output_func=print):
        installer_calls.append((operation, runtime, site_packages, force_reinstall))
        return True

    monkeypatch.setattr(sidecar_module, "_run_sidecar_install_subprocess", fake_subprocess_install)

    installed = sidecar_module.install_learned_iqa_sidecar(runtime="cuda", site_packages=site_packages)

    assert installed is True
    assert len(installer_calls) == 1
    assert installer_calls[0][0] == "learned-iqa"
    assert installer_calls[0][1] == "cuda"
    assert installer_calls[0][2] != site_packages
    assert installer_calls[0][3] is False


def test_install_learned_iqa_sidecar_replaces_old_tree_after_clean_staged_install(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    site_packages = tmp_path / "site-packages"
    site_packages.mkdir(parents=True)
    (site_packages / "torch").mkdir()
    (site_packages / "torch" / "__init__.py").write_text("torch", encoding="utf-8")
    (site_packages / "torchvision").mkdir()
    (site_packages / "torchvision" / "__init__.py").write_text("torchvision", encoding="utf-8")
    (site_packages / "transformers").mkdir()
    (site_packages / "transformers" / "stale-module.py").write_text("stale", encoding="utf-8")

    def fake_subprocess_install(*, operation: str, runtime: str, site_packages: Path, force_reinstall: bool, output_func=print):
        assert operation == "learned-iqa"
        assert runtime == "cuda"
        assert force_reinstall is True
        assert (site_packages / "torch" / "__init__.py").read_text(encoding="utf-8") == "torch"
        assert (site_packages / "torchvision" / "__init__.py").read_text(encoding="utf-8") == "torchvision"
        assert not (site_packages / "transformers" / "stale-module.py").exists()
        (site_packages / "pyiqa").mkdir(parents=True)
        (site_packages / "pyiqa" / "__init__.py").write_text("", encoding="utf-8")
        (site_packages / "transformers").mkdir()
        (site_packages / "transformers" / "fresh-module.py").write_text("fresh", encoding="utf-8")
        return True

    monkeypatch.setattr(sidecar_module, "_run_sidecar_install_subprocess", fake_subprocess_install)

    installed = sidecar_module.install_learned_iqa_sidecar(
        runtime="cuda",
        site_packages=site_packages,
        force_reinstall=True,
    )

    assert installed is True
    assert not (site_packages / "transformers" / "stale-module.py").exists()
    assert (site_packages / "transformers" / "fresh-module.py").read_text(encoding="utf-8") == "fresh"
    state = json.loads((site_packages / ".shotsieve-runtime.json").read_text(encoding="utf-8"))
    assert state["plan"]["target_id"] == site_packages.name
    assert state["learned_iqa_complete"] is True


def test_prepare_learned_iqa_staging_skips_python_bytecode_caches(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    staging_dir = tmp_path / "staging"
    regular_package = source_dir / "networkx" / "algorithms" / "centrality" / "tests"
    regular_package.mkdir(parents=True)
    (regular_package / "test_centrality.py").write_text("source", encoding="utf-8")
    cache_dir = regular_package / "__pycache__"
    cache_dir.mkdir()
    (cache_dir / "test_centrality.cpython-314.pyc").write_bytes(b"bytecode")
    (source_dir / "stale-module.pyc").write_bytes(b"bytecode")

    sidecar_module._prepare_learned_iqa_staging(
        source_dir=source_dir,
        staging_dir=staging_dir,
        runtime="cuda",
    )

    assert (staging_dir / "networkx" / "algorithms" / "centrality" / "tests" / "test_centrality.py").read_text(encoding="utf-8") == "source"
    assert not (staging_dir / "networkx" / "algorithms" / "centrality" / "tests" / "__pycache__").exists()
    assert not (staging_dir / "stale-module.pyc").exists()


def test_run_sidecar_install_subprocess_uses_frozen_helper_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    site_packages = tmp_path / "site packages"
    calls: list[dict[str, Any]] = []
    messages: list[str] = []

    def fake_run(command, **kwargs):
        calls.append({"command": command, **kwargs})
        return type("Completed", (), {"returncode": 0, "stdout": "installed\n", "stderr": ""})()

    monkeypatch.setattr(sidecar_module.sys, "frozen", True, raising=False)
    monkeypatch.setattr(sidecar_module.subprocess, "run", fake_run)

    installed = sidecar_module._run_sidecar_install_subprocess(
        operation="torch",
        runtime="xpu",
        site_packages=site_packages,
        force_reinstall=True,
        output_func=messages.append,
    )

    assert installed is True
    assert calls[0]["command"][0] == sys.executable
    assert calls[0]["command"][1:] == [
        sidecar_module.SIDECAR_INSTALL_COMMAND,
        "torch",
        "xpu",
        str(site_packages),
        "1",
    ]
    assert calls[0]["env"] is None
    assert calls[0]["capture_output"] is True
    assert messages == ["installed"]


def test_run_sidecar_install_subprocess_uses_source_module(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_run(command, **kwargs):
        calls.append({"command": command, **kwargs})
        return type("Completed", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(sidecar_module.sys, "frozen", False, raising=False)
    monkeypatch.setattr(sidecar_module.subprocess, "run", fake_run)

    installed = sidecar_module._run_sidecar_install_subprocess(
        operation="learned-iqa",
        runtime="cpu",
        site_packages=tmp_path / "staging",
        force_reinstall=False,
    )

    assert installed is True
    assert calls[0]["command"][:3] == [
        sys.executable,
        "-m",
        "shotsieve.bootstrap_sidecar",
    ]
    assert calls[0]["command"][3] == sidecar_module.SIDECAR_INSTALL_COMMAND
    assert str(Path(sidecar_module.__file__).resolve().parent.parent) in calls[0]["env"]["PYTHONPATH"]


def test_dispatch_sidecar_install_command_runs_requested_installer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    site_packages = tmp_path / "staging"
    calls: list[tuple[str, str, Path, bool]] = []

    def fake_torch_install(*, runtime: str, site_packages: Path, force_reinstall: bool = False):
        calls.append(("torch", runtime, site_packages, force_reinstall))
        return True

    monkeypatch.setattr(sidecar_module, "_install_torch_sidecar_with_embedded_pip", fake_torch_install)
    command_args = [
        sidecar_module.SIDECAR_INSTALL_COMMAND,
        "torch",
        "xpu",
        str(site_packages),
        "1",
    ]

    exit_code = sidecar_module.dispatch_sidecar_install_command(command_args)

    assert exit_code == 0
    assert calls == [("torch", "xpu", site_packages, True)]


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
    monkeypatch.setattr(sidecar_module, "path_has_pyiqa", lambda path: True, raising=False)

    installed = sidecar_module._install_learned_iqa_sidecar_with_embedded_pip(
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
    monkeypatch.setattr(sidecar_module, "path_has_pyiqa", lambda path: True, raising=False)

    installed = sidecar_module._install_learned_iqa_sidecar_with_embedded_pip(
        runtime="cuda",
        site_packages=site_packages,
    )

    assert installed is True
    pyiqa_args = next(args for args in captured_args if any(arg.startswith("pyiqa==") for arg in args))
    assert "--no-deps" in pyiqa_args


def test_embedded_install_learned_iqa_sidecar_does_not_replace_loaded_torch(
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
    monkeypatch.setattr(sidecar_module, "path_has_pyiqa", lambda path: True, raising=False)

    installed = sidecar_module._install_learned_iqa_sidecar_with_embedded_pip(
        runtime="cuda",
        site_packages=site_packages,
    )

    assert installed is True
    requirements = {"timm==1.0.30", "openai-clip==1.0.1", "accelerate==1.15.0", "facexlib==0.3.0"}
    args_by_requirement = {
        requirement: args
        for args in captured_args
        for requirement in requirements
        if requirement in args
    }
    for requirement in ("timm==1.0.30", "openai-clip==1.0.1", "accelerate==1.15.0", "facexlib==0.3.0"):
        assert "--no-deps" in args_by_requirement[requirement]
    assert "--no-build-isolation" in args_by_requirement["openai-clip==1.0.1"]


def test_openai_clip_source_archive_installs_pure_python_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from shotsieve import bootstrap_sidecar as sidecar_module

    archive_buffer = io.BytesIO()
    with tarfile.open(fileobj=archive_buffer, mode="w:gz") as archive:
        files = {
            "clip/__init__.py": b"from .clip import load\n",
            "clip/clip.py": b"def load(*args, **kwargs):\n    return None\n",
            "clip/bpe_simple_vocab_16e6.txt.gz": b"tokenizer-data",
        }
        for relative_name, content in files.items():
            member = tarfile.TarInfo(f"openai-clip-1.0.1/{relative_name}")
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    archive_bytes = archive_buffer.getvalue()

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return archive_bytes

    monkeypatch.setattr(sidecar_module, "_OPENAI_CLIP_SOURCE_SHA256", hashlib.sha256(archive_bytes).hexdigest())
    monkeypatch.setattr(sidecar_module.urllib.request, "urlopen", lambda *_args, **_kwargs: _Response())

    sidecar_module._install_openai_clip_source(tmp_path)

    assert (tmp_path / "clip" / "__init__.py").read_bytes() == files["clip/__init__.py"]
    assert (tmp_path / "clip" / "bpe_simple_vocab_16e6.txt.gz").read_bytes() == files["clip/bpe_simple_vocab_16e6.txt.gz"]
    assert (tmp_path / "openai_clip-1.0.1.dist-info" / "METADATA").read_text(encoding="utf-8").startswith(
        "Metadata-Version: 2.1\nName: openai-clip\nVersion: 1.0.1\n"
    )


def test_frozen_amd_install_uses_bundled_selector_wheel_and_stable_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_args: list[list[str]] = []

    def fake_pip_main(args):
        captured_args.append(list(args))
        return 0

    monkeypatch.setattr(sidecar_module.sys, "frozen", True, raising=False)
    monkeypatch.setattr(sidecar_module, "_load_embedded_pip_main", lambda: fake_pip_main)
    monkeypatch.setattr(sidecar_module, "_patch_distlib_finder_for_frozen", lambda: None)
    monkeypatch.setattr(sidecar_module, "_patch_pip_scriptmaker_for_embedded_install", lambda: None)
    monkeypatch.setattr(sidecar_module, "path_has_torch", lambda _path: True)

    target_id = "windows-amd-rocm"
    runtime_root = tmp_path / "portable" / "data" / "runtime"
    site_packages = runtime_root / "site-packages" / target_id
    selector_wheel_dir = runtime_root / "wheels"
    selector_wheel_dir.mkdir(parents=True)
    selector_wheel = selector_wheel_dir / "rocm-10.0.0-py3-none-any.whl"
    selector_wheel.write_bytes(b"built-selector-wheel")

    installed = sidecar_module._install_torch_sidecar_with_embedded_pip(
        runtime="rocm",
        site_packages=site_packages,
    )

    assert installed is True
    assert len(captured_args) == 1
    args_text = " ".join(captured_args[0])
    assert str(selector_wheel) in args_text
    assert "torch[device-gfx1103]==2.13.0+rocm10.0.0" in args_text
    assert "torchvision[device-gfx1103]==0.28.0+rocm10.0.0" in args_text
    assert "https://stable.repo.amd.com/rocm/whl-next/" in args_text
    assert "--only-binary=rocm" in args_text
    assert str(selector_wheel_dir) in args_text


def test_frozen_amd_install_fails_cleanly_without_selector_wheel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pip_called = False

    def fail_if_pip_is_called(_args):
        nonlocal pip_called
        pip_called = True
        raise AssertionError("ROCm 10 install requires its bundled selector wheel")

    messages: list[str] = []
    monkeypatch.setattr(sidecar_module.sys, "frozen", True, raising=False)
    monkeypatch.setattr(sidecar_module, "_load_embedded_pip_main", lambda: fail_if_pip_is_called)
    monkeypatch.setattr(sidecar_module, "_patch_distlib_finder_for_frozen", lambda: None)
    monkeypatch.setattr(sidecar_module, "_patch_pip_scriptmaker_for_embedded_install", lambda: None)

    site_packages = (
        tmp_path / "portable" / "data" / "runtime" / "site-packages"
        / "windows-amd-rocm"
    )
    installed = sidecar_module._install_torch_sidecar_with_embedded_pip(
        runtime="rocm",
        site_packages=site_packages,
        output_func=messages.append,
    )

    assert installed is False
    assert pip_called is False
    assert any("selector wheel is missing or ambiguous" in message for message in messages)


def test_frozen_learned_iqa_install_uses_openai_clip_source_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from shotsieve import bootstrap_sidecar as sidecar_module

    calls: list[Path] = []

    def fake_source_install(site_packages: Path) -> None:
        calls.append(site_packages)

    def fail_if_pip_is_called(_args):
        raise AssertionError("frozen openai-clip install should not invoke pip")

    monkeypatch.setattr(sidecar_module.sys, "frozen", True, raising=False)
    monkeypatch.setattr(sidecar_module, "_install_openai_clip_source", fake_source_install)
    monkeypatch.setattr(sidecar_module, "_load_embedded_pip_main", lambda: fail_if_pip_is_called)
    monkeypatch.setattr(sidecar_module, "_patch_distlib_finder_for_frozen", lambda: None)
    monkeypatch.setattr(sidecar_module, "_patch_pip_scriptmaker_for_embedded_install", lambda: None)
    monkeypatch.setattr(sidecar_module, "_learned_iqa_packages_for_runtime", lambda _runtime: ["openai-clip==1.0.1"])
    monkeypatch.setattr(sidecar_module, "path_has_pyiqa", lambda _path: True)

    installed = sidecar_module._install_learned_iqa_sidecar_with_embedded_pip(
        runtime="cuda",
        site_packages=tmp_path,
    )

    assert installed is True
    assert calls == [tmp_path]


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
    monkeypatch.setattr(sidecar_module, "path_has_pyiqa", lambda path: True, raising=False)

    installed = sidecar_module._install_learned_iqa_sidecar_with_embedded_pip(
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
    monkeypatch.setattr(sidecar_module, "path_has_pyiqa", lambda path: True, raising=False)

    installed = sidecar_module._install_learned_iqa_sidecar_with_embedded_pip(
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
    monkeypatch.setattr(sidecar_module, "path_has_pyiqa", lambda path: True, raising=False)

    installed = sidecar_module._install_learned_iqa_sidecar_with_embedded_pip(
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
    monkeypatch.setattr(sidecar_module.importlib, "import_module", fake_import_module)

    sidecar_module._patch_distlib_finder_for_frozen()

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
    monkeypatch.setattr(sidecar_module.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(sidecar_module.pkgutil, "get_loader", lambda name: None, raising=False)

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        sidecar_module._patch_distlib_finder_for_frozen()

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
    monkeypatch.setattr(sidecar_module.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(sidecar_module.pkgutil, "get_loader", lambda name: PkgutilLoader(), raising=False)

    sidecar_module._patch_distlib_finder_for_frozen()

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
    monkeypatch.setattr(sidecar_module.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(sidecar_module.pkgutil, "get_loader", lambda name: DistlibLoader(), raising=False)

    sidecar_module._patch_distlib_finder_for_frozen()

    resolved = fake_resources_module.finder("pip._vendor.distlib")
    assert isinstance(resolved, FakeResourceFinder)
    assert resolved.package == "pip._vendor.distlib"


def test_embedded_install_torch_sidecar_calls_distlib_patch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    site_packages = tmp_path / "site-packages"
    patch_calls: list[bool] = []

    monkeypatch.setattr(sidecar_module, "_patch_distlib_finder_for_frozen", lambda: patch_calls.append(True))

    fake_pip_main_module = _new_module("pip._internal.cli.main")

    def fake_main(args):
        return 0

    fake_pip_main_module.main = fake_main
    monkeypatch.setitem(sys.modules, "pip._internal.cli.main", fake_pip_main_module)
    monkeypatch.setattr(sidecar_module, "path_has_torch", lambda path: True)

    installed = sidecar_module._install_torch_sidecar_with_embedded_pip(
        runtime="cuda",
        site_packages=site_packages,
    )

    assert installed is True
    assert patch_calls == [True]


def test_embedded_install_torch_sidecar_allows_xpu_native_dependencies(
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
    monkeypatch.setattr(sidecar_module, "path_has_torch", lambda path: True)

    installed = sidecar_module._install_torch_sidecar_with_embedded_pip(
        runtime="xpu",
        site_packages=site_packages,
    )

    assert installed is True
    assert captured_args
    xpu_args = captured_args[0]
    assert "--no-deps" not in xpu_args
    assert "torch==2.14.0+xpu" in xpu_args
    assert "torchvision==0.29.0+xpu" in xpu_args
    assert "https://download.pytorch.org/whl/xpu" in xpu_args


def test_patch_pip_scriptmaker_for_embedded_install_disables_launchers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_wheel_module = _new_module("pip._internal.operations.install.wheel")
    original_import_module = sidecar_module.importlib.import_module

    class FakePipScriptMaker:
        def __init__(self, *args, **kwargs):
            self.add_launchers = True

    fake_wheel_module.PipScriptMaker = FakePipScriptMaker

    def fake_import_module(name: str):
        if name == "pip._internal.operations.install.wheel":
            return fake_wheel_module
        return original_import_module(name)

    monkeypatch.setattr(sidecar_module.importlib, "import_module", fake_import_module)

    sidecar_module._patch_pip_scriptmaker_for_embedded_install()

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

    monkeypatch.setattr(sidecar_module.importlib, "import_module", fake_import_module)

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        sidecar_module._patch_pip_scriptmaker_for_embedded_install()

    assert recorded == []


def test_embedded_install_torch_sidecar_calls_scriptmaker_patch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    site_packages = tmp_path / "site-packages"
    patch_calls: list[bool] = []

    monkeypatch.setattr(
        sidecar_module,
        "_patch_pip_scriptmaker_for_embedded_install",
        lambda: patch_calls.append(True),
    )

    fake_pip_main_module = _new_module("pip._internal.cli.main")

    def fake_main(args):
        return 0

    fake_pip_main_module.main = fake_main
    monkeypatch.setitem(sys.modules, "pip._internal.cli.main", fake_pip_main_module)
    monkeypatch.setattr(sidecar_module, "path_has_torch", lambda path: True)

    installed = sidecar_module._install_torch_sidecar_with_embedded_pip(
        runtime="cuda",
        site_packages=site_packages,
    )

    assert installed is True
    assert patch_calls == [True]
