from __future__ import annotations

import os
from pathlib import Path

import pytest

from shotsieve import runtime_support


def _make_rocm_linux_layout(sidecar: Path) -> tuple[Path, Path, Path]:
    core_lib = sidecar / "_rocm_sdk_core" / "lib"
    sysdeps_lib = core_lib / "rocm_sysdeps" / "lib"
    libraries_lib = sidecar / "_rocm_sdk_libraries" / "lib"
    for path in (core_lib, sysdeps_lib, libraries_lib):
        path.mkdir(parents=True, exist_ok=True)
    return core_lib.resolve(), sysdeps_lib.resolve(), libraries_lib.resolve()


def test_runtime_dll_directories_include_therock_linux_rocm_library_roots(tmp_path: Path) -> None:
    sidecar = tmp_path / "linux-amd-rocm"
    core_lib, sysdeps_lib, libraries_lib = _make_rocm_linux_layout(sidecar)

    directories = runtime_support.runtime_dll_directories(sidecar)

    assert core_lib in directories
    assert sysdeps_lib in directories
    assert libraries_lib in directories


def test_compose_runtime_library_path_prepends_rocm_wheel_libraries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    sidecar = tmp_path / "linux-amd-rocm"
    core_lib, sysdeps_lib, libraries_lib = _make_rocm_linux_layout(sidecar)
    monkeypatch.setattr(runtime_support.sys, "platform", "linux")

    composed = runtime_support.compose_runtime_library_path(
        existing="/system/lib",
        sidecar_path=sidecar,
    ).split(os.pathsep)

    assert composed.index(str(core_lib)) < composed.index("/system/lib")
    assert composed.index(str(sysdeps_lib)) < composed.index("/system/lib")
    assert composed.index(str(libraries_lib)) < composed.index("/system/lib")


def test_prepare_runtime_path_reexecs_frozen_linux_rocm_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    sidecar = tmp_path / "linux-amd-rocm"
    core_lib, sysdeps_lib, libraries_lib = _make_rocm_linux_layout(sidecar)
    executable = "/opt/shotsieve/ShotSieve-AMD-ROCm"
    monkeypatch.setattr(runtime_support.sys, "platform", "linux")
    monkeypatch.setattr(runtime_support.sys, "frozen", True, raising=False)
    monkeypatch.setattr(runtime_support.sys, "executable", executable)
    monkeypatch.setattr(runtime_support.sys, "argv", [executable, "--check-runtime"])
    monkeypatch.delenv(runtime_support.RUNTIME_REEXEC_TARGET_ENV, raising=False)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/system/lib")
    exec_calls: list[tuple[str, list[str], dict[str, str]]] = []

    def fake_exec(path: str, args: list[str], env: dict[str, str]) -> None:
        exec_calls.append((path, args, env))

    monkeypatch.setattr(runtime_support.os, "execve", fake_exec)

    runtime_support.prepare_runtime_dll_search_path(sidecar)

    assert len(exec_calls) == 1
    assert exec_calls[0][0] == executable
    assert exec_calls[0][1] == [executable, "--check-runtime"]
    launch_env = exec_calls[0][2]
    assert launch_env[runtime_support.RUNTIME_REEXEC_TARGET_ENV] == "linux-amd-rocm"
    loader_paths = launch_env["LD_LIBRARY_PATH"].split(os.pathsep)
    assert str(core_lib) in loader_paths
    assert str(sysdeps_lib) in loader_paths
    assert str(libraries_lib) in loader_paths
    assert "/system/lib" in loader_paths


def test_prepare_runtime_path_does_not_reexec_rocm_twice(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    sidecar = tmp_path / "linux-amd-rocm"
    _make_rocm_linux_layout(sidecar)
    monkeypatch.setattr(runtime_support.sys, "platform", "linux")
    monkeypatch.setattr(runtime_support.sys, "frozen", True, raising=False)
    monkeypatch.setenv(runtime_support.RUNTIME_REEXEC_TARGET_ENV, "linux-amd-rocm")
    monkeypatch.setattr(
        runtime_support.os,
        "execve",
        lambda *args: pytest.fail("ROCm launcher must not re-exec twice"),
    )

    runtime_support.prepare_runtime_dll_search_path(sidecar)


def test_prepare_runtime_path_leaves_cuda_reexec_to_desktop_launcher(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    sidecar = tmp_path / "linux-nvidia-cuda"
    (sidecar / "nvidia" / "cublas" / "lib").mkdir(parents=True)
    monkeypatch.setattr(runtime_support.sys, "platform", "linux")
    monkeypatch.setattr(runtime_support.sys, "frozen", True, raising=False)
    monkeypatch.delenv(runtime_support.RUNTIME_REEXEC_TARGET_ENV, raising=False)
    monkeypatch.setattr(
        runtime_support.os,
        "execve",
        lambda *args: pytest.fail("CUDA re-exec remains owned by desktop.py"),
    )

    runtime_support.prepare_runtime_dll_search_path(sidecar)
