from __future__ import annotations

import os
import sys
from pathlib import Path


_WINDOWS_DLL_DIRECTORY_HANDLES: dict[str, object] = {}
RUNTIME_REEXEC_TARGET_ENV = "SHOTSIEVE_RUNTIME_REEXEC_TARGET"


def _path_has_package(path: Path, package_name: str) -> bool:
    return (path / package_name / "__init__.py").exists() or (path / package_name).is_dir()


def path_has_torch(path: Path) -> bool:
    return _path_has_package(path, "torch")


def path_has_pyiqa(path: Path) -> bool:
    return _path_has_package(path, "pyiqa")


def parse_env_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    normalized = value.strip().casefold()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return None


def is_interactive_console() -> bool:
    stdin = getattr(sys, "stdin", None)
    stdout = getattr(sys, "stdout", None)
    return bool(
        stdin
        and stdout
        and hasattr(stdin, "isatty")
        and hasattr(stdout, "isatty")
        and stdin.isatty()
        and stdout.isatty()
    )


def confirm(prompt: str, *, input_func=input) -> bool:
    try:
        response = input_func(prompt)
    except EOFError:
        return False
    normalized = (response or "").strip().casefold()
    return normalized in {"y", "yes"}


def compose_pythonpath(*, existing: str | None, prepend_path: Path) -> str:
    paths = [str(prepend_path)]
    if existing:
        paths.extend(part for part in existing.split(os.pathsep) if part)

    deduplicated: list[str] = []
    seen: set[str] = set()
    for item in paths:
        if item in seen:
            continue
        seen.add(item)
        deduplicated.append(item)
    return os.pathsep.join(deduplicated)


def runtime_dll_directories(sidecar_path: Path) -> tuple[Path, ...]:
    """Return native-library directories used by a pip ``--target`` sidecar.

    Intel's XPU runtime wheels install their Windows DLLs below
    ``Library/bin`` in the target directory. TheRock's ROCm wheels use
    ``_rocm_sdk_*/bin`` on Windows and split Linux shared libraries across
    ``_rocm_sdk_core`` and ``_rocm_sdk_libraries``. CUDA wheels may put their
    Linux shared libraries below ``nvidia/*/lib``. These are not discovered by
    the frozen sidecar's normal process environment, so keep the search scoped
    to known runtime directories rather than adding unrelated packages.
    """
    root = Path(sidecar_path).resolve()
    candidates = (
        root,
        root / "Library" / "bin",
        root / "torch" / "lib",
        root / "bin",
        root / "lib",
        # ROCm 10 / TheRock Linux wheels use these RPATH targets. Include them
        # explicitly for frozen launches and for diagnostics that rely on
        # LD_LIBRARY_PATH rather than the wheel's relative RPATH alone.
        root / "_rocm_sdk_core" / "lib",
        root / "_rocm_sdk_core" / "lib" / "rocm_sysdeps" / "lib",
        root / "_rocm_sdk_libraries" / "lib",
    )
    # TheRock's Windows wheels place native DLLs in separately installed SDK
    # package bin directories.
    candidates += tuple(sorted(root.glob("_rocm_sdk_*/bin")))
    # CUDA 13 wheels use both the per-library layout (for example,
    # ``nvidia/cublas/lib``) and the consolidated ``nvidia/cu13/lib`` layout.
    # The single-level glob covers both without exposing arbitrary package
    # directories to the dynamic loader.
    candidates += tuple(sorted(root.glob("nvidia/*/lib")))

    directories: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if not resolved.is_dir():
            continue
        key = os.path.normcase(str(resolved))
        if key in seen:
            continue
        seen.add(key)
        directories.append(resolved)
    return tuple(directories)


def compose_runtime_dll_path(*, existing: str | None, sidecar_path: Path) -> str:
    """Prepend sidecar native-library directories to a Windows PATH."""
    if sys.platform != "win32":
        return existing or ""

    paths = [str(path) for path in runtime_dll_directories(sidecar_path)]
    if existing:
        paths.extend(part for part in existing.split(os.pathsep) if part)

    deduplicated: list[str] = []
    seen: set[str] = set()
    for item in paths:
        key = os.path.normcase(item)
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(item)
    return os.pathsep.join(deduplicated)


def compose_runtime_library_path(*, existing: str | None, sidecar_path: Path) -> str:
    """Prepend sidecar shared-library directories to a Linux loader path."""
    if sys.platform != "linux":
        return existing or ""

    paths = [str(path) for path in runtime_dll_directories(sidecar_path)]
    if existing:
        paths.extend(part for part in existing.split(os.pathsep) if part)

    deduplicated: list[str] = []
    seen: set[str] = set()
    for item in paths:
        if item in seen:
            continue
        seen.add(item)
        deduplicated.append(item)
    return os.pathsep.join(deduplicated)


def _maybe_reexec_frozen_linux_rocm(sidecar_path: Path) -> None:
    """Restart a frozen ROCm launcher after its loader paths are activated.

    glibc reads ``LD_LIBRARY_PATH`` when the process starts. Updating the
    variable immediately before importing Torch is therefore too late for
    transitive dependencies such as ``libroctracer64.so``. Re-exec once after
    the sidecar exists so the new process inherits the complete TheRock loader
    path before any ROCm native extension is imported.
    """
    target_id = Path(sidecar_path).name
    if (
        sys.platform != "linux"
        or not getattr(sys, "frozen", False)
        or target_id != "linux-amd-rocm"
        or os.environ.get(RUNTIME_REEXEC_TARGET_ENV) == target_id
    ):
        return

    launch_env = os.environ.copy()
    launch_env[RUNTIME_REEXEC_TARGET_ENV] = target_id
    os.execve(sys.executable, [sys.executable, *sys.argv[1:]], launch_env)


def prepare_runtime_dll_search_path(sidecar_path: Path) -> tuple[Path, ...]:
    """Activate sidecar native-library directories for the current process."""
    if sys.platform == "linux":
        os.environ["LD_LIBRARY_PATH"] = compose_runtime_library_path(
            existing=os.environ.get("LD_LIBRARY_PATH"), sidecar_path=sidecar_path,
        )
        _maybe_reexec_frozen_linux_rocm(sidecar_path)
        return runtime_dll_directories(sidecar_path)

    if sys.platform != "win32":
        return ()

    # Intel's SYCL runtime also loads dependencies through LoadLibraryW.
    # AddDllDirectory alone does not cover that search path. This is required
    # for direct EXE launches as well as launches through the bootstrap helper.
    os.environ["PATH"] = compose_runtime_dll_path(
        existing=os.environ.get("PATH"), sidecar_path=sidecar_path,
    )

    add_dll_directory = getattr(os, "add_dll_directory", None)
    if not callable(add_dll_directory):
        return ()

    registered: list[Path] = []
    for directory in runtime_dll_directories(sidecar_path):
        key = os.path.normcase(str(directory))
        if key in _WINDOWS_DLL_DIRECTORY_HANDLES:
            continue
        try:
            handle = add_dll_directory(str(directory))
        except OSError:
            continue
        _WINDOWS_DLL_DIRECTORY_HANDLES[key] = handle
        registered.append(directory)
    return tuple(registered)


def source_checkout_root(module_file: str | Path, *, package_name: str) -> Path | None:
    resolved_module = Path(module_file).resolve()
    package_dir = resolved_module.parent
    src_dir = package_dir.parent
    project_root = src_dir.parent

    if package_dir.name != package_name:
        return None
    if src_dir.name != "src":
        return None
    if not (project_root / "pyproject.toml").exists():
        return None
    if not (src_dir / package_name / "__init__.py").exists():
        return None

    return project_root.resolve()


__all__ = [
    "RUNTIME_REEXEC_TARGET_ENV",
    "compose_runtime_dll_path",
    "compose_pythonpath",
    "compose_runtime_library_path",
    "confirm",
    "is_interactive_console",
    "parse_env_bool",
    "path_has_pyiqa",
    "path_has_torch",
    "prepare_runtime_dll_search_path",
    "runtime_dll_directories",
    "source_checkout_root",
]
