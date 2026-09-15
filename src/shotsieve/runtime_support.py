from __future__ import annotations

import os
import sys
from collections.abc import Callable
from pathlib import Path


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


class RuntimeSupport:
    """Small dependency facade for runtime bootstrap and desktop seams.

    The methods intentionally resolve the module-level functions when called.
    This keeps the facade injectable while preserving the existing tests and
    integrations that monkeypatch those shared helpers.
    """

    def path_has_torch(self, path: Path) -> bool:
        return path_has_torch(path)

    def path_has_pyiqa(self, path: Path) -> bool:
        return path_has_pyiqa(path)

    def parse_env_bool(self, value: str | None) -> bool | None:
        return parse_env_bool(value)

    def is_interactive_console(self) -> bool:
        return is_interactive_console()

    def confirm(self, prompt: str, *, input_func: Callable[[str], str] = input) -> bool:
        return confirm(prompt, input_func=input_func)

    def compose_pythonpath(self, *, existing: str | None, prepend_path: Path) -> str:
        return compose_pythonpath(existing=existing, prepend_path=prepend_path)


shared_runtime_support = RuntimeSupport()


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
    "RuntimeSupport",
    "compose_pythonpath",
    "confirm",
    "is_interactive_console",
    "parse_env_bool",
    "path_has_pyiqa",
    "path_has_torch",
    "shared_runtime_support",
    "source_checkout_root",
]
