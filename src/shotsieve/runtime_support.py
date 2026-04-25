from __future__ import annotations

import os
import sys
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
    "compose_pythonpath",
    "confirm",
    "is_interactive_console",
    "parse_env_bool",
    "path_has_pyiqa",
    "path_has_torch",
    "source_checkout_root",
]