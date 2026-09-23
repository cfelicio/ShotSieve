from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

from shotsieve.dependency_constraints import (
    COMMON_MODEL_REQUIREMENTS,
    ROCM_TORCH_REQUIREMENTS,
    TORCH_REQUIREMENTS,
    XPU_TORCH_REQUIREMENTS,
    model_requirements_for_runtime,
)
from shotsieve.release_targets import runtime_pack_release_targets


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONSTRAINT_DIR = PROJECT_ROOT / "scripts"


def _pins_from_lines(lines: tuple[str, ...] | list[str]) -> dict[str, list[str]]:
    pins: dict[str, list[str]] = {}
    for raw_line in lines:
        line = raw_line.partition("#")[0].strip()
        if not line or "==" not in line:
            continue
        requirement, version = line.split("==", 1)
        package = requirement.split("[", 1)[0].strip().casefold()
        package = re.sub(r"[-_.]+", "-", package)
        pins.setdefault(package, []).append(version.strip())
    return pins


def _pins_from_file(path: Path) -> dict[str, list[str]]:
    return _pins_from_lines(path.read_text(encoding="utf-8").splitlines())


def _requirements_from_project(entries: list[str]) -> dict[str, str]:
    requirements: dict[str, str] = {}
    for entry in entries:
        match = re.fullmatch(
            r"([A-Za-z0-9_.-]+)(?:==|>=)([A-Za-z0-9.+_-]+)(?:,<([A-Za-z0-9.+_-]+))?",
            entry,
        )
        assert match is not None, f"unsupported pyproject requirement: {entry}"
        package = re.sub(r"[-_.]+", "-", match.group(1)).casefold()
        requirements[package] = match.group(2)
    return requirements


def _assert_pins_match(path: Path, requirements: tuple[str, ...], *, exact: bool) -> None:
    expected = _pins_from_lines(list(requirements))
    actual = _pins_from_file(path)
    for package, versions in expected.items():
        assert actual.get(package) == versions, f"{path.name} pin for {package} drifted"
    if exact:
        assert actual == expected, f"{path.name} has missing, duplicate, or unexpected pins"


def test_release_model_constraints_match_runtime_source_of_truth() -> None:
    _assert_pins_match(
        CONSTRAINT_DIR / "release-constraints.txt",
        COMMON_MODEL_REQUIREMENTS,
        exact=False,
    )
    _assert_pins_match(
        CONSTRAINT_DIR / "release-constraints-torch.txt",
        TORCH_REQUIREMENTS,
        exact=True,
    )


def test_pyproject_dependency_floors_match_qualified_versions() -> None:
    metadata = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = metadata["project"]
    extras = project["optional-dependencies"]

    assert _requirements_from_project(metadata["build-system"]["requires"]) == {
        "setuptools": "84.0.0",
        "wheel": "0.48.0",
    }

    assert _requirements_from_project(project["dependencies"]) == {
        "numpy": "2.5.3",
        "pillow": "12.3.0",
    }
    assert _requirements_from_project(extras["format-loaders"]) == {
        "pillow-heif": "1.8.0",
        "rawpy": "0.27.1",
    }
    assert _requirements_from_project(extras["test"]) == {
        "pytest": "9.1.1",
        "playwright": "1.63.0",
        "ruff": "0.16.8",
    }
    assert _requirements_from_project(extras["lint"]) == {"ruff": "0.16.8"}

    expected_models = _pins_from_lines(list(COMMON_MODEL_REQUIREMENTS))
    assert _requirements_from_project(extras["learned-iqa"]) == {
        package: versions[0] for package, versions in expected_models.items()
    }
    assert _requirements_from_project(extras["windows-build"]) == {
        "pip": "26.2.1",
        "setuptools": "81.0.0",
        "wheel": "0.48.0",
        "packaging": "26.3",
        "pyinstaller": "6.22.3",
    }
    assert "setuptools>=81.0.0,<82" in extras["learned-iqa"]
    assert "setuptools>=81.0.0,<82" in extras["windows-build"]


@pytest.mark.parametrize(
    ("filename", "requirements"),
    (
        ("source-constraints-xpu.txt", XPU_TORCH_REQUIREMENTS),
        ("source-constraints-rocm.txt", ROCM_TORCH_REQUIREMENTS),
    ),
)
def test_accelerator_constraint_files_match_runtime_source_of_truth(
    filename: str,
    requirements: tuple[str, ...],
) -> None:
    constraint_path = CONSTRAINT_DIR / filename
    constraint_lines = constraint_path.read_text(encoding="utf-8").splitlines()
    assert all(
        "[" not in line.partition("#")[0]
        for line in constraint_lines
    ), f"{filename} cannot use requirement extras in pip constraints"
    _assert_pins_match(constraint_path, requirements, exact=True)


@pytest.mark.parametrize(
    ("runtime", "expected"),
    (
        ("cpu", TORCH_REQUIREMENTS),
        ("cuda", TORCH_REQUIREMENTS),
        ("mps", TORCH_REQUIREMENTS),
        ("xpu", XPU_TORCH_REQUIREMENTS),
        ("rocm", ROCM_TORCH_REQUIREMENTS),
    ),
)
def test_runtime_dependency_plan_uses_declared_accelerator_pins(
    runtime: str,
    expected: tuple[str, ...],
) -> None:
    actual = model_requirements_for_runtime(runtime)
    assert actual[: len(COMMON_MODEL_REQUIREMENTS)] == COMMON_MODEL_REQUIREMENTS
    assert actual[len(COMMON_MODEL_REQUIREMENTS) :] == expected


def test_amd_targets_use_the_single_rocm10_stack_on_both_platforms() -> None:
    amd_targets = [
        target
        for target in runtime_pack_release_targets()
        if target.torchVariant == "rocm"
    ]
    assert {target.id for target in amd_targets} == {
        "windows-amd-rocm",
        "linux-amd-rocm",
    }
    assert {target.pythonVersion for target in amd_targets} == {"3.14"}
    assert {target.constraintsFile for target in amd_targets} == {
        "scripts/source-constraints-rocm.txt"
    }
    assert all(target.buildProfile == "runtime-pack" for target in amd_targets)
    assert all(
        model_requirements_for_runtime("rocm")[-3:] == ROCM_TORCH_REQUIREMENTS
        for target in amd_targets
    )


def test_qualification_workflow_uses_current_pins_for_online_and_offline_models() -> None:
    workflow = (
        PROJECT_ROOT / ".github" / "workflows" / "model-smoke.yml"
    ).read_text(encoding="utf-8")
    assert '"3.14"' in workflow
    assert "--offline" in workflow
    for model in ("topiq_nr", "clipiqa", "qrealign-mini"):
        assert model in workflow
