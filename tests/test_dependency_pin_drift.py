from __future__ import annotations

import re
from pathlib import Path

import pytest

from shotsieve.dependency_constraints import (
    COMMON_MODEL_REQUIREMENTS,
    ROCM10_GFX1103_TORCH_REQUIREMENTS,
    ROCM_TORCH_REQUIREMENTS,
    ROCM_WINDOWS_TORCH_REQUIREMENTS,
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


@pytest.mark.parametrize(
    ("filename", "requirements"),
    (
        ("source-constraints-xpu.txt", XPU_TORCH_REQUIREMENTS),
        ("source-constraints-rocm.txt", ROCM_TORCH_REQUIREMENTS),
        ("source-constraints-rocm-windows.txt", ROCM_WINDOWS_TORCH_REQUIREMENTS),
        ("source-constraints-rocm10-gfx1103.txt", ROCM10_GFX1103_TORCH_REQUIREMENTS),
    ),
)
def test_accelerator_constraint_files_match_runtime_source_of_truth(
    filename: str,
    requirements: tuple[str, ...],
) -> None:
    _assert_pins_match(CONSTRAINT_DIR / filename, requirements, exact=True)


@pytest.mark.parametrize(
    ("runtime", "platform", "expected"),
    (
        ("cpu", "linux", TORCH_REQUIREMENTS),
        ("cuda", "windows", TORCH_REQUIREMENTS),
        ("mps", "macos", TORCH_REQUIREMENTS),
        ("xpu", "linux", XPU_TORCH_REQUIREMENTS),
        ("rocm", "linux", ROCM_TORCH_REQUIREMENTS),
        ("rocm", "windows", ROCM_WINDOWS_TORCH_REQUIREMENTS),
    ),
)
def test_runtime_dependency_plan_uses_declared_accelerator_pins(
    runtime: str,
    platform: str,
    expected: tuple[str, ...],
) -> None:
    actual = model_requirements_for_runtime(runtime, platform_name=platform)
    assert actual[: len(COMMON_MODEL_REQUIREMENTS)] == COMMON_MODEL_REQUIREMENTS
    assert actual[len(COMMON_MODEL_REQUIREMENTS) :] == expected


def test_rocm10_runtime_targets_select_the_candidate_constraint_file() -> None:
    candidates = [
        target
        for target in runtime_pack_release_targets()
        if target.torchVariant == "rocm10-gfx1103"
    ]
    assert {target.id for target in candidates} == {
        "windows-amd-rocm10-gfx1103",
        "linux-amd-rocm10-gfx1103",
    }
    assert {
        target.constraintsFile for target in candidates
    } == {"scripts/source-constraints-rocm10-gfx1103.txt"}


def test_hugging_face_upgrade_profiles_are_isolated_and_synced_to_current_pins() -> None:
    production = _pins_from_lines(list(COMMON_MODEL_REQUIREMENTS))
    hub_only = {
        "huggingface-hub": ["1.32.0"],
        "transformers": production["transformers"],
    }
    transformers_only = {
        "huggingface-hub": production["huggingface-hub"],
        "transformers": ["5.17.0"],
    }
    combined = {
        "huggingface-hub": ["1.32.0"],
        "transformers": ["5.17.0"],
    }

    for track, expected in (
        ("hub", hub_only),
        ("transformers", transformers_only),
        ("combined", combined),
    ):
        assert _pins_from_file(
            CONSTRAINT_DIR / f"model-upgrade-{track}-constraints.txt"
        ) == expected

    workflow = (
        PROJECT_ROOT / ".github" / "workflows" / "dependency-upgrade-qualification.yml"
    ).read_text(encoding="utf-8")
    assert "workflow_dispatch:" in workflow
    assert "model-upgrade-${{ matrix.upgrade_track }}-constraints.txt" in workflow
    assert "from transformers import AutoModelForImageTextToText" in workflow
    for track in ("hub", "transformers", "combined"):
        assert f"- {track}" in workflow
    for model in ("topiq_nr", "clipiqa", "qrealign-mini"):
        assert f"- {model}" in workflow
    assert "--offline" in workflow
    assert "pip list --format=json" in workflow
    assert "dependency-inventory" in workflow
    assert "runs-on: windows-latest" in workflow
    assert "HF_HUB_DISABLE_SHARED_BLOBS" in (
        PROJECT_ROOT / "src" / "shotsieve" / "learned_iqa_runtime.py"
    ).read_text(encoding="utf-8")
