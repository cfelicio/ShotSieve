"""Pinned learned-IQA dependency selections used by release and sidecar paths."""
from __future__ import annotations

import importlib.metadata

COMMON_MODEL_REQUIREMENTS = (
    "pyiqa==0.1.16",
    "timm==1.0.28",
    "huggingface-hub==1.24.0",
    "transformers==5.14.1",
    "openai-clip==1.0.1",
    "accelerate==1.14.0",
    "sentencepiece==0.2.2",
    "einops==0.8.2",
)

NON_DIRECTML_TORCH_REQUIREMENTS = (
    "torch==2.13.0",
    "torchvision==0.28.0",
)

DIRECTML_TORCH_REQUIREMENTS = (
    "torch==2.4.1",
    "torchvision==0.19.1",
    "torch-directml==0.2.5.dev240914",
)

MODEL_DEPENDENCY_DISTRIBUTIONS = (
    "pyiqa",
    "torch",
    "torchvision",
    "torch-directml",
    "timm",
    "huggingface-hub",
    "transformers",
    "openai-clip",
    "accelerate",
    "sentencepiece",
    "einops",
)


def model_requirements_for_runtime(runtime: str) -> tuple[str, ...]:
    """Return the pinned model stack for a supported runtime family."""
    normalized = runtime.strip().casefold()
    torch_requirements = DIRECTML_TORCH_REQUIREMENTS if normalized == "directml" else NON_DIRECTML_TORCH_REQUIREMENTS
    return COMMON_MODEL_REQUIREMENTS + torch_requirements


def installed_model_dependency_versions() -> dict[str, str]:
    """Report resolved model-stack versions without importing optional runtimes."""
    versions: dict[str, str] = {}
    for distribution_name in MODEL_DEPENDENCY_DISTRIBUTIONS:
        try:
            versions[distribution_name] = importlib.metadata.version(distribution_name)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution_name] = "not-installed"
        except Exception:
            versions[distribution_name] = "unknown"
    return versions


__all__ = [
    "COMMON_MODEL_REQUIREMENTS",
    "DIRECTML_TORCH_REQUIREMENTS",
    "MODEL_DEPENDENCY_DISTRIBUTIONS",
    "NON_DIRECTML_TORCH_REQUIREMENTS",
    "installed_model_dependency_versions",
    "model_requirements_for_runtime",
]
