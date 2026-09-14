"""Pinned learned-IQA dependency selections used by release and sidecar paths."""
from __future__ import annotations

import importlib.metadata

COMMON_MODEL_REQUIREMENTS = (
    "pyiqa==0.1.16",
    "timm==1.0.29",
    "huggingface-hub==1.31.0",
    "transformers==5.17.0",
    "openai-clip==1.0.1",
    "accelerate==1.15.0",
    "sentencepiece==0.2.2",
    "einops==0.8.2",
)

TORCH_REQUIREMENTS = (
    "torch==2.14.0",
    "torchvision==0.29.0",
)

# Native Intel XPU wheels are intentionally separate from the CPU/CUDA/MPS
# release pair. They are consumed by the documented source-install track only;
# no XPU runtime pack or sidecar is built from these requirements.
XPU_TORCH_REQUIREMENTS = (
    "torch==2.14.0+xpu",
    "torchvision==0.29.0+xpu",
)

MODEL_DEPENDENCY_DISTRIBUTIONS = (
    "pyiqa",
    "torch",
    "torchvision",
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
    torch_requirements = XPU_TORCH_REQUIREMENTS if runtime.strip().casefold() == "xpu" else TORCH_REQUIREMENTS
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
    "MODEL_DEPENDENCY_DISTRIBUTIONS",
    "TORCH_REQUIREMENTS",
    "XPU_TORCH_REQUIREMENTS",
    "installed_model_dependency_versions",
    "model_requirements_for_runtime",
]
