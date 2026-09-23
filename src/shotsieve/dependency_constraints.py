"""Pinned learned-IQA dependency selections used by release and sidecar paths."""
from __future__ import annotations

import importlib.metadata

COMMON_MODEL_REQUIREMENTS = (
    # PyPI still classifies PyIQA as Alpha; keep its latest published release
    # because no stable alternative supplies all three supported IQA models.
    "pyiqa==0.1.16",
    # OpenAI CLIP imports pkg_resources; setuptools removed it in 82.0.0.
    "setuptools==81.0.0",
    # PyIQA imports FaceRestoreHelper while registering TOPIQ, including for
    # the non-face topiq_nr model.
    "facexlib==0.3.0",
    "timm==1.0.30",
    "huggingface-hub==1.32.0",
    "transformers==5.17.0",
    "openai-clip==1.0.1",
    "accelerate==1.15.0",
    "sentencepiece==0.2.2",
    "einops==0.8.2",
    "icecream==2.2.0",
)

TORCH_REQUIREMENTS = (
    "torch==2.14.0",
    "torchvision==0.29.0",
)

PYTORCH_CPU_INDEX_URL = "https://download.pytorch.org/whl/cpu"
PYTORCH_CUDA_INDEX_URL = "https://download.pytorch.org/whl/cu130"
PYTORCH_XPU_INDEX_URL = "https://download.pytorch.org/whl/xpu"

# Intel's official XPU wheels are separate from the CPU/CUDA/MPS release pair.
XPU_TORCH_REQUIREMENTS = (
    "torch==2.14.0+xpu",
    "torchvision==0.29.0+xpu",
)

# AMD's stable ROCm 10.0 multi-architecture index publishes the 2.13/0.28
# pair for gfx1103 on both Windows and Linux, including CPython 3.14 wheels.
ROCM_TORCH_REQUIREMENTS = (
    "torch[device-gfx1103]==2.13.0+rocm10.0.0",
    "torchvision[device-gfx1103]==0.28.0+rocm10.0.0",
    "rocm==10.0.0",
)
ROCM_SELECTOR_REQUIREMENT = "rocm==10.0.0"
ROCM_PYTHON_INDEX_URL = "https://stable.repo.amd.com/rocm/whl-next/"

MODEL_DEPENDENCY_DISTRIBUTIONS = (
    "pyiqa",
    "facexlib",
    "torch",
    "torchvision",
    "timm",
    "huggingface-hub",
    "transformers",
    "openai-clip",
    "accelerate",
    "sentencepiece",
    "einops",
    "icecream",
    "setuptools",
)


def model_requirements_for_runtime(
    runtime: str,
) -> tuple[str, ...]:
    """Return the pinned model stack for a supported runtime family."""
    normalized_runtime = runtime.strip().casefold()
    if normalized_runtime == "xpu":
        torch_requirements = XPU_TORCH_REQUIREMENTS
    elif normalized_runtime in {"amd", "rocm"}:
        torch_requirements = ROCM_TORCH_REQUIREMENTS
    else:
        torch_requirements = TORCH_REQUIREMENTS
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
    "PYTORCH_CPU_INDEX_URL",
    "PYTORCH_CUDA_INDEX_URL",
    "PYTORCH_XPU_INDEX_URL",
    "ROCM_PYTHON_INDEX_URL",
    "ROCM_SELECTOR_REQUIREMENT",
    "ROCM_TORCH_REQUIREMENTS",
    "TORCH_REQUIREMENTS",
    "XPU_TORCH_REQUIREMENTS",
    "installed_model_dependency_versions",
    "model_requirements_for_runtime",
]
