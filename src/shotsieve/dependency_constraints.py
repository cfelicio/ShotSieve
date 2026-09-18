"""Pinned learned-IQA dependency selections used by release and sidecar paths."""
from __future__ import annotations

import importlib.metadata

COMMON_MODEL_REQUIREMENTS = (
    "pyiqa==0.1.16",
    # PyIQA imports FaceRestoreHelper while registering TOPIQ, including for
    # the non-face topiq_nr model.
    "facexlib==0.3.0",
    "timm==1.0.29",
    "huggingface-hub==1.31.0",
    # Keep the earlier Q-ReAlign-compatible pin;
    # newer Transformers releases can expose the class while failing its
    # deferred module import.
    "transformers==5.14.1",
    "openai-clip==1.0.1",
    "accelerate==1.15.0",
    "sentencepiece==0.2.2",
    "einops==0.8.2",
)

TORCH_REQUIREMENTS = (
    "torch==2.14.0",
    "torchvision==0.29.0",
)

PYTORCH_CPU_INDEX_URL = "https://download.pytorch.org/whl/cpu"
PYTORCH_CUDA_INDEX_URL = "https://download.pytorch.org/whl/cu130"
PYTORCH_XPU_INDEX_URL = "https://download.pytorch.org/whl/xpu"

# Native Intel XPU wheels are intentionally separate from the CPU/CUDA/MPS
# release pair. They are consumed by the documented source and release tracks.
XPU_TORCH_REQUIREMENTS = (
    "torch==2.14.0+xpu",
    "torchvision==0.29.0+xpu",
)

# AMD's validated ROCm 7.2.1 Radeon wheels are a separate release track. They
# intentionally do not replace the common CPU/CUDA/MPS release pair.
ROCM_TORCH_REQUIREMENTS = (
    "torch==2.9.1+rocm7.2.1.lw.gitff65f5bc",
    "torchvision==0.24.0+rocm7.2.1.gitb919bd0c",
)

ROCM_WINDOWS_TORCH_REQUIREMENTS = (
    "torch==2.9.1+rocm7.2.1",
    "torchvision==0.24.1",
)

# These URLs are deliberately kept as exact, target-specific inputs.  In
# particular, XPU and ROCm must never silently resolve to the generic PyPI
# torch wheels.  The release workflow and the frozen sidecar installer share
# these values.
ROCM_LINUX_PACKAGE_URLS = (
    "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/torch-2.9.1%2Brocm7.2.1.lw.gitff65f5bc-cp312-cp312-linux_x86_64.whl",
    "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/torchvision-0.24.0%2Brocm7.2.1.gitb919bd0c-cp312-cp312-linux_x86_64.whl",
    "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/torchaudio-2.9.0%2Brocm7.2.1.gite3c6ee2b-cp312-cp312-linux_x86_64.whl",
    "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/triton-3.5.1%2Brocm7.2.1.gita272dfa8-cp312-cp312-linux_x86_64.whl",
)

ROCM_WINDOWS_PACKAGE_URLS = (
    "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_core-7.2.1-py3-none-win_amd64.whl",
    "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl",
    "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl",
    "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm-7.2.1.tar.gz",
    "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl",
    "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torchaudio-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl",
    "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torchvision-0.24.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl",
)

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
)


def model_requirements_for_runtime(
    runtime: str,
    *,
    platform_name: str | None = None,
) -> tuple[str, ...]:
    """Return the pinned model stack for a supported runtime family."""
    normalized_runtime = runtime.strip().casefold()
    if normalized_runtime == "xpu":
        torch_requirements = XPU_TORCH_REQUIREMENTS
    elif normalized_runtime in {"amd", "rocm"}:
        normalized_platform = (platform_name or "").strip().casefold()
        torch_requirements = (
            ROCM_WINDOWS_TORCH_REQUIREMENTS
            if normalized_platform in {"windows", "win32"}
            else ROCM_TORCH_REQUIREMENTS
        )
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
    "ROCM_LINUX_PACKAGE_URLS",
    "ROCM_TORCH_REQUIREMENTS",
    "ROCM_WINDOWS_PACKAGE_URLS",
    "ROCM_WINDOWS_TORCH_REQUIREMENTS",
    "TORCH_REQUIREMENTS",
    "XPU_TORCH_REQUIREMENTS",
    "installed_model_dependency_versions",
    "model_requirements_for_runtime",
]
