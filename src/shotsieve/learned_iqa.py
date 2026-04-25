from __future__ import annotations

import importlib
import logging
from pathlib import Path
import threading
from typing import Sequence, cast

from . import learned_iqa_backend as backend_core
from . import learned_iqa_catalog
from . import learned_iqa_preprocessing
from . import learned_iqa_runtime


DEFAULT_BATCH_SIZE = learned_iqa_catalog.DEFAULT_BATCH_SIZE
DEFAULT_MODEL_NAME = learned_iqa_catalog.DEFAULT_MODEL_NAME
DEFAULT_INPUT_SIZE = learned_iqa_catalog.DEFAULT_INPUT_SIZE
DEFAULT_DEVICE_POLICY = learned_iqa_catalog.DEFAULT_DEVICE_POLICY
DEFAULT_INPUT_SIZES = learned_iqa_catalog.DEFAULT_INPUT_SIZES
MAX_BATCH_SIZES = learned_iqa_catalog.MAX_BATCH_SIZES
MODEL_WEIGHT_MB = learned_iqa_catalog.MODEL_WEIGHT_MB
PER_IMAGE_ACTIVATION_MB = learned_iqa_catalog.PER_IMAGE_ACTIVATION_MB
DEVICE_TARGET_ALIASES = learned_iqa_catalog.DEVICE_TARGET_ALIASES
MODEL_NAME_ALIASES = learned_iqa_catalog.MODEL_NAME_ALIASES
SUPPORTED_MODEL_NAMES = learned_iqa_catalog.SUPPORTED_MODEL_NAMES
MODERN_MODEL_NAMES = learned_iqa_catalog.MODERN_MODEL_NAMES
UI_MODEL_CATALOG = learned_iqa_catalog.UI_MODEL_CATALOG

TIMM_LAYERS_DEPRECATION_PATTERN = learned_iqa_runtime.TIMM_LAYERS_DEPRECATION_PATTERN
PKG_RESOURCES_DEPRECATION_PATTERN = learned_iqa_runtime.PKG_RESOURCES_DEPRECATION_PATTERN
TORCHSCRIPT_ARCHIVE_WARNING_PATTERN = learned_iqa_runtime.TORCHSCRIPT_ARCHIVE_WARNING_PATTERN
TRANSFORMERS_GENERATION_FLAGS_WARNING_PATTERN = learned_iqa_runtime.TRANSFORMERS_GENERATION_FLAGS_WARNING_PATTERN
TRANSFORMERS_RETURN_DICT_DEPRECATION_PATTERN = learned_iqa_runtime.TRANSFORMERS_RETURN_DICT_DEPRECATION_PATTERN
HF_UNAUTHENTICATED_REQUEST_WARNING_PATTERN = learned_iqa_runtime.HF_UNAUTHENTICATED_REQUEST_WARNING_PATTERN
DEFAULT_RUNTIME_STATUS_TEXT = learned_iqa_runtime.DEFAULT_RUNTIME_STATUS_TEXT
RUNTIME_STATUS_ORDER = learned_iqa_runtime.RUNTIME_STATUS_ORDER
RESOURCE_PROFILES = learned_iqa_runtime.RESOURCE_PROFILES
DEFAULT_RESOURCE_PROFILE = learned_iqa_runtime.DEFAULT_RESOURCE_PROFILE

ResolvedDevice = learned_iqa_runtime.ResolvedDevice
LearnedBackendUnavailableError = backend_core.LearnedBackendUnavailableError
LearnedScoreResult = backend_core.LearnedScoreResult
LearnedIqaBackend = backend_core.LearnedIqaBackend

normalize_model_name = learned_iqa_catalog.normalize_model_name
supported_learned_models = learned_iqa_catalog.supported_learned_models
supported_runtime_targets = learned_iqa_catalog.supported_runtime_targets
preferred_model_names = learned_iqa_catalog.preferred_model_names
is_model_runtime_compatible = learned_iqa_catalog.is_model_runtime_compatible
runtime_compatible_model_names = learned_iqa_catalog.runtime_compatible_model_names

current_system_name = learned_iqa_runtime.current_system_name
normalize_device_target = learned_iqa_runtime.normalize_device_target
auto_runtime_order = learned_iqa_runtime.auto_runtime_order
runtime_candidates = learned_iqa_runtime.runtime_candidates
has_cuda = learned_iqa_runtime.has_cuda
has_xpu = learned_iqa_runtime.has_xpu
has_mps = learned_iqa_runtime.has_mps
load_directml_device = learned_iqa_runtime.load_directml_device
resolve_device = learned_iqa_runtime.resolve_device
runtime_statuses = learned_iqa_runtime.runtime_statuses

_detect_vram_windows_registry = learned_iqa_runtime._detect_vram_windows_registry
_detect_vram_linux_nvidia_smi = learned_iqa_runtime._detect_vram_linux_nvidia_smi
_detect_vram_linux_amd = learned_iqa_runtime._detect_vram_linux_amd
_detect_vram_linux_amd_sysfs = learned_iqa_runtime._detect_vram_linux_amd_sysfs
_detect_vram_linux_rocm_smi = learned_iqa_runtime._detect_vram_linux_rocm_smi
detect_system_ram_mb = learned_iqa_runtime.detect_system_ram_mb
_effective_cpu_count = learned_iqa_runtime._effective_cpu_count
_valid_profile = learned_iqa_runtime._valid_profile

configure_runtime_noise_controls = learned_iqa_runtime.configure_runtime_noise_controls
ensure_pkg_resources_packaging_compat = learned_iqa_runtime.ensure_pkg_resources_packaging_compat
install_runtime_warning_filters = learned_iqa_runtime.install_runtime_warning_filters

_load_single_image = learned_iqa_preprocessing._load_single_image
_arrays_to_tensor = learned_iqa_preprocessing._arrays_to_tensor
load_batch_tensor = learned_iqa_preprocessing.load_batch_tensor
flatten_tensor = learned_iqa_preprocessing.flatten_tensor
confidence_values = learned_iqa_preprocessing.confidence_values
normalize_score = learned_iqa_preprocessing.normalize_score
parse_score_range = learned_iqa_preprocessing.parse_score_range

_bypass_torch_load_cve_check = backend_core._bypass_torch_load_cve_check


log = logging.getLogger(__name__)
_cached_hw_capabilities: dict[str, object] | None = None
_hw_capabilities_lock = threading.Lock()


def build_learned_backend(model_name: str, *, device: str | None = None) -> LearnedIqaBackend:
    return cast(LearnedIqaBackend, backend_core.build_learned_backend(
        model_name,
        device=device,
        backend_cls=PyiqaBackend,
        normalize_model_name_fn=normalize_model_name,
    ))


def _coerce_vram_mb(value: object) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return int(stripped)
        except ValueError:
            return None
    return None


def release_learned_backend(backend: object) -> None:
    backend_core.release_learned_backend(backend)


def import_pyiqa_runtime():
    return learned_iqa_runtime.import_pyiqa_runtime()


def runtime_supported_learned_models() -> tuple[str, ...]:
    try:
        pyiqa, torch = import_pyiqa_runtime()
        discovered_models = [normalize_model_name(name) for name in pyiqa.list_models(metric_mode="NR")]
        resolved_runtime = resolve_device(None, torch_module=torch).runtime
        compatible_models = set(
            runtime_compatible_model_names(
                discovered_models,
                torch_version=getattr(torch, "__version__", None),
                runtime=resolved_runtime,
            )
        )
        filtered = tuple(model for model in SUPPORTED_MODEL_NAMES if model in compatible_models)
        return filtered or SUPPORTED_MODEL_NAMES
    except Exception:
        return SUPPORTED_MODEL_NAMES


def runtime_curated_learned_models() -> tuple[str, ...]:
    available_models = set(runtime_supported_learned_models())
    curated = tuple(model for model in UI_MODEL_CATALOG if model in available_models)
    return curated or UI_MODEL_CATALOG


def detect_gpu_vram_mb(*, torch_module=None, import_module=importlib.import_module) -> int | None:
    return learned_iqa_runtime.detect_gpu_vram_mb(
        torch_module=torch_module,
        import_module=import_module,
        import_pyiqa_runtime_fn=import_pyiqa_runtime,
        detect_system_ram_mb_fn=detect_system_ram_mb,
    )


def invalidate_hw_cache() -> None:
    global _cached_hw_capabilities
    with _hw_capabilities_lock:
        _cached_hw_capabilities = None


def detect_hardware_capabilities() -> dict[str, object]:
    global _cached_hw_capabilities
    with _hw_capabilities_lock:
        if _cached_hw_capabilities is not None:
            return _cached_hw_capabilities

        _cached_hw_capabilities = {
            "cpu_count": _effective_cpu_count(),
            "ram_mb": detect_system_ram_mb(),
            "vram_mb": detect_gpu_vram_mb(),
        }
        return _cached_hw_capabilities


def recommended_cpu_workers(
    resource_profile: str | None = None,
    *,
    ram_mb: int | None = None,
    for_threads: bool = False,
) -> int:
    return learned_iqa_runtime.recommended_cpu_workers(
        resource_profile,
        ram_mb=ram_mb,
        for_threads=for_threads,
        cpu_count_fn=_effective_cpu_count,
        detect_system_ram_mb_fn=detect_system_ram_mb,
    )


def recommended_batch_size(model_name: str, *, vram_mb: int | None = None, resource_profile: str | None = None) -> int:
    return learned_iqa_runtime.recommended_batch_size(
        model_name,
        vram_mb=vram_mb,
        resource_profile=resource_profile,
    )


def _runtime_status_text_from_torch_import(
    *,
    import_module=importlib.import_module,
    system_name: str | None = None,
) -> str:
    try:
        torch_module = import_module("torch")
    except Exception:
        return DEFAULT_RUNTIME_STATUS_TEXT

    try:
        statuses = runtime_statuses(
            torch_module=torch_module,
            import_module=import_module,
            system_name=system_name,
        )
    except Exception:
        return DEFAULT_RUNTIME_STATUS_TEXT

    return ",".join(f"{runtime}:{statuses.get(runtime, 'unknown')}" for runtime in RUNTIME_STATUS_ORDER)


def unavailable_backend_payload(
    *,
    status: str,
    error: str | None = None,
    resource_profile: str | None = None,
    import_module=importlib.import_module,
    system_name: str | None = None,
) -> dict[str, object]:
    catalog = ",".join(supported_learned_models())
    runtime_targets = ",".join(supported_runtime_targets())
    auto_priority = ",".join(auto_runtime_order())
    vendor_aliases = "nvidia->cuda,amd->directml(windows),intel->xpu/directml,apple->mps"
    runtime_status_text = _runtime_status_text_from_torch_import(import_module=import_module, system_name=system_name)
    hardware = detect_hardware_capabilities()
    vram_mb = _coerce_vram_mb(hardware.get("vram_mb"))
    profile = _valid_profile(resource_profile)
    batch_recommendations = {
        model_name: recommended_batch_size(model_name, vram_mb=vram_mb, resource_profile=profile)
        for model_name in SUPPORTED_MODEL_NAMES
    }

    payload = {
        "pyiqa": status,
        "default_model": DEFAULT_MODEL_NAME,
        "device_policy": DEFAULT_DEVICE_POLICY,
        "default_device": "cpu",
        "default_runtime": "cpu",
        "runtime_targets": runtime_targets,
        "runtime_status": runtime_status_text,
        "auto_runtime_priority": auto_priority,
        "vendor_aliases": vendor_aliases,
        "modern_model_catalog": catalog,
        "modern_models_available": "",
        "hardware": hardware,
        "recommended_batch_sizes": batch_recommendations,
        "resource_profile": profile,
    }
    if error:
        payload["pyiqa_error"] = error
    return payload


def create_metric_safely(pyiqa_module, model_name: str, *, device):
    return backend_core.create_metric_safely(
        pyiqa_module,
        model_name,
        device=device,
        configure_runtime_noise_controls_fn=configure_runtime_noise_controls,
        install_runtime_warning_filters_fn=install_runtime_warning_filters,
    )


def resolve_learned_model_version(model_name: str, *, device: str | None = None) -> str:
    return backend_core.resolve_learned_model_version(
        model_name,
        device=device,
        import_pyiqa_runtime_fn=import_pyiqa_runtime,
        normalize_model_name_fn=normalize_model_name,
        preferred_model_names_fn=preferred_model_names,
        resolve_device_fn=resolve_device,
    )


class PyiqaBackend:
    name: str
    model_version: str
    runtime: str
    device: str
    tensor_device: object
    input_size: int
    lower_better: bool
    score_range: str
    metric: object
    _pyiqa: object
    _torch: object

    def __init__(self, model_name: str, *, device: str | None = None) -> None:
        backend_core.initialize_backend(
            self,
            model_name,
            device=device,
            import_pyiqa_runtime_fn=import_pyiqa_runtime,
            normalize_model_name_fn=normalize_model_name,
            preferred_model_names_fn=preferred_model_names,
            resolve_device_fn=resolve_device,
            create_metric_safely_fn=create_metric_safely,
            default_input_sizes=DEFAULT_INPUT_SIZES,
            default_input_size=DEFAULT_INPUT_SIZE,
        )

    def close(self) -> None:
        backend_core.close_backend(self)

    def score_paths(self, image_paths: Sequence[Path], *, batch_size: int = DEFAULT_BATCH_SIZE, resource_profile: str | None = None) -> list[LearnedScoreResult]:
        return backend_core.score_paths(
            self,
            image_paths,
            batch_size=batch_size,
            resource_profile=resource_profile,
            recommended_cpu_workers_fn=recommended_cpu_workers,
            max_batch_sizes=MAX_BATCH_SIZES,
            load_batch_tensor_fn=load_batch_tensor,
            arrays_to_tensor_fn=_arrays_to_tensor,
            load_single_image_fn=_load_single_image,
            result_cls=LearnedScoreResult,
            log_module=log,
        )

    def _score_tensor_batch(self, batch_tensor):
        return backend_core.score_tensor_batch(
            self,
            batch_tensor,
            flatten_tensor_fn=flatten_tensor,
            confidence_values_fn=confidence_values,
            normalize_score_fn=normalize_score,
            result_cls=LearnedScoreResult,
        )


def available_learned_backends(*, resource_profile: str | None = None) -> dict[str, object]:
    return backend_core.available_learned_backends(
        resource_profile=resource_profile,
        import_pyiqa_runtime_fn=import_pyiqa_runtime,
        unavailable_backend_payload_fn=unavailable_backend_payload,
        resolve_device_fn=resolve_device,
        runtime_statuses_fn=runtime_statuses,
        runtime_compatible_model_names_fn=runtime_compatible_model_names,
        preferred_model_names_fn=preferred_model_names,
        detect_hardware_capabilities_fn=detect_hardware_capabilities,
        valid_profile_fn=_valid_profile,
        recommended_batch_size_fn=recommended_batch_size,
        supported_model_names=SUPPORTED_MODEL_NAMES,
        default_model_name=DEFAULT_MODEL_NAME,
        default_device_policy=DEFAULT_DEVICE_POLICY,
        supported_runtime_targets_fn=supported_runtime_targets,
        auto_runtime_order_fn=auto_runtime_order,
        runtime_status_order=RUNTIME_STATUS_ORDER,
    )


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_DEVICE_POLICY",
    "DEFAULT_INPUT_SIZE",
    "DEFAULT_INPUT_SIZES",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_RESOURCE_PROFILE",
    "DEFAULT_RUNTIME_STATUS_TEXT",
    "DEVICE_TARGET_ALIASES",
    "HF_UNAUTHENTICATED_REQUEST_WARNING_PATTERN",
    "LearnedBackendUnavailableError",
    "LearnedIqaBackend",
    "LearnedScoreResult",
    "MAX_BATCH_SIZES",
    "MODEL_NAME_ALIASES",
    "MODEL_WEIGHT_MB",
    "MODERN_MODEL_NAMES",
    "PER_IMAGE_ACTIVATION_MB",
    "PKG_RESOURCES_DEPRECATION_PATTERN",
    "PyiqaBackend",
    "RESOURCE_PROFILES",
    "RUNTIME_STATUS_ORDER",
    "ResolvedDevice",
    "SUPPORTED_MODEL_NAMES",
    "TIMM_LAYERS_DEPRECATION_PATTERN",
    "TORCHSCRIPT_ARCHIVE_WARNING_PATTERN",
    "TRANSFORMERS_GENERATION_FLAGS_WARNING_PATTERN",
    "TRANSFORMERS_RETURN_DICT_DEPRECATION_PATTERN",
    "UI_MODEL_CATALOG",
    "_arrays_to_tensor",
    "_bypass_torch_load_cve_check",
    "_cached_hw_capabilities",
    "_detect_vram_linux_amd",
    "_detect_vram_linux_amd_sysfs",
    "_detect_vram_linux_nvidia_smi",
    "_detect_vram_linux_rocm_smi",
    "_detect_vram_windows_registry",
    "_effective_cpu_count",
    "_load_single_image",
    "_runtime_status_text_from_torch_import",
    "_valid_profile",
    "auto_runtime_order",
    "available_learned_backends",
    "build_learned_backend",
    "confidence_values",
    "configure_runtime_noise_controls",
    "create_metric_safely",
    "current_system_name",
    "detect_gpu_vram_mb",
    "detect_hardware_capabilities",
    "detect_system_ram_mb",
    "ensure_pkg_resources_packaging_compat",
    "flatten_tensor",
    "has_cuda",
    "has_mps",
    "has_xpu",
    "import_pyiqa_runtime",
    "install_runtime_warning_filters",
    "invalidate_hw_cache",
    "is_model_runtime_compatible",
    "load_batch_tensor",
    "load_directml_device",
    "normalize_device_target",
    "normalize_model_name",
    "normalize_score",
    "parse_score_range",
    "preferred_model_names",
    "recommended_batch_size",
    "recommended_cpu_workers",
    "release_learned_backend",
    "resolve_device",
    "resolve_learned_model_version",
    "runtime_candidates",
    "runtime_compatible_model_names",
    "runtime_curated_learned_models",
    "runtime_statuses",
    "runtime_supported_learned_models",
    "supported_learned_models",
    "supported_runtime_targets",
    "unavailable_backend_payload",
]
