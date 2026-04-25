from __future__ import annotations

from dataclasses import dataclass
import importlib
import logging
import os
import platform
import threading
import warnings

from .learned_iqa_catalog import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DEVICE_POLICY,
    DEFAULT_MODEL_NAME,
    DEVICE_TARGET_ALIASES,
    MAX_BATCH_SIZES,
    MODEL_WEIGHT_MB,
    PER_IMAGE_ACTIVATION_MB,
    SUPPORTED_MODEL_NAMES,
    normalize_model_name,
    supported_learned_models,
    supported_runtime_targets,
)


TIMM_LAYERS_DEPRECATION_PATTERN = r"Importing from timm\.models\.layers is deprecated, please import via timm\.layers"
PKG_RESOURCES_DEPRECATION_PATTERN = r"pkg_resources is deprecated as an API"
TORCHSCRIPT_ARCHIVE_WARNING_PATTERN = r"'torch\.load' received a zip file that looks like a TorchScript archive"
TRANSFORMERS_GENERATION_FLAGS_WARNING_PATTERN = r"The following generation flags are not valid and may be ignored"
TRANSFORMERS_RETURN_DICT_DEPRECATION_PATTERN = r"`use_return_dict` is deprecated! Use `return_dict` instead!"
HF_UNAUTHENTICATED_REQUEST_WARNING_PATTERN = r"Warning: You are sending unauthenticated requests to the HF Hub"
DEFAULT_RUNTIME_STATUS_TEXT = "cuda:unavailable,xpu:unavailable,directml:not-installed,mps:unsupported,cpu:available"
RUNTIME_STATUS_ORDER = ("cuda", "xpu", "directml", "mps", "cpu")
RESOURCE_PROFILES = {
    "aggressive": {"vram_factor": 0.80, "cpu_factor": 2.0, "ram_factor": 0.75},
    "normal": {"vram_factor": 0.50, "cpu_factor": 1.0, "ram_factor": 0.50},
    "low": {"vram_factor": 0.30, "cpu_factor": 0.5, "ram_factor": 0.25},
}
DEFAULT_RESOURCE_PROFILE = "normal"
_ESTIMATED_WORKER_OVERHEAD_MB = 100
_RESERVED_RAM_MB = 1024


@dataclass(slots=True)
class ResolvedDevice:
    requested: str
    runtime: str
    metric_device: object
    tensor_device: object
    display_device: str


_cached_hw_capabilities: dict[str, object] | None = None
_hw_capabilities_lock = threading.Lock()


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


def current_system_name(system_name: str | None = None) -> str:
    return system_name or platform.system()


def normalize_device_target(device: str | None, *, system_name: str | None = None) -> str:
    if device is None:
        return "auto"

    requested = device.strip().casefold()
    normalized = DEVICE_TARGET_ALIASES.get(requested, requested)
    system = current_system_name(system_name)

    if normalized == "amd":
        return "directml" if system == "Windows" else "amd"

    if normalized == "apple":
        return "mps" if system == "Darwin" else "apple"

    return normalized


def auto_runtime_order(system_name: str | None = None) -> tuple[str, ...]:
    system = current_system_name(system_name)
    if system == "Darwin":
        return ("mps", "cpu")
    if system == "Linux":
        return ("cuda", "xpu", "cpu")
    return ("cuda", "xpu", "directml", "cpu")


def runtime_candidates(requested: str, *, system_name: str | None = None) -> tuple[str, ...]:
    system = current_system_name(system_name)
    if requested == "auto":
        return auto_runtime_order(system)
    if requested == "amd":
        return ("directml", "cpu") if system == "Windows" else ("cpu",)
    if requested == "apple":
        return ("mps", "cpu") if system == "Darwin" else ("cpu",)
    if requested == "intel":
        return ("xpu", "directml", "cpu") if system == "Windows" else ("xpu", "cpu")
    if requested in {"cuda", "xpu", "directml", "mps", "cpu"}:
        return (requested, "cpu") if requested != "cpu" else ("cpu",)
    return ("cpu",)


def has_cuda(torch_module) -> bool:
    try:
        cuda = getattr(torch_module, "cuda", None)
        return bool(cuda and cuda.is_available())
    except Exception:
        return False


def has_xpu(torch_module) -> bool:
    try:
        xpu = getattr(torch_module, "xpu", None)
        return bool(xpu and xpu.is_available())
    except Exception:
        return False


def has_mps(torch_module) -> bool:
    try:
        backends = getattr(torch_module, "backends", None)
        mps = getattr(backends, "mps", None)
        return bool(mps and mps.is_available())
    except Exception:
        return False


def load_directml_device(*, import_module=importlib.import_module) -> object | None:
    try:
        torch_directml = import_module("torch_directml")
    except Exception:
        return None
    try:
        return torch_directml.device(torch_directml.default_device())
    except Exception:
        return None


def resolve_device(device: str | None, *, torch_module, import_module=importlib.import_module, system_name: str | None = None) -> ResolvedDevice:
    system = current_system_name(system_name)
    requested = normalize_device_target(device, system_name=system)

    for runtime in runtime_candidates(requested, system_name=system):
        if runtime == "cuda" and has_cuda(torch_module):
            device_object = torch_module.device("cuda")
            return ResolvedDevice(requested=requested, runtime="cuda", metric_device=device_object, tensor_device=device_object, display_device="cuda")

        if runtime == "xpu" and has_xpu(torch_module):
            device_object = torch_module.device("xpu")
            return ResolvedDevice(requested=requested, runtime="xpu", metric_device=device_object, tensor_device=device_object, display_device="xpu")

        if runtime == "directml":
            device_object = load_directml_device(import_module=import_module)
            if device_object is not None:
                return ResolvedDevice(requested=requested, runtime="directml", metric_device=device_object, tensor_device=device_object, display_device="directml")

        if runtime == "mps" and has_mps(torch_module):
            device_object = torch_module.device("mps")
            return ResolvedDevice(requested=requested, runtime="mps", metric_device=device_object, tensor_device=device_object, display_device="mps")

        if runtime == "cpu":
            device_object = torch_module.device("cpu")
            return ResolvedDevice(requested=requested, runtime="cpu", metric_device=device_object, tensor_device=device_object, display_device="cpu")

    device_object = torch_module.device("cpu")
    return ResolvedDevice(requested=requested, runtime="cpu", metric_device=device_object, tensor_device=device_object, display_device="cpu")


def runtime_statuses(*, torch_module, import_module=importlib.import_module, system_name: str | None = None) -> dict[str, str]:
    system = current_system_name(system_name)
    statuses = {"cpu": "available"}
    statuses["cuda"] = "available" if has_cuda(torch_module) else "unavailable"
    statuses["xpu"] = "available" if has_xpu(torch_module) else ("unsupported" if system == "Darwin" else "unavailable")
    if system == "Windows":
        statuses["directml"] = "available" if load_directml_device(import_module=import_module) is not None else "not-installed"
    else:
        statuses["directml"] = "unsupported"
    if system == "Darwin":
        statuses["mps"] = "available" if has_mps(torch_module) else "unavailable"
    else:
        statuses["mps"] = "unsupported"
    return statuses


def _detect_vram_windows_registry() -> int | None:
    import json
    import subprocess

    try:
        result = subprocess.run(
            [
                "powershell", "-NoProfile", "-Command",
                (
                    "Get-ItemProperty"
                    " 'HKLM:\\SYSTEM\\CurrentControlSet\\Control\\Class"
                    "\\{4d36e968-e325-11ce-bfc1-08002be10318}\\0*'"
                    " -ErrorAction SilentlyContinue"
                    " | Where-Object { $_.'HardwareInformation.qwMemorySize' }"
                    " | Select-Object DriverDesc,"
                    " @{N='VRAM_MB';E={[math]::Round($_.'HardwareInformation.qwMemorySize' / 1MB)}}"
                    " | ConvertTo-Json"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None

        data = json.loads(result.stdout)
        if isinstance(data, dict):
            data = [data]

        best = 0
        for entry in data:
            vram = entry.get("VRAM_MB")
            if isinstance(vram, (int, float)) and vram > best:
                best = int(vram)
        return best if best > 0 else None
    except Exception:
        return None


def _detect_vram_linux_nvidia_smi() -> int | None:
    import subprocess

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None

        best = 0
        for line in result.stdout.strip().splitlines():
            try:
                vram = int(line.strip())
            except ValueError:
                continue
            if vram > best:
                best = vram
        return best if best > 0 else None
    except Exception:
        return None


def _detect_vram_linux_amd() -> int | None:
    vram = _detect_vram_linux_amd_sysfs()
    if vram is not None:
        return vram
    return _detect_vram_linux_rocm_smi()


def _detect_vram_linux_amd_sysfs() -> int | None:
    import glob

    try:
        paths = glob.glob("/sys/class/drm/card*/device/mem_info_vram_total")
        if not paths:
            return None

        best = 0
        for sysfs_path in paths:
            try:
                with open(sysfs_path) as handle:
                    vram_bytes = int(handle.read().strip())
            except (OSError, ValueError):
                continue
            vram_mb = vram_bytes // (1024 * 1024)
            if vram_mb > best:
                best = vram_mb
        return best if best > 0 else None
    except Exception:
        return None


def _detect_vram_linux_rocm_smi() -> int | None:
    import subprocess

    try:
        result = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--csv"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None

        best = 0
        for line in result.stdout.strip().splitlines()[1:]:
            parts = line.split(",")
            if len(parts) < 2:
                continue
            try:
                vram_bytes = int(parts[1].strip())
            except ValueError:
                continue
            vram_mb = vram_bytes // (1024 * 1024)
            if vram_mb > best:
                best = vram_mb
        return best if best > 0 else None
    except Exception:
        return None


def detect_system_ram_mb() -> int | None:
    try:
        if platform.system() == "Windows":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return int(stat.ullTotalPhys / (1024 * 1024))

        sysconf = getattr(os, "sysconf", None)
        if callable(sysconf):
            page_size = _coerce_vram_mb(sysconf("SC_PAGE_SIZE"))
            page_count = _coerce_vram_mb(sysconf("SC_PHYS_PAGES"))
            if page_size is None or page_count is None:
                return None
            if page_size > 0 and page_count > 0:
                return int((page_size * page_count) / (1024 * 1024))
    except Exception:
        return None
    return None


def _effective_cpu_count() -> int:
    host_cores = os.cpu_count() or 4

    if platform.system() != "Linux":
        return host_cores

    try:
        with open("/sys/fs/cgroup/cpu.max") as handle:
            parts = handle.read().strip().split()
        if len(parts) == 2 and parts[0] != "max":
            quota = int(parts[0])
            period = int(parts[1])
            if period > 0:
                return min(host_cores, max(1, quota // period))
    except (OSError, ValueError):
        pass

    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as handle:
            quota = int(handle.read().strip())
        if quota > 0:
            with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as handle:
                period = int(handle.read().strip())
            if period > 0:
                return min(host_cores, max(1, quota // period))
    except (OSError, ValueError):
        pass

    return host_cores


def detect_gpu_vram_mb(*, torch_module=None, import_module=importlib.import_module, import_pyiqa_runtime_fn=None, detect_system_ram_mb_fn=None) -> int | None:
    import_runtime = import_pyiqa_runtime_fn or import_pyiqa_runtime
    detect_ram = detect_system_ram_mb_fn or detect_system_ram_mb

    try:
        if torch_module is None:
            _, torch_module = import_runtime()
    except Exception:
        torch_module = None

    if torch_module is not None:
        if has_cuda(torch_module):
            cuda = getattr(torch_module, "cuda", None)
            get_device_properties = getattr(cuda, "get_device_properties", None)
            if callable(get_device_properties):
                try:
                    props = get_device_properties(0)
                    total = getattr(props, "total_memory", 0) or getattr(props, "total_mem", 0)
                    if total > 0:
                        return int(total / (1024 * 1024))
                except Exception:
                    pass

        if has_xpu(torch_module):
            xpu = getattr(torch_module, "xpu", None)
            get_device_properties = getattr(xpu, "get_device_properties", None)
            if callable(get_device_properties):
                try:
                    props = get_device_properties(0)
                    total = getattr(props, "total_memory", None) or getattr(props, "total_mem", None)
                    if total:
                        return int(total / (1024 * 1024))
                except Exception:
                    pass

        if has_mps(torch_module):
            ram_mb = detect_ram()
            if ram_mb:
                return int(ram_mb * 0.75)

    if platform.system() == "Windows":
        vram = _detect_vram_windows_registry()
        if vram is not None:
            return vram

    if platform.system() == "Linux":
        vram = _detect_vram_linux_nvidia_smi()
        if vram is not None:
            return vram
        vram = _detect_vram_linux_amd()
        if vram is not None:
            return vram

    return None


def invalidate_hw_cache() -> None:
    global _cached_hw_capabilities
    with _hw_capabilities_lock:
        _cached_hw_capabilities = None


def detect_hardware_capabilities(*, cpu_count_fn=None, detect_system_ram_mb_fn=None, detect_gpu_vram_mb_fn=None) -> dict[str, object]:
    global _cached_hw_capabilities
    effective_cpu_count_fn = cpu_count_fn or _effective_cpu_count
    detect_ram = detect_system_ram_mb_fn or detect_system_ram_mb
    detect_vram = detect_gpu_vram_mb_fn or detect_gpu_vram_mb

    with _hw_capabilities_lock:
        if _cached_hw_capabilities is not None:
            return _cached_hw_capabilities

        _cached_hw_capabilities = {
            "cpu_count": effective_cpu_count_fn(),
            "ram_mb": detect_ram(),
            "vram_mb": detect_vram(),
        }
        return _cached_hw_capabilities


def _valid_profile(profile: str | None) -> str:
    normalized = (profile or DEFAULT_RESOURCE_PROFILE).strip().lower()
    return normalized if normalized in RESOURCE_PROFILES else DEFAULT_RESOURCE_PROFILE


def recommended_cpu_workers(resource_profile: str | None = None, *, ram_mb: int | None = None, for_threads: bool = False, cpu_count_fn=None, detect_system_ram_mb_fn=None) -> int:
    effective_cpu_count_fn = cpu_count_fn or _effective_cpu_count
    detect_ram = detect_system_ram_mb_fn or detect_system_ram_mb
    profile = _valid_profile(resource_profile)
    cfg = RESOURCE_PROFILES[profile]
    cores = effective_cpu_count_fn()
    cpu_limit = max(2, int(cores * cfg["cpu_factor"]))

    if for_threads:
        return cpu_limit

    detected_ram = ram_mb if ram_mb is not None else detect_ram()
    if detected_ram is None or detected_ram <= 0:
        return min(cpu_limit, 16)

    available_for_workers = max(0, detected_ram - _RESERVED_RAM_MB)
    ram_budget = available_for_workers * cfg["ram_factor"]
    ram_limit = max(2, int(ram_budget / _ESTIMATED_WORKER_OVERHEAD_MB))
    return min(cpu_limit, ram_limit)


def recommended_batch_size(model_name: str, *, vram_mb: int | None = None, resource_profile: str | None = None) -> int:
    normalized_model = normalize_model_name(model_name)
    profile = _valid_profile(resource_profile)
    hard_max = MAX_BATCH_SIZES.get(normalized_model)

    if vram_mb is None or vram_mb <= 0:
        return hard_max if hard_max is not None else DEFAULT_BATCH_SIZE

    weight_mb = MODEL_WEIGHT_MB.get(normalized_model, 100)
    per_image_mb = PER_IMAGE_ACTIVATION_MB.get(normalized_model, 10)
    usable_vram = vram_mb * RESOURCE_PROFILES[profile]["vram_factor"]
    available_for_batch = max(0, usable_vram - weight_mb)
    computed = DEFAULT_BATCH_SIZE if per_image_mb <= 0 else max(1, int(available_for_batch / per_image_mb))
    if hard_max is not None:
        computed = min(computed, hard_max)
    return min(computed, 128)


def _runtime_status_text_from_torch_import(*, import_module=importlib.import_module, system_name: str | None = None) -> str:
    try:
        torch_module = import_module("torch")
    except Exception:
        return DEFAULT_RUNTIME_STATUS_TEXT

    try:
        statuses = runtime_statuses(torch_module=torch_module, import_module=import_module, system_name=system_name)
    except Exception:
        return DEFAULT_RUNTIME_STATUS_TEXT

    return ",".join(f"{runtime}:{statuses.get(runtime, 'unknown')}" for runtime in RUNTIME_STATUS_ORDER)


def unavailable_backend_payload(*, status: str, error: str | None = None, resource_profile: str | None = None, import_module=importlib.import_module, system_name: str | None = None) -> dict[str, object]:
    catalog = ",".join(supported_learned_models())
    runtime_targets = ",".join(supported_runtime_targets())
    auto_priority = ",".join(auto_runtime_order())
    vendor_aliases = "nvidia->cuda,amd->directml(windows),intel->xpu/directml,apple->mps"
    runtime_status_text = _runtime_status_text_from_torch_import(import_module=import_module, system_name=system_name)
    hw = detect_hardware_capabilities()
    vram_mb = _coerce_vram_mb(hw.get("vram_mb"))
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
        "hardware": hw,
        "recommended_batch_sizes": batch_recommendations,
        "resource_profile": profile,
    }
    if error:
        payload["pyiqa_error"] = error
    return payload


def configure_runtime_noise_controls() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    logging.getLogger("pyiqa").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("datasets").setLevel(logging.ERROR)


def ensure_pkg_resources_packaging_compat(*, import_module=importlib.import_module) -> None:
    with warnings.catch_warnings():
        install_runtime_warning_filters()
        try:
            pkg_resources = import_module("pkg_resources")
        except Exception:
            return

    if getattr(pkg_resources, "packaging", None) is not None:
        return

    try:
        packaging_module = import_module("packaging")
    except Exception:
        return

    try:
        setattr(pkg_resources, "packaging", packaging_module)
    except Exception:
        return


def install_runtime_warning_filters() -> None:
    warnings.filterwarnings("ignore", message=TIMM_LAYERS_DEPRECATION_PATTERN, category=FutureWarning)
    warnings.filterwarnings("ignore", message=PKG_RESOURCES_DEPRECATION_PATTERN, category=UserWarning)
    warnings.filterwarnings("ignore", message=TORCHSCRIPT_ARCHIVE_WARNING_PATTERN, category=UserWarning)
    warnings.filterwarnings("ignore", message=TRANSFORMERS_GENERATION_FLAGS_WARNING_PATTERN, category=UserWarning)
    warnings.filterwarnings("ignore", message=TRANSFORMERS_RETURN_DICT_DEPRECATION_PATTERN, category=UserWarning)
    warnings.filterwarnings("ignore", message=HF_UNAUTHENTICATED_REQUEST_WARNING_PATTERN, category=UserWarning)


def import_pyiqa_runtime(*, import_module=importlib.import_module):
    configure_runtime_noise_controls()
    with warnings.catch_warnings():
        install_runtime_warning_filters()
        ensure_pkg_resources_packaging_compat(import_module=import_module)
        pyiqa = import_module("pyiqa")
        torch = import_module("torch")

    return pyiqa, torch


__all__ = [
    "DEFAULT_RESOURCE_PROFILE",
    "DEFAULT_RUNTIME_STATUS_TEXT",
    "HF_UNAUTHENTICATED_REQUEST_WARNING_PATTERN",
    "PKG_RESOURCES_DEPRECATION_PATTERN",
    "RESOURCE_PROFILES",
    "RUNTIME_STATUS_ORDER",
    "ResolvedDevice",
    "TIMM_LAYERS_DEPRECATION_PATTERN",
    "TORCHSCRIPT_ARCHIVE_WARNING_PATTERN",
    "TRANSFORMERS_GENERATION_FLAGS_WARNING_PATTERN",
    "TRANSFORMERS_RETURN_DICT_DEPRECATION_PATTERN",
    "_cached_hw_capabilities",
    "_detect_vram_linux_amd",
    "_detect_vram_linux_amd_sysfs",
    "_detect_vram_linux_nvidia_smi",
    "_detect_vram_linux_rocm_smi",
    "_detect_vram_windows_registry",
    "_effective_cpu_count",
    "_runtime_status_text_from_torch_import",
    "_valid_profile",
    "auto_runtime_order",
    "configure_runtime_noise_controls",
    "current_system_name",
    "detect_gpu_vram_mb",
    "detect_hardware_capabilities",
    "detect_system_ram_mb",
    "ensure_pkg_resources_packaging_compat",
    "has_cuda",
    "has_mps",
    "has_xpu",
    "import_pyiqa_runtime",
    "install_runtime_warning_filters",
    "invalidate_hw_cache",
    "load_directml_device",
    "normalize_device_target",
    "recommended_batch_size",
    "recommended_cpu_workers",
    "resolve_device",
    "runtime_candidates",
    "runtime_statuses",
    "unavailable_backend_payload",
]
