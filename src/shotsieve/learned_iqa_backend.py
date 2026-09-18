from __future__ import annotations

from contextlib import AbstractContextManager, ExitStack, contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
import gc
import io
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Callable, Protocol, Sequence, cast
import warnings

from .learned_iqa_catalog import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_INPUT_SIZE,
    DEFAULT_INPUT_SIZES,
    MAX_BATCH_SIZES,
    MODEL_CATALOG,
    is_model_runtime_compatible,
    is_supported_model_name,
    validate_model_name,
)
from .learned_iqa_runtime import LearnedRuntimeUnavailableError
from .image_conversion import DEFAULT_MAX_DECODE_PIXELS


log = logging.getLogger(__name__)

AUTOCAST_ENABLED_RUNTIMES = frozenset({"cuda", "mps"})
AUTOCAST_BLOCKED_MODELS_BY_RUNTIME: dict[str, frozenset[str]] = {
    "cuda": frozenset(),
    "mps": frozenset(),
}


class LearnedBackendUnavailableError(RuntimeError):
    pass


@dataclass(slots=True)
class LearnedScoreResult:
    raw_score: float | None
    normalized_score: float | None
    confidence: float | None = None
    error: str | None = None

    @property
    def failed(self) -> bool:
        return self.raw_score is None or self.normalized_score is None or bool(self.error)


class LearnedIqaBackend(Protocol):
    name: str
    model_version: str

    def score_paths(
        self,
        image_paths: Sequence[Path],
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
        resource_profile: str | None = None,
        max_decode_pixels: int | None = None,
    ) -> list[LearnedScoreResult]:
        ...


class _GcModuleLike(Protocol):
    def collect(self) -> object:
        ...


_stdio_capture_lock = threading.Lock()


def _validate_product_model(model_name: str, *, normalize_model_name_fn) -> str:
    canonical = normalize_model_name_fn(model_name)
    if not is_supported_model_name(canonical):
        try:
            validate_model_name(canonical)
        except ValueError as exc:
            raise LearnedBackendUnavailableError(str(exc)) from exc
        raise LearnedBackendUnavailableError(f"Learned IQA model '{canonical}' is not supported for new runs.")
    return canonical


def build_learned_backend(model_name: str, *, device: str | None = None, backend_cls, normalize_model_name_fn):
    canonical_model_name = _validate_product_model(model_name, normalize_model_name_fn=normalize_model_name_fn)
    return backend_cls(canonical_model_name, device=device)


def release_learned_backend(backend: object) -> None:
    close_method = getattr(backend, "close", None)
    if callable(close_method):
        try:
            close_method()
        except Exception:
            return


def _exception_chain_text(exc: BaseException) -> str:
    """Retain the real lazy-import cause behind generic backend exceptions."""
    parts: list[str] = []
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen and len(parts) < 4:
        seen.add(id(current))
        text = f"{type(current).__name__}: {current}"
        if text not in parts:
            parts.append(text)
        current = current.__cause__ or current.__context__
    return " | caused by ".join(parts)


def create_metric_safely(pyiqa_module, model_name: str, *, device, configure_runtime_noise_controls_fn, install_runtime_warning_filters_fn):
    configure_runtime_noise_controls_fn()
    metric_options = {}
    spec = next((spec for spec in MODEL_CATALOG if spec.canonical_id == model_name), None)
    if spec is not None and spec.upstream_model_id and spec.checkpoint_revision:
        from huggingface_hub import snapshot_download

        # PyIQA's Q-ReAlign constructor accepts a local model path, but does
        # not forward a revision to its processor and model loaders.
        metric_options["model"] = snapshot_download(
            repo_id=spec.upstream_model_id,
            revision=spec.checkpoint_revision,
            cache_dir=os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE"),
            local_files_only=any(
                os.environ.get(name, "").strip().casefold() in {"1", "true", "yes", "on"}
                for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
            ),
        )
    with warnings.catch_warnings():
        install_runtime_warning_filters_fn()
        capture_stdio = threading.active_count() == 1
        if not capture_stdio:
            return pyiqa_module.create_metric(model_name, device=device, **metric_options)

        with _stdio_capture_lock:
            with io.StringIO() as stdout_buffer, io.StringIO() as stderr_buffer:
                try:
                    with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                        return pyiqa_module.create_metric(model_name, device=device, **metric_options)
                except Exception:
                    captured_stdout = stdout_buffer.getvalue().strip()
                    captured_stderr = stderr_buffer.getvalue().strip()
                    if captured_stdout:
                        print(captured_stdout, file=sys.stdout)
                    if captured_stderr:
                        print(captured_stderr, file=sys.stderr)
                    raise


def _ensure_model_runtime_compatible(model_name: str, *, runtime: str, torch_version: str | None) -> None:
    if is_model_runtime_compatible(model_name, torch_version=torch_version, runtime=runtime):
        return

    raise LearnedBackendUnavailableError(
        f"Learned IQA model '{model_name}' is not compatible with runtime '{runtime}'. "
        "Choose a supported accelerator runtime or a different model."
    )


def _model_version(pyiqa_module, model_name: str, runtime: str) -> str:
    version = f"pyiqa:{getattr(pyiqa_module, '__version__', 'unknown')}:{model_name}:{runtime}"
    spec = next(spec for spec in MODEL_CATALOG if spec.canonical_id == model_name)
    if spec.checkpoint_revision:
        version += f":{spec.checkpoint_revision}"
    return version


def resolve_learned_model_version(model_name: str, *, device: str | None = None, import_pyiqa_runtime_fn, normalize_model_name_fn, preferred_model_names_fn, resolve_device_fn) -> str:
    canonical_model_name = _validate_product_model(model_name, normalize_model_name_fn=normalize_model_name_fn)

    try:
        pyiqa, torch = import_pyiqa_runtime_fn()
    except ImportError as exc:
        raise LearnedBackendUnavailableError(
            "Learned IQA requires optional dependencies. Install shotsieve[learned-iqa]."
        ) from exc
    except Exception as exc:
        raise LearnedBackendUnavailableError(
            f"Learned IQA runtime failed to initialize: {exc}"
        ) from exc

    try:
        available_models = {
            normalize_model_name_fn(name)
            for name in pyiqa.list_models(metric_mode="NR")
        }
    except Exception as exc:
        raise LearnedBackendUnavailableError(
            f"Learned IQA runtime is installed but unavailable: {exc}"
        ) from exc

    if canonical_model_name not in available_models:
        preferred = preferred_model_names_fn(available_models)
        catalog_text = ", ".join(preferred) if preferred else ", ".join(sorted(available_models)[:12])
        raise LearnedBackendUnavailableError(
            f"Learned IQA model '{canonical_model_name}' is not available in the installed pyiqa runtime. "
            f"Available NR models include: {catalog_text}"
        )

    try:
        resolved_device = resolve_device_fn(device, torch_module=torch)
    except LearnedRuntimeUnavailableError as exc:
        raise LearnedBackendUnavailableError(str(exc)) from exc
    _ensure_model_runtime_compatible(
        canonical_model_name,
        runtime=resolved_device.runtime,
        torch_version=getattr(torch, "__version__", None),
    )
    return _model_version(pyiqa, canonical_model_name, resolved_device.runtime)


def _restore_cudnn_benchmark(backend) -> None:
    torch_module = getattr(backend, "_torch", None)
    if torch_module is None or not hasattr(backend, "_previous_cudnn_benchmark"):
        return

    cudnn_backend = getattr(getattr(torch_module, "backends", None), "cudnn", None)
    if cudnn_backend is not None and hasattr(cudnn_backend, "benchmark"):
        setattr(cudnn_backend, "benchmark", backend._previous_cudnn_benchmark)
    delattr(backend, "_previous_cudnn_benchmark")


def initialize_backend(backend, model_name: str, *, device: str | None = None, import_pyiqa_runtime_fn, normalize_model_name_fn, preferred_model_names_fn, resolve_device_fn, create_metric_safely_fn, default_input_sizes=None, default_input_size: int = DEFAULT_INPUT_SIZE) -> None:
    input_sizes = default_input_sizes or DEFAULT_INPUT_SIZES
    canonical_model_name = _validate_product_model(model_name, normalize_model_name_fn=normalize_model_name_fn)

    try:
        pyiqa, torch = import_pyiqa_runtime_fn()
    except ImportError as exc:
        raise LearnedBackendUnavailableError(
            "Learned IQA requires optional dependencies. Install shotsieve[learned-iqa]."
        ) from exc
    except Exception as exc:
        raise LearnedBackendUnavailableError(
            f"Learned IQA runtime failed to initialize: {exc}"
        ) from exc

    try:
        available_models = {
            normalize_model_name_fn(name)
            for name in pyiqa.list_models(metric_mode="NR")
        }
    except Exception as exc:
        raise LearnedBackendUnavailableError(
            f"Learned IQA runtime is installed but unavailable: {exc}"
        ) from exc

    if canonical_model_name not in available_models:
        preferred = preferred_model_names_fn(available_models)
        catalog_text = ", ".join(preferred) if preferred else ", ".join(sorted(available_models)[:12])
        raise LearnedBackendUnavailableError(
            f"Learned IQA model '{canonical_model_name}' is not available in the installed pyiqa runtime. "
            f"Available NR models include: {catalog_text}"
        )

    backend._pyiqa = pyiqa
    backend._torch = torch
    try:
        resolved_device = resolve_device_fn(device, torch_module=torch)
    except LearnedRuntimeUnavailableError as exc:
        raise LearnedBackendUnavailableError(str(exc)) from exc
    _ensure_model_runtime_compatible(
        canonical_model_name,
        runtime=resolved_device.runtime,
        torch_version=getattr(torch, "__version__", None),
    )

    if resolved_device.runtime == "cuda":
        cudnn_backend = getattr(getattr(torch, "backends", None), "cudnn", None)
        if cudnn_backend is not None and hasattr(cudnn_backend, "benchmark"):
            backend._previous_cudnn_benchmark = getattr(cudnn_backend, "benchmark")
            setattr(cudnn_backend, "benchmark", True)

    try:
        backend.metric = create_metric_safely_fn(pyiqa, canonical_model_name, device=resolved_device.metric_device)
    except Exception as exc:
        _restore_cudnn_benchmark(backend)
        raise LearnedBackendUnavailableError(
            f"Failed to initialize learned IQA model '{canonical_model_name}': {_exception_chain_text(exc)}"
        ) from exc

    backend.name = canonical_model_name
    backend.runtime = resolved_device.runtime
    backend.device = resolved_device.display_device
    backend.tensor_device = resolved_device.tensor_device
    backend.lower_better = bool(getattr(backend.metric, "lower_better", False))
    backend.score_range = str(getattr(backend.metric, "score_range", "0, 1"))
    backend.input_size = getattr(getattr(backend.metric, "net", None), "test_img_size", None) or input_sizes.get(canonical_model_name, default_input_size)
    backend.model_version = _model_version(pyiqa, canonical_model_name, backend.runtime)


def close_backend(backend, *, gc_module: _GcModuleLike = gc) -> None:
    if getattr(backend, "metric", None) is not None:
        backend.metric = None

    _restore_cudnn_benchmark(backend)

    torch_module = getattr(backend, "_torch", None)
    if torch_module is not None:
        for runtime_name in ("cuda", "xpu", "mps"):
            runtime_module = getattr(torch_module, runtime_name, None)
            empty_cache = getattr(runtime_module, "empty_cache", None)
            if callable(empty_cache):
                try:
                    empty_cache()
                except Exception:
                    continue

    gc_module.collect()


def score_paths(backend, image_paths: Sequence[Path], *, batch_size: int = DEFAULT_BATCH_SIZE, resource_profile: str | None = None, recommended_cpu_workers_fn, max_batch_sizes=None, load_batch_tensor_fn, arrays_to_tensor_fn, load_single_image_fn, result_cls=LearnedScoreResult, log_module=log, max_decode_pixels: int | None = None) -> list[LearnedScoreResult]:
    from concurrent.futures import Future, ThreadPoolExecutor

    batch_sizes = max_batch_sizes or MAX_BATCH_SIZES
    results: list[LearnedScoreResult] = []
    effective_batch_size = min(batch_size, batch_sizes.get(backend.name, batch_size))
    total = len(image_paths)
    pool_workers = min(effective_batch_size, recommended_cpu_workers_fn(resource_profile, for_threads=True))
    runtime = getattr(backend, "runtime", None)
    use_channels_last = runtime in {"cpu", "cuda", "rocm"}
    allow_prefetch_overlap = runtime != "cpu"
    effective_max_decode_pixels = (
        DEFAULT_MAX_DECODE_PIXELS if max_decode_pixels is None else max_decode_pixels
    )

    def load_batch(paths):
        return load_batch_tensor_fn(
            paths,
            image_size=backend.input_size,
            torch_module=backend._torch,
            tensor_device=backend.tensor_device,
            executor=load_pool,
            use_channels_last=use_channels_last,
            max_decode_pixels=effective_max_decode_pixels,
        )

    def load_single(path):
        return load_single_image_fn(
            path,
            backend.input_size,
            max_decode_pixels=effective_max_decode_pixels,
        )

    with ThreadPoolExecutor(max_workers=pool_workers) as load_pool:
        prefetch_futures: list[Future] | None = None

        for start in range(0, total, effective_batch_size):
            batch_paths = image_paths[start : start + effective_batch_size]

            try:
                if prefetch_futures is not None:
                    arrays = [future.result() for future in prefetch_futures]
                    batch_tensor = arrays_to_tensor_fn(
                        arrays,
                        torch_module=backend._torch,
                        tensor_device=backend.tensor_device,
                        use_channels_last=use_channels_last,
                    )
                    prefetch_futures = None
                else:
                    batch_tensor = load_batch(batch_paths)

                next_start = start + effective_batch_size
                if allow_prefetch_overlap and next_start < total:
                    next_paths = list(image_paths[next_start : next_start + effective_batch_size])
                    prefetch_futures = [
                        load_pool.submit(load_single, path)
                        for path in next_paths
                    ]

                results.extend(backend._score_tensor_batch(batch_tensor))
            except Exception as exc:
                log_module.warning("Batch scoring failed, falling back to individual scoring: %s", exc)
                prefetch_futures = None
                for single_path in batch_paths:
                    try:
                        single_tensor = load_batch([single_path])
                        results.extend(backend._score_tensor_batch(single_tensor))
                    except Exception as inner_exc:
                        log_module.error("Failed to score image %s: %s", single_path, inner_exc)
                        results.append(
                            result_cls(
                                raw_score=None,
                                normalized_score=None,
                                confidence=None,
                                error=str(inner_exc),
                            )
                        )

    return results


def _score_metric_output(backend, batch_tensor):
    try:
        return backend.metric(batch_tensor, return_mos=True, return_dist=True)
    except TypeError:
        return backend.metric(batch_tensor)


@contextmanager
def _tensor_autocast_context(backend):
    runtime = getattr(backend, "runtime", None)
    if runtime not in AUTOCAST_ENABLED_RUNTIMES:
        yield False
        return

    model_name = getattr(backend, "name", None)
    if model_name in AUTOCAST_BLOCKED_MODELS_BY_RUNTIME.get(runtime, frozenset()):
        yield False
        return

    torch_module = getattr(backend, "_torch", None)
    autocast = getattr(torch_module, "autocast", None)
    dtype = getattr(torch_module, "float16", None)
    if not callable(autocast) or dtype is None:
        yield False
        return

    try:
        autocast_context = cast(AbstractContextManager[object], autocast(runtime, dtype=dtype))
    except Exception as exc:
        log.warning(
            "Failed to enable %s autocast for learned-IQA model %s: %s; retrying without autocast",
            runtime,
            model_name or "<unknown>",
            exc,
        )
        yield False
        return

    with ExitStack() as stack:
        try:
            stack.enter_context(autocast_context)
        except Exception as exc:
            log.warning(
                "Failed to enter %s autocast for learned-IQA model %s: %s; retrying without autocast",
                runtime,
                model_name or "<unknown>",
                exc,
            )
            yield False
            return

        yield True


@contextmanager
def _tensor_grad_context(backend):
    torch_module = getattr(backend, "_torch", None)
    if torch_module is None:
        yield
        return

    context_factory = getattr(torch_module, "inference_mode", None)
    if not callable(context_factory):
        context_factory = getattr(torch_module, "no_grad", None)
    if not callable(context_factory):
        yield
        return

    context_factory = cast(Callable[[], AbstractContextManager[object]], context_factory)
    with context_factory():
        yield


def score_tensor_batch(backend, batch_tensor, *, flatten_tensor_fn, confidence_values_fn, normalize_score_fn, result_cls=LearnedScoreResult) -> list[LearnedScoreResult]:
    with _tensor_grad_context(backend):
        with _tensor_autocast_context(backend) as autocast_enabled:
            try:
                output = _score_metric_output(backend, batch_tensor)
            except Exception as exc:
                if not autocast_enabled:
                    raise
                log.warning(
                    "Forward pass failed under %s autocast for learned-IQA model %s: %s; retrying without autocast",
                    getattr(backend, "runtime", "<unknown>"),
                    getattr(backend, "name", None) or "<unknown>",
                    exc,
                )
                output = None

        if output is None:
            output = _score_metric_output(backend, batch_tensor)

    if isinstance(output, (list, tuple)):
        mos_tensor = output[0]
        dist_tensor = output[1] if len(output) > 1 else None
    else:
        mos_tensor = output
        dist_tensor = None

    raw_scores = flatten_tensor_fn(mos_tensor)
    confidences = confidence_values_fn(dist_tensor, torch_module=backend._torch) if dist_tensor is not None else [None] * len(raw_scores)

    return [
        result_cls(
            raw_score=raw_score,
            normalized_score=normalize_score_fn(raw_score, score_range=backend.score_range, lower_better=backend.lower_better),
            confidence=confidence,
        )
        for raw_score, confidence in zip(raw_scores, confidences, strict=True)
    ]


def available_learned_backends(*, resource_profile: str | None = None, import_pyiqa_runtime_fn, unavailable_backend_payload_fn, resolve_device_fn, runtime_statuses_fn, runtime_compatible_model_names_fn, preferred_model_names_fn, detect_hardware_capabilities_fn, valid_profile_fn, recommended_batch_size_fn, supported_model_names: Sequence[str], default_model_name: str, default_device_policy: str, supported_runtime_targets_fn, auto_runtime_order_fn, runtime_status_order: Sequence[str], model_catalog_payload_fn=None) -> dict[str, object]:
    catalog = ",".join(supported_model_names)
    runtime_targets = ",".join(supported_runtime_targets_fn())
    auto_priority = ",".join(auto_runtime_order_fn())
    vendor_aliases = "nvidia->cuda,amd->rocm(with validated HIP build),intel->xpu,apple->mps"

    try:
        pyiqa, torch = import_pyiqa_runtime_fn()
    except ImportError as exc:
        return unavailable_backend_payload_fn(status="not-installed", error=str(exc), resource_profile=resource_profile)
    except Exception as exc:
        return unavailable_backend_payload_fn(status="unavailable", error=str(exc), resource_profile=resource_profile)

    try:
        resolved = resolve_device_fn(None, torch_module=torch)
        statuses = runtime_statuses_fn(torch_module=torch)

        try:
            models = set(pyiqa.list_models(metric_mode="NR"))
        except Exception:
            models = set()

        compatible_models = set(
            runtime_compatible_model_names_fn(
                models,
                torch_version=getattr(torch, "__version__", None),
                runtime=resolved.runtime,
            )
        )
        if not compatible_models:
            return unavailable_backend_payload_fn(
                status="unavailable",
                error="The installed pyiqa runtime exposes no supported product models.",
                resource_profile=resource_profile,
            )
        preferred = preferred_model_names_fn(compatible_models)
        status_text = ",".join(f"{runtime}:{statuses[runtime]}" for runtime in runtime_status_order)
        hardware = detect_hardware_capabilities_fn()
        vram_mb = hardware.get("vram_mb")
        profile = valid_profile_fn(resource_profile)
        batch_recommendations = {
            model_name: recommended_batch_size_fn(model_name, vram_mb=vram_mb, resource_profile=profile)
            for model_name in supported_model_names
        }

        return {
            "pyiqa": "installed",
            "default_model": preferred[0] if preferred else default_model_name,
            "device_policy": default_device_policy,
            "default_device": resolved.display_device,
            "default_runtime": resolved.runtime,
            "runtime_fallback_reason": getattr(resolved, "fallback_reason", None),
            "runtime_targets": runtime_targets,
            "runtime_status": status_text,
            "auto_runtime_priority": auto_priority,
            "vendor_aliases": vendor_aliases,
            "modern_model_catalog": catalog,
            "modern_models_available": ",".join(preferred),
            "model_catalog": (
                model_catalog_payload_fn(available_models=preferred)
                if model_catalog_payload_fn is not None
                else None
            ),
            "hardware": hardware,
            "recommended_batch_sizes": batch_recommendations,
            "resource_profile": profile,
        }
    except Exception as exc:
        return unavailable_backend_payload_fn(status="unavailable", error=str(exc), resource_profile=resource_profile)


__all__ = [
    "LearnedBackendUnavailableError",
    "LearnedIqaBackend",
    "LearnedScoreResult",
    "available_learned_backends",
    "build_learned_backend",
    "close_backend",
    "create_metric_safely",
    "initialize_backend",
    "release_learned_backend",
    "resolve_learned_model_version",
    "score_paths",
    "score_tensor_batch",
]
