"""Small, durable model-preparation and readiness diagnostics.

Preparation deliberately lives outside the catalog database.  Model downloads and
initialization can fail after a caller has opened a transaction, so a tiny atomic
JSON record is a better fit for the last preparation attempt than another DB table.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import tempfile
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from shotsieve.learned_iqa_catalog import (
    MODEL_CATALOG,
    is_model_runtime_compatible,
    validate_model_name,
)

PREPARATION_RECORD_NAME = "model-preparation.json"
PREPARATION_STATES = ("not_checked", "preparing", "prepared", "failed", "runtime_unavailable")
MODEL_DIAGNOSTIC_SCHEMA_VERSION = 1
_OFFLINE_ENV_NAMES = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
_VERSION_PACKAGE_NAMES = (
    "shotsieve",
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
_SENSITIVE_ENV_NAMES = {
    "HF_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "PIP_INDEX_URL",
    "PIP_EXTRA_INDEX_URL",
}
_SENSITIVE_MESSAGE_PATTERN = re.compile(
    r"(?i)(authorization|api[_-]?key|access[_-]?token|password|secret|bearer|token)\s*[:=]\s*[^\s,;]+"
)
_URL_PATTERN = re.compile(r"https?://[^\s'\"]+")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def preparation_record_path(data_dir: Path) -> Path:
    return Path(data_dir) / PREPARATION_RECORD_NAME


def _path_text(value: object) -> str | None:
    if value is None or not str(value).strip():
        return None
    try:
        return str(Path(str(value)).expanduser().resolve())
    except (OSError, RuntimeError, ValueError):
        return str(value)


def effective_cache_paths(
    *,
    cache_dir: Path | str | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str | None]:
    """Return effective split cache locations without importing ML libraries."""
    env = os.environ if environ is None else environ
    root = _path_text(cache_dir)
    root_path = Path(root) if root else None

    default_cache = Path(env.get("XDG_CACHE_HOME") or str(Path.home() / ".cache"))
    hf_home = env.get("HF_HOME") or str((root_path or default_cache) / "huggingface")
    hf_hub_cache = env.get("HF_HUB_CACHE") or env.get("HUGGINGFACE_HUB_CACHE")
    if not hf_hub_cache and hf_home:
        hf_hub_cache = str(Path(hf_home) / "hub")
    torch_home = env.get("TORCH_HOME") or str((root_path or default_cache) / "torch")
    return {
        "requested_root": root,
        "hf_home": _path_text(hf_home),
        "hf_hub_cache": _path_text(hf_hub_cache),
        "torch_home": _path_text(torch_home),
    }


def effective_cache_volumes(*, cache_paths: Mapping[str, object]) -> dict[str, dict[str, object]]:
    """Report free space for the volumes backing the effective model caches."""
    volumes: dict[str, dict[str, object]] = {}
    for name in ("requested_root", "hf_home", "hf_hub_cache", "torch_home"):
        raw_path = cache_paths.get(name)
        if not isinstance(raw_path, str) or not raw_path.strip():
            continue
        path = Path(raw_path)
        probe = path
        try:
            while not probe.exists() and probe != probe.parent:
                probe = probe.parent
            usage = shutil.disk_usage(probe)
        except (OSError, ValueError) as exc:
            volumes[name] = {
                "path": str(path),
                "volume": str(path.anchor or path),
                "status": "unavailable",
                "error": f"{type(exc).__name__}: {exc}",
            }
            continue
        volumes[name] = {
            "path": str(path),
            "volume": str(probe.anchor or probe),
            "status": "available",
            "free_bytes": usage.free,
            "total_bytes": usage.total,
        }
    return volumes


def apply_model_cache_dir(
    cache_dir: Path | str | None,
    *,
    environ: dict[str, str] | None = None,
) -> dict[str, str | None]:
    """Set default cache roots, preserving explicit user environment settings."""
    env = os.environ if environ is None else environ
    if cache_dir is None:
        return effective_cache_paths(environ=env)

    root = Path(cache_dir).expanduser().resolve()
    env.setdefault("HF_HOME", str(root / "huggingface"))
    env.setdefault("HF_HUB_CACHE", env.get("HUGGINGFACE_HUB_CACHE") or str(Path(env["HF_HOME"]) / "hub"))
    env.setdefault("TORCH_HOME", str(root / "torch"))
    return effective_cache_paths(cache_dir=root, environ=env)


def _offline_flags(environ: Mapping[str, str]) -> dict[str, bool]:
    return {
        name: str(environ.get(name, "")).strip().casefold() in {"1", "true", "yes", "on"}
        for name in _OFFLINE_ENV_NAMES
    }


def _dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package_name in _VERSION_PACKAGE_NAMES:
        try:
            versions[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            versions[package_name] = "not-installed"
        except Exception:
            versions[package_name] = "unknown"
    return versions


def _dependency_fingerprint(
    model_name: str,
    *,
    cache_paths: Mapping[str, object],
    dependency_versions: Mapping[str, str],
    environ: Mapping[str, str],
) -> str:
    spec = _model_spec(model_name)
    payload = {
        # Include the upstream identity and immutable revision so a readiness
        # record can never be reused for a renamed checkpoint or a changed
        # Q-ReAlign Mini revision.
        "model": model_name,
        "upstream_model_id": spec.upstream_model_id,
        "checkpoint_revision": spec.checkpoint_revision,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cache_paths": dict(cache_paths),
        "dependency_versions": dict(dependency_versions),
        "offline": _offline_flags(environ),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _model_spec(model_name: str):
    return next(spec for spec in MODEL_CATALOG if spec.canonical_id == model_name)


def expected_resources(model_name: str) -> dict[str, object]:
    spec = _model_spec(model_name)
    return {
        "status": "candidates",
        "upstream_model_id": spec.upstream_model_id,
        "checkpoint_revision": spec.checkpoint_revision,
        "cache_families": list(spec.cache_families),
        "first_use_disclosure": spec.first_use_disclosure,
        "resource_labels": list(spec.resource_labels),
    }


def storage_estimate(model_name: str) -> dict[str, object]:
    """Return an intentionally advisory estimate when upstream totals vary."""
    spec = _model_spec(model_name)
    return {
        "status": "advisory",
        "model_input_size": spec.input_size,
        "weight_mb": 2210 if spec.canonical_id == "qrealign-mini" else None,
        "temporary_overhead_mb": None,
        "total_mb": None,
        "note": (
            "The Q-ReAlign Mini safetensors file is about 2.21 GB; tokenizer, processor, "
            "temporary loading, and runtime memory are additional. Verify free space and "
            "record actual peak memory and throughput before claiming support."
            if spec.canonical_id == "qrealign-mini"
            else "Upstream assets and temporary download overhead vary; verify free space before preparation."
        ),
    }


def _base_record(
    model_name: str,
    *,
    requested_runtime: str = "cpu",
    cache_paths: Mapping[str, object],
    dependency_versions: Mapping[str, str],
    dependency_fingerprint: str,
    environ: Mapping[str, str],
) -> dict[str, object]:
    model_spec = _model_spec(model_name)
    now = _utc_now()
    return {
        "state": "preparing",
        "readiness": "last_check",
        "model": model_name,
        "upstream_model_id": model_spec.upstream_model_id,
        "model_revision": model_spec.checkpoint_revision,
        "process_id": os.getpid(),
        "started_at": now,
        "updated_at": now,
        "finished_at": None,
        "phase": "checking_storage",
        "requested_runtime": requested_runtime,
        "actual_runtime": None,
        "tested_runtime": None,
        "dependency_fingerprint": dependency_fingerprint,
        "dependency_versions": dict(dependency_versions),
        "cache_paths": dict(cache_paths),
        "cache_volumes": effective_cache_volumes(cache_paths=cache_paths),
        "offline": _offline_flags(environ),
        "expected_resources": expected_resources(model_name),
        "disk_estimate": storage_estimate(model_name),
        "processed_counts": {"validation_images": 0},
        "model_version": None,
        "validation_scores": [],
        "asset_check": {"status": "pending", "method": "backend_initialization_and_inference"},
        "error": None,
        "error_report": None,
        "recovery_action": None,
    }


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=True, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


def write_preparation_record(data_dir: Path, record: Mapping[str, object]) -> dict[str, object]:
    payload = dict(record)
    payload["updated_at"] = _utc_now()
    _atomic_write_json(preparation_record_path(data_dir), payload)
    return payload


def _not_checked_record(
    *,
    reason: str,
    model_name: str | None = None,
    cache_paths: Mapping[str, object] | None = None,
) -> dict[str, object]:
    return {
        "state": "not_checked",
        "readiness": "last_check",
        "model": model_name,
        "reason": reason,
        "cache_paths": dict(cache_paths or {}),
        "cache_volumes": effective_cache_volumes(cache_paths=cache_paths or {}),
        "last_preparation": None,
    }


def _process_is_alive(process_id: object) -> bool:
    try:
        pid = int(process_id)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except PermissionError:
        return True
    except (OSError, ProcessLookupError):
        return False
    return True


def _orphaned_preparation_record(
    record: dict[str, object],
    *,
    cache_paths: Mapping[str, object],
    environ: Mapping[str, str],
) -> dict[str, object]:
    model_name = record.get("model")
    diagnostic = build_model_diagnostic(
        RuntimeError("The previous model preparation process ended before it completed."),
        model_name=model_name if isinstance(model_name, str) else None,
        requested_runtime=record.get("requested_runtime"),
        actual_runtime=record.get("actual_runtime"),
        phase=str(record.get("phase") or "preparing_model"),
        cache_paths=cache_paths,
        environ=environ,
    )
    diagnostic["category"] = "interrupted_process"
    diagnostic["recovery_action"] = "Preparation was interrupted. Open Settings and choose Prepare selected model, then retry."
    return {
        **record,
        "state": "failed",
        "readiness": "last_check",
        "finished_at": _utc_now(),
        "updated_at": _utc_now(),
        "error": diagnostic["cause"],
        "error_report": diagnostic,
        "recovery_action": diagnostic["recovery_action"],
        "orphaned": True,
        "cache_paths": dict(cache_paths),
        "cache_volumes": effective_cache_volumes(cache_paths=cache_paths),
    }


def read_preparation_record(
    data_dir: Path,
    *,
    cache_dir: Path | str | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, object]:
    env = os.environ if environ is None else environ
    cache_paths = effective_cache_paths(cache_dir=cache_dir, environ=env)
    path = preparation_record_path(data_dir)
    try:
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except FileNotFoundError:
        return _not_checked_record(reason="no_preparation_record", cache_paths=cache_paths)
    except (OSError, ValueError, TypeError):
        return _not_checked_record(reason="invalid_preparation_record", cache_paths=cache_paths)

    if not isinstance(raw, dict):
        return _not_checked_record(reason="invalid_preparation_record", cache_paths=cache_paths)
    record = dict(raw)
    model_name = record.get("model")
    if not isinstance(model_name, str) or not model_name:
        return _not_checked_record(reason="record_missing_model", cache_paths=cache_paths)
    try:
        canonical_model = validate_model_name(model_name)
    except ValueError:
        return _not_checked_record(reason="record_model_not_supported", model_name=model_name, cache_paths=cache_paths)

    if record.get("state") == "preparing" and not _process_is_alive(record.get("process_id")):
        record = _orphaned_preparation_record(record, cache_paths=cache_paths, environ=env)
        try:
            write_preparation_record(data_dir, record)
        except Exception as exc:
            record["record_write_error"] = _sanitize_text(exc, environ=env)

    current_versions = _dependency_versions()
    current_fingerprint = _dependency_fingerprint(
        canonical_model,
        cache_paths=cache_paths,
        dependency_versions=current_versions,
        environ=env,
    )
    if record.get("dependency_fingerprint") != current_fingerprint:
        return {
            "state": "not_checked",
            "model": canonical_model,
            "reason": "preparation_context_changed",
            "invalidated_state": record.get("state"),
            "last_preparation": record,
            "cache_paths": cache_paths,
            "cache_volumes": effective_cache_volumes(cache_paths=cache_paths),
            "dependency_versions": current_versions,
            "dependency_fingerprint": current_fingerprint,
        }
    if record.get("state") == "prepared":
        recorded_paths = record.get("cache_paths")
        if isinstance(recorded_paths, dict):
            required_roots = _model_spec(canonical_model).required_cache_roots
            missing_roots = [
                name
                for name in required_roots
                if isinstance(recorded_paths.get(name), str)
                and recorded_paths[name]
                and not Path(str(recorded_paths[name])).exists()
            ]
            if missing_roots:
                return {
                    "state": "not_checked",
                    "model": canonical_model,
                    "reason": "checked_cache_root_missing",
                    "invalidated_state": record.get("state"),
                    "missing_cache_roots": missing_roots,
                    "last_preparation": record,
                    "cache_paths": cache_paths,
                    "cache_volumes": effective_cache_volumes(cache_paths=cache_paths),
                    "dependency_versions": current_versions,
                    "dependency_fingerprint": current_fingerprint,
                }
    record.setdefault("readiness", "last_check")
    record["cache_paths"] = cache_paths
    record["cache_volumes"] = effective_cache_volumes(cache_paths=cache_paths)
    return record


def _sanitize_text(value: object, *, environ: Mapping[str, str]) -> str:
    text = str(value or "").strip()
    for name, secret in environ.items():
        if name in _SENSITIVE_ENV_NAMES and secret:
            text = text.replace(secret, "<redacted>")
    text = re.sub(r"(?i)\b(authorization\s*[:=]\s*(?:bearer|basic)|bearer)\s+[^\s,;]+", r"\1 <redacted>", text)
    text = _SENSITIVE_MESSAGE_PATTERN.sub(lambda match: f"{match.group(1)}=<redacted>", text)

    def redact_url(match: re.Match[str]) -> str:
        raw_url = match.group(0)
        try:
            parsed = urlsplit(raw_url)
            hostname = parsed.hostname or "host"
            netloc = hostname
            if parsed.port:
                netloc = f"{hostname}:{parsed.port}"
            return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))
        except ValueError:
            return "<redacted-url>"

    return _URL_PATTERN.sub(redact_url, text)[:2000]


def _exception_chain(exc: BaseException) -> list[BaseException]:
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen and len(chain) < 8:
        seen.add(id(current))
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain


def classify_preparation_error(
    exc: BaseException,
    *,
    phase: str,
    environ: Mapping[str, str] | None = None,
) -> dict[str, object]:
    env = os.environ if environ is None else environ
    chain = _exception_chain(exc)
    messages = " ".join(_sanitize_text(item, environ=env).casefold() for item in chain)
    type_names = {type(item).__name__.casefold() for item in chain}
    offline = any(_offline_flags(env).values())

    if {"outofmemoryerror", "memoryerror"} & type_names or "out of memory" in messages:
        category = "runtime_out_of_memory"
        recovery = (
            "Free accelerator/system memory, lower the batch size, or use a smaller model, then choose Prepare selected model. "
            "Q-ReAlign Mini's safetensors are about 2.2 GB before processor, temporary loading, "
            "and inference overhead; record actual peak memory on the target host."
        )
    elif offline and any(token in messages for token in ("not found in cache", "offline", "local_files_only", "no cached")):
        category = "missing_offline_assets"
        recovery = "Populate the required model caches or turn off offline mode, then open Settings and choose Prepare selected model."
    elif any(isinstance(item, (PermissionError,)) or getattr(item, "errno", None) in {13, 1} for item in chain) or any(
        token in messages for token in ("permission denied", "access is denied", "read-only file system")
    ):
        category = "cache_permissions"
        recovery = "Choose a writable model cache directory, open Settings, and choose Prepare selected model."
    elif any(getattr(item, "errno", None) == 28 for item in chain) or any(
        token in messages for token in ("no space left", "not enough space", "disk full")
    ):
        category = "cache_no_space"
        recovery = "Free disk space or choose a larger writable model cache directory, then choose Prepare selected model."
    elif any(token in messages for token in ("checksum", "corrupt", "incomplete download", "invalid archive")):
        category = "cache_corruption"
        recovery = "Remove only the affected upstream cache entry using its upstream cache tools, then choose Prepare selected model."
    elif any(
        token in messages
        for token in (
            "ssl",
            "tls",
            "certificate",
            "proxy",
            "timed out",
            "timeout",
            "connection",
            "hugging face",
            "huggingface",
            "hub request",
            "http://",
            "https://",
        )
    ) or {"connectionerror", "timeouterror", "sslerror", "proxyerror"} & type_names:
        category = "network_or_hub"
        recovery = "Check network, proxy, certificate, or Hub endpoint settings, then choose Prepare selected model."
    elif any(isinstance(item, (ImportError, ModuleNotFoundError)) for item in chain):
        category = "missing_dependencies"
        recovery = "Install or repair the optional learned-IQA dependencies, then choose Prepare selected model."
    elif "learnedruntimeunavailableerror" in type_names or "learnedbackendunavailableerror" in type_names:
        category = "runtime_unavailable"
        recovery = "Install or repair a supported learned-IQA runtime, then choose Prepare selected model."
    elif phase == "validating_initialization":
        category = "execution_or_operator"
        recovery = "Open Settings and choose Prepare selected model, then retry the operation; if it repeats, retain this diagnostic."
    else:
        category = "unknown"
        recovery = "Open Settings and choose Prepare selected model, then retry the operation. If it repeats, retain this diagnostic when reporting the issue."

    causes = [f"{type(item).__name__}: {_sanitize_text(item, environ=env)}" for item in chain]
    return {
        "category": category,
        "phase": phase,
        "cause": causes[0] if causes else type(exc).__name__,
        "cause_chain": causes,
        "recovery_action": recovery,
    }


def build_model_diagnostic(
    exc: BaseException,
    *,
    phase: str,
    model_name: object = None,
    requested_runtime: object = None,
    actual_runtime: object = None,
    cache_dir: Path | str | None = None,
    cache_paths: Mapping[str, object] | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Build the shared, sanitized diagnostic used by prepare, score, and compare."""
    env = os.environ if environ is None else environ
    paths = dict(cache_paths) if cache_paths is not None else effective_cache_paths(cache_dir=cache_dir, environ=env)
    report = classify_preparation_error(exc, phase=phase, environ=env)
    model_text = str(model_name).strip() if model_name is not None and str(model_name).strip() else None
    requested_text = str(requested_runtime).strip().casefold() if requested_runtime is not None and str(requested_runtime).strip() else "auto"
    actual_text = str(actual_runtime).strip().casefold() if actual_runtime is not None and str(actual_runtime).strip() else "unknown"
    model_revision = None
    if model_text:
        try:
            model_revision = _model_spec(validate_model_name(model_text)).checkpoint_revision
        except (ValueError, StopIteration):
            pass
    return {
        "schema_version": MODEL_DIAGNOSTIC_SCHEMA_VERSION,
        **report,
        "model": model_text,
        "requested_runtime": requested_text,
        "actual_runtime": actual_text,
        "model_revision": model_revision,
        "cache_paths": paths,
        "cache_volumes": effective_cache_volumes(cache_paths=paths),
        "offline": _offline_flags(env),
        "prepare_action": "Open Settings and choose Prepare selected model before retrying scoring or comparison.",
    }


def attach_model_diagnostic(
    exc: BaseException,
    *,
    phase: str,
    model_name: object = None,
    requested_runtime: object = None,
    actual_runtime: object = None,
    cache_dir: Path | str | None = None,
    cache_paths: Mapping[str, object] | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, object]:
    diagnostic = build_model_diagnostic(
        exc,
        phase=phase,
        model_name=model_name,
        requested_runtime=requested_runtime,
        actual_runtime=actual_runtime,
        cache_dir=cache_dir,
        cache_paths=cache_paths,
        environ=environ,
    )
    try:
        setattr(exc, "model_diagnostic", diagnostic)
    except Exception:
        pass
    return diagnostic


def recover_orphaned_preparation(data_dir: Path, *, environ: Mapping[str, str] | None = None) -> dict[str, object]:
    """Read readiness once at startup and downgrade an abandoned preparation."""
    return read_preparation_record(data_dir, environ=environ)


def _emit_progress(
    callback: Callable[[dict[str, object]], None] | None,
    record: Mapping[str, object],
) -> None:
    if callback is not None:
        callback(dict(record))


def prepare_model(
    model_name: str,
    *,
    data_dir: Path,
    device: str | None = None,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
    cancel_check: Callable[[], None] | None = None,
    backend_factory: Callable[..., object] | None = None,
    backend_release: Callable[[object], None] | None = None,
    record_writer: Callable[[Path, Mapping[str, object]], dict[str, object]] = write_preparation_record,
    environ: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Prepare exactly one supported model and validate one tiny inference."""
    canonical_model = validate_model_name(model_name)
    model_spec = _model_spec(canonical_model)
    requested_runtime = str(device or ("cpu" if "cpu" in model_spec.supported_runtimes else "auto")).strip().casefold()
    if not requested_runtime:
        requested_runtime = "auto"
    env = os.environ if environ is None else environ
    cache_paths = effective_cache_paths(environ=env)
    dependency_versions = _dependency_versions()
    fingerprint = _dependency_fingerprint(
        canonical_model,
        cache_paths=cache_paths,
        dependency_versions=dependency_versions,
        environ=env,
    )
    record = _base_record(
        canonical_model,
        requested_runtime=requested_runtime,
        cache_paths=cache_paths,
        dependency_versions=dependency_versions,
        dependency_fingerprint=fingerprint,
        environ=env,
    )
    backend: object | None = None
    current_phase = "checking_storage"

    def save(**updates: object) -> None:
        record.update(updates)
        try:
            record_writer(data_dir, record)
            record.pop("record_write_error", None)
        except Exception as record_error:
            # Readiness persistence is diagnostic state.  It must not hide the
            # original model failure or turn a successful preparation into a
            # failed capability.
            record["record_write_error"] = _sanitize_text(record_error, environ=env)
        _emit_progress(progress_callback, record)

    try:
        save()
        if cancel_check is not None:
            cancel_check()
        data_dir.mkdir(parents=True, exist_ok=True)
        usage = shutil.disk_usage(data_dir)
        save(
            storage_check={
                "data_dir": {"free_bytes": usage.free, "total_bytes": usage.total},
                "cache_volumes": effective_cache_volumes(cache_paths=cache_paths),
                "estimate": "advisory",
            }
        )

        current_phase = "preparing_model"
        save(phase=current_phase)
        if cancel_check is not None:
            cancel_check()
        if backend_factory is None:
            from shotsieve.learned_iqa import build_learned_backend

            backend_factory = build_learned_backend
        backend = backend_factory(canonical_model, device=requested_runtime)
        actual_runtime = str(getattr(backend, "runtime", "unknown")).casefold()
        save(actual_runtime=actual_runtime)
        if not is_model_runtime_compatible(
            canonical_model,
            torch_version=None,
            runtime=actual_runtime,
        ):
            raise RuntimeError(
                f"Preparation runtime '{actual_runtime}' is not compatible with model '{canonical_model}'."
            )

        current_phase = "validating_initialization"
        save(phase=current_phase)
        if cancel_check is not None:
            cancel_check()
        with tempfile.TemporaryDirectory(prefix="shotsieve-model-check-") as temporary_dir:
            validation_path = Path(temporary_dir) / "validation.png"
            from PIL import Image

            Image.new("RGB", (32, 32), (127, 127, 127)).save(validation_path, format="PNG")
            save(processed_counts={"validation_images": 1})
            results = backend.score_paths([validation_path], batch_size=1, resource_profile="low")
            if not results:
                raise RuntimeError("The model returned no result during validation inference.")
            failed = [item for item in results if bool(getattr(item, "failed", False))]
            if failed:
                detail = getattr(failed[0], "error", None) or "the model returned a failed result"
                raise RuntimeError(f"Validation inference failed: {detail}")
            validation_scores = [
                {
                    "raw_score": getattr(item, "raw_score", None),
                    "normalized_score": getattr(item, "normalized_score", None),
                    "confidence": getattr(item, "confidence", None),
                }
                for item in results
            ]
            save(
                model_version=getattr(backend, "model_version", None),
                validation_scores=validation_scores,
            )
        save(
            state="prepared",
            phase="complete",
            actual_runtime=actual_runtime,
            tested_runtime=actual_runtime,
            asset_check={"status": "passed", "method": "backend_initialization_and_inference"},
            finished_at=_utc_now(),
            processed_counts={"validation_images": 1},
            error=None,
            error_report=None,
            recovery_action=None,
        )
        return dict(record)
    except InterruptedError as exc:
        report = build_model_diagnostic(
            exc,
            phase=current_phase,
            model_name=canonical_model,
            requested_runtime=requested_runtime,
            actual_runtime=record.get("actual_runtime"),
            cache_paths=cache_paths,
            environ=env,
        )
        report.update({
            "category": "cancelled",
            "recovery_action": "Preparation was cancelled. Open Settings and choose Prepare selected model to retry.",
        })
        save(
            state="failed",
            phase=current_phase,
            finished_at=_utc_now(),
            error=report["cause"],
            error_report=report,
            recovery_action=report["recovery_action"],
            cancelled=True,
        )
        attached = attach_model_diagnostic(
            exc,
            phase=current_phase,
            model_name=canonical_model,
            requested_runtime=requested_runtime,
            actual_runtime=record.get("actual_runtime"),
            cache_paths=cache_paths,
            environ=env,
        )
        if record.get("record_write_error"):
            attached["record_write_error"] = record["record_write_error"]
        raise
    except Exception as exc:
        report = attach_model_diagnostic(
            exc,
            phase=current_phase,
            model_name=canonical_model,
            requested_runtime=requested_runtime,
            actual_runtime=record.get("actual_runtime"),
            cache_paths=cache_paths,
            environ=env,
        )
        state = "runtime_unavailable" if report["category"] == "runtime_unavailable" else "failed"
        save(
            state=state,
            phase=current_phase,
            finished_at=_utc_now(),
            error=report["cause"],
            error_report=report,
            recovery_action=report["recovery_action"],
        )
        if record.get("record_write_error"):
            report["record_write_error"] = record["record_write_error"]
        raise
    finally:
        if backend is not None:
            if backend_release is None:
                from shotsieve.learned_iqa import release_learned_backend

                backend_release = release_learned_backend
            try:
                backend_release(backend)
            except Exception:
                # Cleanup must never replace the useful preparation failure or
                # make a successful preparation appear to have crashed.
                pass


__all__ = [
    "MODEL_DIAGNOSTIC_SCHEMA_VERSION",
    "PREPARATION_RECORD_NAME",
    "PREPARATION_STATES",
    "attach_model_diagnostic",
    "apply_model_cache_dir",
    "build_model_diagnostic",
    "classify_preparation_error",
    "effective_cache_paths",
    "effective_cache_volumes",
    "expected_resources",
    "preparation_record_path",
    "prepare_model",
    "recover_orphaned_preparation",
    "read_preparation_record",
    "storage_estimate",
    "write_preparation_record",
]
