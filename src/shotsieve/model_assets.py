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

from shotsieve.learned_iqa_catalog import MODEL_CATALOG, validate_model_name

PREPARATION_RECORD_NAME = "model-preparation.json"
PREPARATION_STATES = ("not_checked", "preparing", "prepared", "failed", "runtime_unavailable")
_OFFLINE_ENV_NAMES = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
_VERSION_PACKAGE_NAMES = (
    "shotsieve",
    "pyiqa",
    "torch",
    "torchvision",
    "torch-directml",
    "timm",
    "huggingface-hub",
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
    payload = {
        "model": model_name,
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
        "weight_mb": None,
        "temporary_overhead_mb": None,
        "total_mb": None,
        "note": "Upstream assets and temporary download overhead vary; verify free space before preparation.",
    }


def _base_record(
    model_name: str,
    *,
    cache_paths: Mapping[str, object],
    dependency_versions: Mapping[str, str],
    dependency_fingerprint: str,
    environ: Mapping[str, str],
) -> dict[str, object]:
    now = _utc_now()
    return {
        "state": "preparing",
        "model": model_name,
        "started_at": now,
        "updated_at": now,
        "finished_at": None,
        "phase": "checking_storage",
        "requested_runtime": "cpu",
        "actual_runtime": None,
        "tested_runtime": None,
        "dependency_fingerprint": dependency_fingerprint,
        "dependency_versions": dict(dependency_versions),
        "cache_paths": dict(cache_paths),
        "offline": _offline_flags(environ),
        "expected_resources": expected_resources(model_name),
        "disk_estimate": storage_estimate(model_name),
        "processed_counts": {"validation_images": 0},
        "asset_check": {"status": "pending", "method": "backend_initialization_and_cpu_inference"},
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
        "model": model_name,
        "reason": reason,
        "cache_paths": dict(cache_paths or {}),
        "last_preparation": None,
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
            "dependency_versions": current_versions,
            "dependency_fingerprint": current_fingerprint,
        }
    if record.get("state") == "prepared":
        recorded_paths = record.get("cache_paths")
        if isinstance(recorded_paths, dict):
            required_roots = ("hf_hub_cache", "torch_home") if canonical_model == "topiq_nr" else ("torch_home",)
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
                    "dependency_versions": current_versions,
                    "dependency_fingerprint": current_fingerprint,
                }
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

    if offline and any(token in messages for token in ("not found in cache", "offline", "local_files_only", "no cached")):
        category = "missing_offline_assets"
        recovery = "Turn off offline mode or populate the required model caches, then retry preparation."
    elif any(isinstance(item, (PermissionError,)) or getattr(item, "errno", None) in {13, 1} for item in chain) or any(
        token in messages for token in ("permission denied", "access is denied", "read-only file system")
    ):
        category = "cache_permissions"
        recovery = "Choose a writable model cache directory and retry preparation."
    elif any(getattr(item, "errno", None) == 28 for item in chain) or any(
        token in messages for token in ("no space left", "not enough space", "disk full")
    ):
        category = "cache_no_space"
        recovery = "Free disk space or choose a larger writable model cache directory, then retry."
    elif any(token in messages for token in ("checksum", "corrupt", "incomplete download", "invalid archive")):
        category = "cache_corruption"
        recovery = "Remove only the affected upstream cache entry using its upstream cache tools, then retry."
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
        recovery = "Check network, proxy, certificate, or Hub endpoint settings, then retry preparation."
    elif any(isinstance(item, (ImportError, ModuleNotFoundError)) for item in chain):
        category = "missing_dependencies"
        recovery = "Install or repair the optional learned-IQA dependencies, then retry preparation."
    elif "learnedruntimeunavailableerror" in type_names or "learnedbackendunavailableerror" in type_names:
        category = "runtime_unavailable"
        recovery = "Install or repair a supported learned-IQA runtime, then retry preparation."
    elif phase == "validating_initialization":
        category = "execution_or_operator"
        recovery = "Retry preparation; if it repeats, check the selected model runtime and its diagnostic details."
    else:
        category = "unknown"
        recovery = "Retry preparation. If it repeats, retain this diagnostic when reporting the issue."

    causes = [f"{type(item).__name__}: {_sanitize_text(item, environ=env)}" for item in chain]
    return {
        "category": category,
        "phase": phase,
        "cause": causes[0] if causes else type(exc).__name__,
        "cause_chain": causes,
        "recovery_action": recovery,
    }


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
    progress_callback: Callable[[dict[str, object]], None] | None = None,
    cancel_check: Callable[[], None] | None = None,
    backend_factory: Callable[..., object] | None = None,
    backend_release: Callable[[object], None] | None = None,
    record_writer: Callable[[Path, Mapping[str, object]], dict[str, object]] = write_preparation_record,
    environ: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Prepare exactly one supported model on CPU and validate one tiny inference."""
    canonical_model = validate_model_name(model_name)
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
        cache_paths=cache_paths,
        dependency_versions=dependency_versions,
        dependency_fingerprint=fingerprint,
        environ=env,
    )
    backend: object | None = None
    current_phase = "checking_storage"

    def save(**updates: object) -> None:
        record.update(updates)
        record_writer(data_dir, record)
        _emit_progress(progress_callback, record)

    try:
        save()
        if cancel_check is not None:
            cancel_check()
        data_dir.mkdir(parents=True, exist_ok=True)
        usage = shutil.disk_usage(data_dir)
        save(storage_check={"free_bytes": usage.free, "total_bytes": usage.total, "estimate": "advisory"})

        current_phase = "preparing_model"
        save(phase=current_phase)
        if cancel_check is not None:
            cancel_check()
        if backend_factory is None:
            from shotsieve.learned_iqa import build_learned_backend

            backend_factory = build_learned_backend
        backend = backend_factory(canonical_model, device="cpu")
        actual_runtime = str(getattr(backend, "runtime", "cpu")).casefold()
        save(actual_runtime=actual_runtime)
        if actual_runtime != "cpu":
            raise RuntimeError("CPU preparation returned a non-CPU learned runtime.")

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
        save(
            state="prepared",
            phase="complete",
            actual_runtime="cpu",
            tested_runtime="cpu",
            asset_check={"status": "passed", "method": "backend_initialization_and_cpu_inference"},
            finished_at=_utc_now(),
            processed_counts={"validation_images": 1},
            error=None,
            error_report=None,
            recovery_action=None,
        )
        return dict(record)
    except InterruptedError as exc:
        report = {
            "category": "cancelled",
            "phase": current_phase,
            "cause": _sanitize_text(exc, environ=env),
            "cause_chain": [_sanitize_text(exc, environ=env)],
            "recovery_action": "Preparation was cancelled. Retry to complete model validation.",
        }
        try:
            save(
                state="failed",
                phase=current_phase,
                finished_at=_utc_now(),
                error=report["cause"],
                error_report=report,
                recovery_action=report["recovery_action"],
                cancelled=True,
            )
        except Exception as record_error:
            record["record_write_error"] = _sanitize_text(record_error, environ=env)
        raise
    except Exception as exc:
        report = classify_preparation_error(exc, phase=current_phase, environ=env)
        state = "runtime_unavailable" if report["category"] == "runtime_unavailable" else "failed"
        try:
            save(
                state=state,
                phase=current_phase,
                finished_at=_utc_now(),
                error=report["cause"],
                error_report=report,
                recovery_action=report["recovery_action"],
            )
        except Exception as record_error:
            record["record_write_error"] = _sanitize_text(record_error, environ=env)
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
    "PREPARATION_RECORD_NAME",
    "PREPARATION_STATES",
    "apply_model_cache_dir",
    "classify_preparation_error",
    "effective_cache_paths",
    "expected_resources",
    "preparation_record_path",
    "prepare_model",
    "read_preparation_record",
    "storage_estimate",
    "write_preparation_record",
]
