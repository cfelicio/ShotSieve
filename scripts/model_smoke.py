"""Prepare one supported learned model, including a fresh offline-process check."""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import socket
import sys
import time
from pathlib import Path

from shotsieve.dependency_constraints import installed_model_dependency_versions
from shotsieve.model_assets import build_model_diagnostic


_PRIVATE_ARTIFACT_PATH_PATTERN = re.compile(
    r"(?i)(?:[a-z]:[\\/]|/)[^\s'\",;]+\.(?:jpg|jpeg|png|gif|tif|tiff|webp|raw|cr2|nef|arw|bin|pth|pt|ckpt|safetensors)\b"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Supported model id, such as topiq_nr, clipiqa, or qalign")
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument(
        "--device",
        default=None,
        help="Requested validation runtime; CPU-compatible models default to CPU, while accelerator-only models default to Auto.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Set offline flags and reject socket connections before model preparation.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        help="Write the sanitized pass/failure report here; defaults inside --data-dir.",
    )
    parser.add_argument(
        "--driver-version",
        help="Record the exact accelerator driver version captured from the host OS.",
    )
    return parser


def _disable_network() -> None:
    def blocked_connect(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("network disabled for offline model smoke")

    socket.socket.connect = blocked_connect  # type: ignore[method-assign]
    socket.socket.connect_ex = blocked_connect  # type: ignore[method-assign]
    socket.create_connection = blocked_connect  # type: ignore[assignment]
    socket.getaddrinfo = blocked_connect  # type: ignore[assignment]


def _write_report(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _sanitize_private_artifacts(value: object) -> object:
    if isinstance(value, dict):
        return {key: _sanitize_private_artifacts(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_private_artifacts(item) for item in value]
    if isinstance(value, str):
        return _PRIVATE_ARTIFACT_PATH_PATTERN.sub("<redacted-artifact-path>", value)
    return value


def _start_runtime_measurement() -> dict[str, object]:
    """Import Torch lazily and reset supported accelerator peak counters."""
    try:
        import torch
    except Exception:
        return {"torch_module": None}

    for runtime in ("cuda", "xpu", "mps"):
        runtime_module = getattr(torch, runtime, None)
        reset_peak_memory_stats = getattr(runtime_module, "reset_peak_memory_stats", None)
        if callable(reset_peak_memory_stats):
            try:
                reset_peak_memory_stats()
            except Exception:
                continue
    return {"torch_module": torch}


def _runtime_evidence(
    *,
    measurement: dict[str, object],
    requested_runtime: str,
    actual_runtime: object,
    driver_version: str | None,
    elapsed_seconds: float,
) -> dict[str, object]:
    """Return shareable host/runtime facts for a model smoke report."""
    torch_module = measurement.get("torch_module")
    actual = str(actual_runtime or "unknown").strip().casefold()
    evidence: dict[str, object] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "requested_runtime": requested_runtime,
        "actual_runtime": actual,
        "driver_version": driver_version or "not-recorded",
        "elapsed_seconds": round(elapsed_seconds, 3),
        "peak_memory_mb": None,
    }
    if torch_module is None:
        evidence["torch_runtime"] = "not-imported"
        return evidence

    evidence["torch_runtime"] = str(getattr(torch_module, "__version__", "unknown"))
    runtime_module = getattr(torch_module, "cuda", None) if actual == "rocm" else getattr(torch_module, actual, None)
    if runtime_module is None:
        return evidence

    synchronize = getattr(runtime_module, "synchronize", None)
    if callable(synchronize):
        try:
            synchronize()
        except Exception:
            pass

    peak_memory = getattr(runtime_module, "max_memory_allocated", None)
    if callable(peak_memory):
        try:
            evidence["peak_memory_mb"] = round(float(peak_memory()) / (1024 * 1024), 2)
        except Exception:
            pass

    if actual == "rocm":
        hip_version = getattr(getattr(torch_module, "version", None), "hip", None)
        evidence["rocm_version"] = str(hip_version or "not-reported")
        device_count = getattr(runtime_module, "device_count", None)
        if callable(device_count):
            try:
                evidence["rocm_device_count"] = int(device_count())
            except Exception:
                pass
        get_device_name = getattr(runtime_module, "get_device_name", None)
        if callable(get_device_name):
            try:
                evidence["rocm_device_name"] = str(get_device_name(0))
            except Exception:
                pass
        get_device_properties = getattr(runtime_module, "get_device_properties", None)
        if callable(get_device_properties):
            try:
                properties = get_device_properties(0)
                architecture = getattr(properties, "gcnArchName", None) or getattr(properties, "name", None)
                if architecture:
                    evidence["rocm_gpu_architecture"] = str(architecture)
            except Exception:
                pass
    elif actual == "xpu":
        device_count = getattr(runtime_module, "device_count", None)
        if callable(device_count):
            try:
                evidence["xpu_device_count"] = int(device_count())
            except Exception:
                pass
        get_device_name = getattr(runtime_module, "get_device_name", None)
        if callable(get_device_name):
            try:
                evidence["xpu_device_name"] = str(get_device_name(0))
            except Exception:
                pass

    return evidence


def main() -> None:
    args = build_parser().parse_args()
    report_path = args.report_path or (args.data_dir / "model-smoke-report.json")
    requested_runtime = str(args.device or "auto").strip().casefold()
    measurement: dict[str, object] = {"torch_module": None}
    started = time.perf_counter()
    record: dict[str, object] = {}
    try:
        if args.offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            _disable_network()

        from shotsieve.model_assets import apply_model_cache_dir, prepare_model

        apply_model_cache_dir(args.cache_dir)
        measurement = _start_runtime_measurement()
        record = prepare_model(args.model, data_dir=args.data_dir, device=args.device)
        if record.get("state") != "prepared":
            raise RuntimeError("Model smoke did not produce a prepared record.")

        dependency_versions = installed_model_dependency_versions()
        report = {
            "status": "passed",
            "model": record.get("model"),
            "state": record.get("state"),
            "tested_runtime": record.get("tested_runtime"),
            "processed_counts": record.get("processed_counts"),
            "model_version": record.get("model_version"),
            "validation_scores": record.get("validation_scores", []),
            "cache_paths": record.get("cache_paths"),
            "dependency_versions": dependency_versions,
            "runtime_evidence": _runtime_evidence(
                measurement=measurement,
                requested_runtime=requested_runtime,
                actual_runtime=record.get("tested_runtime"),
                driver_version=args.driver_version,
                elapsed_seconds=time.perf_counter() - started,
            ),
        }
        _write_report(report_path, report)
        print(json.dumps(report, sort_keys=True))
    except Exception as exc:
        actual_runtime = record.get("tested_runtime") or record.get("actual_runtime")
        diagnostic = build_model_diagnostic(
            exc,
            phase="validating_initialization",
            model_name=args.model,
            requested_runtime=requested_runtime,
            actual_runtime=actual_runtime,
            cache_dir=args.cache_dir,
        )
        report = {
            "status": "failed",
            "model": args.model,
            "dependency_versions": installed_model_dependency_versions(),
            "runtime_evidence": _runtime_evidence(
                measurement=measurement,
                requested_runtime=requested_runtime,
                actual_runtime=actual_runtime,
                driver_version=args.driver_version,
                elapsed_seconds=time.perf_counter() - started,
            ),
            "diagnostic": _sanitize_private_artifacts(diagnostic),
        }
        try:
            _write_report(report_path, report)
        except OSError as report_error:
            print(f"Unable to write model smoke report: {report_error}", file=sys.stderr)
        print(json.dumps(report, sort_keys=True), file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
