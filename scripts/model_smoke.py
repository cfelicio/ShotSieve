"""Prepare one supported learned model, including a fresh offline-process check."""
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import sys
from pathlib import Path

from shotsieve.dependency_constraints import installed_model_dependency_versions
from shotsieve.model_assets import build_model_diagnostic


_PRIVATE_ARTIFACT_PATH_PATTERN = re.compile(
    r"(?i)(?:[a-z]:[\\/]|/)[^\s'\",;]+\.(?:jpg|jpeg|png|gif|tif|tiff|webp|raw|cr2|nef|arw|bin|pth|pt|ckpt|safetensors)\b"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Supported model id, such as topiq_nr or clipiqa")
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
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


def main() -> None:
    args = build_parser().parse_args()
    report_path = args.report_path or (args.data_dir / "model-smoke-report.json")
    try:
        if args.offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            _disable_network()

        from shotsieve.model_assets import apply_model_cache_dir, prepare_model

        apply_model_cache_dir(args.cache_dir)
        record = prepare_model(args.model, data_dir=args.data_dir)
        if record.get("state") != "prepared" or record.get("tested_runtime") != "cpu":
            raise RuntimeError("Model smoke did not produce a prepared CPU record.")

        report = {
            "status": "passed",
            "model": record.get("model"),
            "state": record.get("state"),
            "tested_runtime": record.get("tested_runtime"),
            "processed_counts": record.get("processed_counts"),
            "cache_paths": record.get("cache_paths"),
            "dependency_versions": installed_model_dependency_versions(),
        }
        _write_report(report_path, report)
        print(json.dumps(report, sort_keys=True))
    except Exception as exc:
        diagnostic = build_model_diagnostic(
            exc,
            phase="validating_initialization",
            model_name=args.model,
            requested_runtime="cpu",
            cache_dir=args.cache_dir,
        )
        report = {
            "status": "failed",
            "model": args.model,
            "dependency_versions": installed_model_dependency_versions(),
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
