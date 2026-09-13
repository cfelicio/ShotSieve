"""Prepare one supported learned model, including a fresh offline-process check."""
from __future__ import annotations

import argparse
import json
import os
import socket
from pathlib import Path


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
    return parser


def _disable_network() -> None:
    def blocked_connect(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("network disabled for offline model smoke")

    socket.socket.connect = blocked_connect  # type: ignore[method-assign]
    socket.socket.connect_ex = blocked_connect  # type: ignore[method-assign]
    socket.create_connection = blocked_connect  # type: ignore[assignment]
    socket.getaddrinfo = blocked_connect  # type: ignore[assignment]


def main() -> None:
    args = build_parser().parse_args()
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        _disable_network()

    from shotsieve.model_assets import apply_model_cache_dir, prepare_model

    apply_model_cache_dir(args.cache_dir)
    record = prepare_model(args.model, data_dir=args.data_dir)
    if record.get("state") != "prepared" or record.get("tested_runtime") != "cpu":
        raise SystemExit(f"Model smoke did not produce a prepared CPU record: {record}")
    print(json.dumps({
        "model": record.get("model"),
        "state": record.get("state"),
        "tested_runtime": record.get("tested_runtime"),
        "processed_counts": record.get("processed_counts"),
        "cache_paths": record.get("cache_paths"),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
