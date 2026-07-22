#!/usr/bin/env python
"""Measure local catalog, scan, and preview behavior without exposing photo paths."""
from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sqlite3
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, TypeVar

from shotsieve.config import DEFAULT_SUPPORTED_EXTENSIONS, HEIF_EXTENSIONS, RAW_CAMERA_EXTENSIONS
from shotsieve.db import database, initialize_database, root_path_filter
from shotsieve.performance import explain_query_plan, monotonic_seconds
from shotsieve.preview import generate_preview
from shotsieve.review import count_review_files, list_review_files, review_overview, review_selection_revision
from shotsieve.scanner import scan_root
from shotsieve.scoring import count_score_rows, fetch_score_rows


_T = TypeVar("_T")
_STANDARD_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff"})


def _measure(timings_ms: dict[str, float], label: str, action: Callable[[], _T]) -> _T:
    started_at = monotonic_seconds()
    result = action()
    timings_ms[label] = round((monotonic_seconds() - started_at) * 1000, 3)
    return result


def _capabilities() -> dict[str, bool]:
    return {
        "pyiqa": importlib.util.find_spec("pyiqa") is not None,
        "rawpy": importlib.util.find_spec("rawpy") is not None,
        "pillow_heif": importlib.util.find_spec("pillow_heif") is not None,
    }


def _preview_group(path: Path) -> str | None:
    suffix = path.suffix.casefold()
    if suffix in _STANDARD_EXTENSIONS:
        return "standard"
    if suffix in HEIF_EXTENSIONS:
        return "heif"
    if suffix in RAW_CAMERA_EXTENSIONS:
        return "raw"
    return None


def _preview_samples(root: Path, *, per_group: int) -> dict[str, list[Path]]:
    selected: dict[str, list[Path]] = {"standard": [], "heif": [], "raw": []}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        group = _preview_group(path)
        if group is None or len(selected[group]) >= per_group:
            continue
        selected[group].append(path)
        if all(len(paths) >= per_group for paths in selected.values()):
            break
    return selected


def _insert_query_scores(connection: sqlite3.Connection) -> None:
    """Seed disposable query scores so Review queries use the real scanned metadata."""
    connection.execute(
        """
        INSERT INTO scores(
            file_id, overall_score, learned_backend, learned_raw_score,
            learned_score_normalized, learned_confidence, source_modified_time,
            source_size_bytes, preset_name, model_version, computed_time
        )
        SELECT
            id,
            50.0,
            'benchmark-placeholder',
            0.5,
            50.0,
            100.0,
            modified_time,
            size_bytes,
            'benchmark-placeholder',
            'benchmark-placeholder',
            '1970-01-01T00:00:00+00:00'
        FROM files
        """
    )


def measure(root: Path, *, data_dir: Path, preview_samples_per_group: int) -> dict[str, object]:
    source_root = root.expanduser().resolve()
    benchmark_dir = data_dir.expanduser().resolve()
    db_path = benchmark_dir / "shotsieve.db"
    preview_dir = benchmark_dir / "previews"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    initialize_database(db_path)

    timings_ms: dict[str, float] = {}
    with database(db_path) as connection:
        cold_scan = _measure(
            timings_ms,
            "metadata_scan_cold",
            lambda: scan_root(
                connection,
                root=source_root,
                recursive=True,
                extensions=DEFAULT_SUPPORTED_EXTENSIONS,
                preview_dir=preview_dir,
                generate_previews=False,
            ),
        )
        warm_scan = _measure(
            timings_ms,
            "metadata_scan_warm",
            lambda: scan_root(
                connection,
                root=source_root,
                recursive=True,
                extensions=DEFAULT_SUPPORTED_EXTENSIONS,
                preview_dir=preview_dir,
                generate_previews=False,
            ),
        )
        _measure(timings_ms, "seed_disposable_review_scores", lambda: _insert_query_scores(connection))
        overview = _measure(timings_ms, "catalog_overview", lambda: review_overview(connection))
        review_total = _measure(
            timings_ms,
            "review_count",
            lambda: count_review_files(connection, root=str(source_root)),
        )
        review_page = _measure(
            timings_ms,
            "review_page",
            lambda: list_review_files(connection, root=str(source_root), sort="score_desc", limit=60),
        )
        review_revision = _measure(
            timings_ms,
            "review_selection_revision",
            lambda: review_selection_revision(connection, scope="review-browser", root=str(source_root)),
        )
        score_total = _measure(
            timings_ms,
            "score_row_count",
            lambda: count_score_rows(connection, raw_root=str(source_root)),
        )
        score_rows = _measure(
            timings_ms,
            "score_row_fetch",
            lambda: fetch_score_rows(connection, raw_root=str(source_root), limit=100),
        )
        root_clause, root_params = root_path_filter("files.path_key", source_root)
        query_plan = explain_query_plan(
            connection,
            f"SELECT files.id FROM files WHERE {root_clause} ORDER BY files.id ASC LIMIT ?",
            (*root_params, 60),
        )

    preview_results: dict[str, dict[str, object]] = {}
    for group, paths in _preview_samples(source_root, per_group=preview_samples_per_group).items():
        group_dir = preview_dir / "samples" / group
        group_timings: list[float] = []
        statuses: Counter[str] = Counter()
        dimensions_available = 0
        for path in paths:
            started_at = monotonic_seconds()
            result = generate_preview(path, group_dir)
            group_timings.append(round((monotonic_seconds() - started_at) * 1000, 3))
            statuses[result.status] += 1
            if result.width is not None and result.height is not None:
                dimensions_available += 1
        preview_results[group] = {
            "sample_count": len(paths),
            "status_counts": dict(sorted(statuses.items())),
            "dimensions_available": dimensions_available,
            "timings_ms": group_timings,
            "mean_ms": round(sum(group_timings) / len(group_timings), 3) if group_timings else None,
        }

    return {
        "schema": "shotsieve-performance-measurement-v1",
        "environment": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "sqlite": sqlite3.sqlite_version,
            "capabilities": _capabilities(),
        },
        "source": {
            "supported_extensions": list(DEFAULT_SUPPORTED_EXTENSIONS),
            "preview_samples_per_group": preview_samples_per_group,
            "privacy": "Report intentionally excludes source paths, filenames, and image metadata.",
        },
        "metadata_scan": {
            "cold": asdict(cold_scan),
            "warm": asdict(warm_scan),
        },
        "query_measurements": {
            "timings_ms": timings_ms,
            "overview_summary": overview["summary"],
            "review_total": review_total,
            "review_page_count": len(review_page),
            "review_selection_revision_present": bool(review_revision),
            "score_row_total": score_total,
            "score_row_fetch_count": len(score_rows),
            "query_plan": query_plan,
            "note": "Scores are disposable placeholders used only to exercise Review SQL; no learned-IQA inference is measured.",
        },
        "preview_samples": preview_results,
        "limitations": [
            "Metadata scans do not generate previews.",
            "Learned-IQA model startup and inference are not run by this utility.",
            "RAW and HEIF preview results depend on locally installed optional loaders.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Local photo folder to measure")
    parser.add_argument("--data-dir", type=Path, required=True, help="Disposable local database and preview directory")
    parser.add_argument("--output", type=Path, required=True, help="JSON report path")
    parser.add_argument("--preview-samples-per-group", type=int, default=3, help="Maximum standard/HEIF/RAW preview samples")
    args = parser.parse_args()

    if args.preview_samples_per_group < 1:
        parser.error("--preview-samples-per-group must be at least 1")
    if not args.root.exists() or not args.root.is_dir():
        parser.error("root must be an existing directory")

    report = measure(
        args.root,
        data_dir=args.data_dir,
        preview_samples_per_group=args.preview_samples_per_group,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
