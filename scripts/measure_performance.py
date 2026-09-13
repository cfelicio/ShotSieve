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
from typing import Callable, TypeVar

from shotsieve.config import DEFAULT_SUPPORTED_EXTENSIONS, HEIF_EXTENSIONS, RAW_CAMERA_EXTENSIONS
from shotsieve.db import database, initialize_database, root_path_filter
from shotsieve.performance import explain_query_plan, monotonic_seconds
from shotsieve.preview import generate_preview
from shotsieve.review import count_review_files, list_review_files, review_overview, review_selection_revision
from shotsieve.review_filters import SORT_ORDERS, _build_review_browser_where
from shotsieve.scanner import scan_root
from shotsieve.scoring import count_score_rows, fetch_score_rows


_T = TypeVar("_T")
_STANDARD_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff"})
_REVIEW_PAGE_SIZE = 60
_REVIEW_NAVIGATION_SORTS = ("score_desc", "path", "date_desc")
_REVIEW_FILTER_CASES: tuple[tuple[str, dict[str, object]], ...] = (
    ("score_band", {"min_score": 25.0, "max_score": 75.0}),
    ("jpeg", {"formats": ["jpeg"]}),
    ("unmarked", {"marked": "none"}),
    ("metadata_valid", {"metadata": "valid"}),
    ("issues", {"issues": "issues"}),
    ("path_search", {"query": "."}),
)


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


def _review_page_offsets(total: int) -> dict[str, int]:
    last_offset = max(0, total - _REVIEW_PAGE_SIZE)
    return {
        "early": 0,
        "middle": max(0, (total // 2) - (_REVIEW_PAGE_SIZE // 2)),
        "deep": last_offset,
    }


def _review_query_plan(
    connection: sqlite3.Connection,
    *,
    root: str | None,
    filter_options: dict[str, object],
    operation: str,
    sort: str | None = None,
    offset: int = 0,
) -> list[str]:
    where_clause, params = _build_review_browser_where(root=root, **filter_options)
    joins = """
        FROM files
        LEFT JOIN scores ON scores.file_id = files.id
        LEFT JOIN review_state ON review_state.file_id = files.id
    """
    if operation == "count":
        sql = f"SELECT COUNT(*) AS total {joins} {where_clause}"
    elif operation == "revision":
        sql = f"""
            SELECT
                COUNT(*) AS total,
                COALESCE(MIN(files.id), 0) AS min_id,
                COALESCE(MAX(files.id), 0) AS max_id,
                COALESCE(SUM(files.id), 0) AS sum_id,
                COALESCE(SUM(files.id * files.id), 0) AS sum_sq_id
            {joins}
            {where_clause}
        """
    elif operation == "list":
        if sort not in SORT_ORDERS:
            raise ValueError(f"Unsupported performance sort: {sort}")
        sql = f"""
            SELECT files.id, files.path, files.format, files.preview_status, files.preview_path,
                   files.width, files.height, files.size_bytes, files.capture_time, files.last_error,
                   scores.overall_score,
                   scores.learned_backend, scores.learned_score_normalized, scores.learned_confidence,
                   COALESCE(review_state.decision_state, 'pending') AS decision_state,
                   COALESCE(review_state.delete_marked, 0) AS delete_marked,
                   COALESCE(review_state.export_marked, 0) AS export_marked,
                   review_state.updated_time
            {joins}
            {where_clause}
            ORDER BY {SORT_ORDERS[sort]}
            LIMIT ? OFFSET ?
        """
        params.extend([_REVIEW_PAGE_SIZE, offset])
    else:
        raise ValueError(f"Unsupported performance operation: {operation}")
    return explain_query_plan(connection, sql, params)


def _measure_review_query(
    connection: sqlite3.Connection,
    *,
    measurements: list[dict[str, object]],
    query_plans: list[dict[str, object]],
    scope: str,
    root: str | None,
    filter_name: str,
    filter_options: dict[str, object],
    operation: str,
    sort: str | None = None,
    page: str | None = None,
    offset: int = 0,
) -> object:
    query_id_parts = [scope, filter_name, operation]
    if sort:
        query_id_parts.append(sort)
    if page:
        query_id_parts.append(page)
    query_id = ".".join(query_id_parts)
    query_plans.append({
        "query_id": query_id,
        "scope": scope,
        "filter": filter_name,
        "operation": operation,
        "sort": sort,
        "page": page,
        "details": _review_query_plan(
            connection,
            root=root,
            filter_options=filter_options,
            operation=operation,
            sort=sort,
            offset=offset,
        ),
    })

    if operation == "count":
        def action() -> object:
            return count_review_files(connection, root=root, **filter_options)
    elif operation == "revision":
        def action() -> object:
            return review_selection_revision(
                connection,
                scope="review-browser",
                root=root,
                **filter_options,
            )
    elif operation == "list":
        def action() -> object:
            return list_review_files(
                connection,
                root=root,
                sort=sort or "score_desc",
                limit=_REVIEW_PAGE_SIZE,
                offset=offset,
                **filter_options,
            )
    else:
        raise ValueError(f"Unsupported performance operation: {operation}")

    started_at = monotonic_seconds()
    result = action()
    measurement: dict[str, object] = {
        "query_id": query_id,
        "scope": scope,
        "filter": filter_name,
        "operation": operation,
        "sort": sort,
        "page": page,
        "offset": offset if operation == "list" else None,
        "elapsed_ms": round((monotonic_seconds() - started_at) * 1000, 3),
    }
    if operation == "count":
        measurement["result_count"] = int(result)
    elif operation == "revision":
        measurement["revision_present"] = bool(result)
    else:
        measurement["result_count"] = len(result)
    measurements.append(measurement)
    return result


def _measure_review_navigation(
    connection: sqlite3.Connection,
    *,
    active_root: Path,
    active_total: int,
    catalog_total: int,
) -> dict[str, object]:
    measurements: list[dict[str, object]] = []
    query_plans: list[dict[str, object]] = []
    page_offsets = {
        "active": _review_page_offsets(active_total),
        "global": _review_page_offsets(catalog_total),
    }
    scopes = (("global", None), ("active", str(active_root)))

    for scope, root in scopes:
        _measure_review_query(
            connection,
            measurements=measurements,
            query_plans=query_plans,
            scope=scope,
            root=root,
            filter_name="all",
            filter_options={},
            operation="count",
        )
        _measure_review_query(
            connection,
            measurements=measurements,
            query_plans=query_plans,
            scope=scope,
            root=root,
            filter_name="all",
            filter_options={},
            operation="revision",
        )
        for sort in _REVIEW_NAVIGATION_SORTS:
            for page, offset in page_offsets[scope].items():
                _measure_review_query(
                    connection,
                    measurements=measurements,
                    query_plans=query_plans,
                    scope=scope,
                    root=root,
                    filter_name="all",
                    filter_options={},
                    operation="list",
                    sort=sort,
                    page=page,
                    offset=offset,
                )
        for filter_name, filter_options in _REVIEW_FILTER_CASES:
            _measure_review_query(
                connection,
                measurements=measurements,
                query_plans=query_plans,
                scope=scope,
                root=root,
                filter_name=filter_name,
                filter_options=filter_options,
                operation="count",
            )
            _measure_review_query(
                connection,
                measurements=measurements,
                query_plans=query_plans,
                scope=scope,
                root=root,
                filter_name=filter_name,
                filter_options=filter_options,
                operation="revision",
            )
            _measure_review_query(
                connection,
                measurements=measurements,
                query_plans=query_plans,
                scope=scope,
                root=root,
                filter_name=filter_name,
                filter_options=filter_options,
                operation="list",
                sort="score_desc",
                page="early",
            )

    return {
        "page_size": _REVIEW_PAGE_SIZE,
        "page_offsets": page_offsets,
        "sorts": list(_REVIEW_NAVIGATION_SORTS),
        "filter_cases": [name for name, _ in _REVIEW_FILTER_CASES],
        "timings": measurements,
        "query_plans": query_plans,
    }


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
        review_navigation = _measure(
            timings_ms,
            "review_navigation",
            lambda: _measure_review_navigation(
                connection,
                active_root=source_root,
                active_total=review_total,
                catalog_total=int(overview["catalog"]["total_files"]),
            ),
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
            (*root_params, _REVIEW_PAGE_SIZE),
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
        "schema": "shotsieve-performance-measurement-v2",
        "environment": {
            "machine": platform.node() or "unknown",
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
        "review_navigation": review_navigation,
        "preview_samples": preview_results,
        "limitations": [
            "Metadata scans do not generate previews.",
            "Learned-IQA model startup and inference are not run by this utility.",
            "RAW and HEIF preview results depend on locally installed optional loaders.",
            "Navigation timings are comparative measurements for this machine and catalog state; no absolute CI threshold is implied.",
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
