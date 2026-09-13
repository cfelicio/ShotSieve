"""Opt-in performance baseline for deep Review navigation.

Run manually with:

    SHOTSIEVE_RUN_PERFORMANCE_BASELINE=1 python -m pytest tests/test_performance_baseline.py -s -q

The fixture intentionally inserts database rows directly. It measures catalog and
Review query behavior, not filesystem traversal, preview generation, or IQA model
startup. The report contains separate timings and query plans so a slow page can
be compared with the work SQLite selected for that page.
"""
from __future__ import annotations

import json
import os
import platform
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import pytest

from shotsieve.db import database, initialize_database, normalize_path_case, root_path_filter
from shotsieve.performance import explain_query_plan, monotonic_seconds
from shotsieve.review import count_review_files, list_review_files, review_overview, review_selection_revision
from shotsieve.review_filters import SORT_ORDERS, _build_review_browser_where
from shotsieve.scoring import count_score_rows, fetch_score_rows


_PERFORMANCE_BASELINE_ENV = "SHOTSIEVE_RUN_PERFORMANCE_BASELINE"
_ARCHIVE_FILE_COUNT = 10_000
_ACTIVE_ROOT_FILE_COUNT = 100_000
_CATALOG_FILE_COUNT = _ARCHIVE_FILE_COUNT + _ACTIVE_ROOT_FILE_COUNT
_PAGE_SIZE = 60


def _page_offsets(total: int) -> dict[str, int]:
    return {
        "early": 0,
        "middle": max(0, (total // 2) - (_PAGE_SIZE // 2)),
        "deep": max(0, total - _PAGE_SIZE),
    }


_PAGE_OFFSETS = {
    "active": _page_offsets(_ACTIVE_ROOT_FILE_COUNT),
    "global": _page_offsets(_CATALOG_FILE_COUNT),
}
_NAVIGATION_SORTS = ("score_desc", "path", "date_desc")
_FILTER_CASES: tuple[tuple[str, dict[str, object]], ...] = (
    ("score_band", {"min_score": 25.0, "max_score": 75.0}),
    ("jpeg", {"formats": ["jpeg"]}),
    ("rejected", {"marked": "delete"}),
    ("metadata_unknown", {"metadata": "unknown"}),
    ("path_search", {"query": "asset-01"}),
)
_T = TypeVar("_T")

pytestmark = pytest.mark.skipif(
    os.environ.get(_PERFORMANCE_BASELINE_ENV, "").strip().casefold() not in {"1", "true", "yes", "on"},
    reason=f"set {_PERFORMANCE_BASELINE_ENV}=1 to run the opt-in deep-navigation baseline",
)


def _measure(timings_ms: dict[str, float], label: str, action: Callable[[], _T]) -> _T:
    started_at = monotonic_seconds()
    result = action()
    timings_ms[label] = round((monotonic_seconds() - started_at) * 1000, 3)
    return result


def _fixture_file_values(root: Path, file_number: int) -> tuple[object, ...]:
    extension = ("jpg", "png", "tif", "heic")[file_number % 4]
    path_text = str(root / f"asset-{file_number:06d}.{extension}")
    has_unknown_metadata = file_number % 997 == 0
    return (
        path_text,
        normalize_path_case(path_text),
        2_000_000 + file_number,
        1_700_000_000.0 + file_number,
        extension,
        None if has_unknown_metadata else 6000 + file_number % 3,
        None if has_unknown_metadata else 4000 + file_number % 5,
        None if file_number % 503 == 0 else f"2024-01-{(file_number % 28) + 1:02d}T00:00:00+00:00",
        str(root / "previews" / f"asset-{file_number:06d}.jpg"),
        "ready",
        "2026-01-01T00:00:00+00:00",
        "benchmark warning" if file_number % 251 == 0 else None,
        "unchanged",
    )


def _insert_catalog_rows(connection, *, archive_root: Path, active_root: Path) -> None:
    def rows_for_root(root: Path, start: int, count: int):
        for index in range(start, start + count):
            yield _fixture_file_values(root, index + 1)

    insert_files_sql = """
        INSERT INTO files(
            path, path_key, size_bytes, modified_time, format,
            width, height, capture_time, preview_path, preview_status,
            last_scan_time, last_error, scan_status
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    connection.executemany(
        insert_files_sql,
        rows_for_root(archive_root, 0, _ARCHIVE_FILE_COUNT),
    )
    connection.executemany(
        insert_files_sql,
        rows_for_root(active_root, _ARCHIVE_FILE_COUNT, _ACTIVE_ROOT_FILE_COUNT),
    )
    connection.executemany(
        """
        INSERT INTO scores(
            file_id, overall_score, learned_backend, learned_raw_score,
            learned_score_normalized, learned_confidence, source_modified_time,
            source_size_bytes, preset_name, model_version, computed_time
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            (
                file_id,
                float(file_id % 101),
                "topiq_nr",
                0.82,
                float(file_id % 101),
                91.0,
                1_700_000_000.0 + file_id - 1,
                2_000_000 + file_id - 1,
                "learned-only",
                "learned:benchmark",
                "2026-01-01T00:00:00+00:00",
            )
            for file_id in range(1, _CATALOG_FILE_COUNT + 1)
        ),
    )
    connection.executemany(
        """
        INSERT INTO review_state(file_id, decision_state, delete_marked, export_marked, updated_time)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            (
                file_id,
                "delete" if file_id % 29 == 0 else "export",
                int(file_id % 29 == 0),
                int(file_id % 29 != 0 and file_id % 31 == 0),
                "2026-01-01T00:00:00+00:00",
            )
            for file_id in range(1, _CATALOG_FILE_COUNT + 1)
            if file_id % 29 == 0 or file_id % 31 == 0
        ),
    )


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
            raise ValueError(f"Unsupported benchmark sort: {sort}")
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
        params.extend([_PAGE_SIZE, offset])
    else:
        raise ValueError(f"Unsupported benchmark operation: {operation}")
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
                limit=_PAGE_SIZE,
                offset=offset,
                **filter_options,
            )
    else:
        raise ValueError(f"Unsupported benchmark operation: {operation}")

    started_at = monotonic_seconds()
    result = action()
    elapsed_ms = round((monotonic_seconds() - started_at) * 1000, 3)
    measurement: dict[str, object] = {
        "query_id": query_id,
        "scope": scope,
        "filter": filter_name,
        "operation": operation,
        "sort": sort,
        "page": page,
        "offset": offset if operation == "list" else None,
        "elapsed_ms": elapsed_ms,
    }
    if operation == "count":
        measurement["result_count"] = int(result)
    elif operation == "revision":
        measurement["revision_present"] = bool(result)
    else:
        measurement["result_count"] = len(result)
    measurements.append(measurement)
    return result


def _measure_review_navigation(connection, *, active_root: Path) -> dict[str, object]:
    measurements: list[dict[str, object]] = []
    query_plans: list[dict[str, object]] = []
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
        for sort in _NAVIGATION_SORTS:
            for page, offset in _PAGE_OFFSETS[scope].items():
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
        for filter_name, filter_options in _FILTER_CASES:
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
        "page_size": _PAGE_SIZE,
        "page_offsets": _PAGE_OFFSETS,
        "sorts": list(_NAVIGATION_SORTS),
        "filter_cases": [name for name, _ in _FILTER_CASES],
        "timings": measurements,
        "query_plans": query_plans,
    }


def test_deep_review_navigation_remains_measurable_with_100000_active_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    archive_root = (tmp_path / "archive-catalog").resolve()
    active_root = (tmp_path / "active-library").resolve()
    archive_root.mkdir()
    active_root.mkdir()
    initialize_database(db_path)
    timings_ms: dict[str, float] = {}

    with database(db_path) as connection:
        _measure(
            timings_ms,
            "fixture_insert",
            lambda: _insert_catalog_rows(connection, archive_root=archive_root, active_root=active_root),
        )
        catalog_overview = _measure(timings_ms, "catalog_overview", lambda: review_overview(connection))
        active_total = _measure(
            timings_ms,
            "active_review_count",
            lambda: count_review_files(connection, root=str(active_root)),
        )
        navigation = _measure_review_navigation(connection, active_root=active_root)
        active_score_total = _measure(
            timings_ms,
            "active_score_count",
            lambda: count_score_rows(connection, raw_root=str(active_root)),
        )
        score_rows = _measure(
            timings_ms,
            "active_score_fetch",
            lambda: fetch_score_rows(connection, raw_root=str(active_root), limit=100),
        )
        root_clause, root_params = root_path_filter("files.path_key", active_root)
        query_plan = explain_query_plan(
            connection,
            f"SELECT files.id FROM files WHERE {root_clause} ORDER BY files.id ASC LIMIT ?",
            (*root_params, _PAGE_SIZE),
        )

    report = {
        "schema": "shotsieve-performance-baseline-v2",
        "environment": {
            "machine": platform.node() or "unknown",
            "platform": platform.platform(),
            "python": platform.python_version(),
            "sqlite": sqlite3.sqlite_version,
        },
        "fixture": {
            "catalog_rows": _CATALOG_FILE_COUNT,
            "active_library_rows": _ACTIVE_ROOT_FILE_COUNT,
            "other_cached_rows": _ARCHIVE_FILE_COUNT,
            "privacy": "Synthetic metadata only; report excludes fixture paths and filenames.",
        },
        "setup_timings_ms": timings_ms,
        "review_navigation": navigation,
        "additional_query_plans": {
            "active_id_page": query_plan,
        },
    }
    print(json.dumps(report, sort_keys=True))

    summary = catalog_overview["summary"]
    assert summary["total_files"] == _CATALOG_FILE_COUNT
    assert summary["scored_files"] == _CATALOG_FILE_COUNT
    assert active_total == _ACTIVE_ROOT_FILE_COUNT
    assert active_score_total == _ACTIVE_ROOT_FILE_COUNT
    assert len(score_rows) == 100
    assert query_plan

    measurements = navigation["timings"]
    query_plans = navigation["query_plans"]
    assert len(measurements) == len(query_plans)
    assert all(measurement["elapsed_ms"] >= 0 for measurement in measurements)
    assert all(measurement["query_id"] == plan["query_id"] for measurement, plan in zip(measurements, query_plans))
    assert all(plan["details"] for plan in query_plans)
    assert all(
        measurement["result_count"] == _PAGE_SIZE
        for measurement in measurements
        if measurement["operation"] == "list"
        and measurement["filter"] == "all"
        and measurement["scope"] == "active"
    )
