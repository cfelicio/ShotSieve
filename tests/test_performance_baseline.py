"""Opt-in performance baseline for a large historical catalog.

Run manually with:

    SHOTSIEVE_RUN_PERFORMANCE_BASELINE=1 python -m pytest tests/test_performance_baseline.py -s -q

The fixture intentionally inserts database rows directly. It measures catalog and
active-root query behavior, not filesystem traversal, preview generation, or IQA
model startup.
"""
from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import pytest

from shotsieve.db import database, initialize_database, normalize_path_case, root_path_filter
from shotsieve.performance import explain_query_plan, monotonic_seconds
from shotsieve.review import count_review_files, list_review_files, review_overview, review_selection_revision
from shotsieve.scoring import count_score_rows, fetch_score_rows


_PERFORMANCE_BASELINE_ENV = "SHOTSIEVE_RUN_PERFORMANCE_BASELINE"
_CATALOG_FILE_COUNT = 60_000
_ACTIVE_ROOT_FILE_COUNT = 100
_T = TypeVar("_T")

pytestmark = pytest.mark.skipif(
    os.environ.get(_PERFORMANCE_BASELINE_ENV, "").strip().casefold() not in {"1", "true", "yes", "on"},
    reason=f"set {_PERFORMANCE_BASELINE_ENV}=1 to run the opt-in large-catalog baseline",
)


def _measure(timings_ms: dict[str, float], label: str, action: Callable[[], _T]) -> _T:
    started_at = monotonic_seconds()
    result = action()
    timings_ms[label] = round((monotonic_seconds() - started_at) * 1000, 3)
    return result


def _insert_catalog_rows(connection, *, archive_root: Path, active_root: Path) -> None:
    def rows_for_root(root: Path, start: int, count: int):
        for index in range(start, start + count):
            path_text = str(root / f"asset-{index:05d}.jpg")
            yield (
                path_text,
                normalize_path_case(path_text),
                2_000_000 + index,
                1_700_000_000.0 + index,
                "jpg",
                6000,
                4000,
                "2026-01-01T00:00:00+00:00",
                str(root / "previews" / f"asset-{index:05d}.jpg"),
                "ready",
                "2026-01-01T00:00:00+00:00",
                None,
                "unchanged",
            )

    archive_count = _CATALOG_FILE_COUNT - _ACTIVE_ROOT_FILE_COUNT
    connection.executemany(
        """
        INSERT INTO files(
            path, path_key, size_bytes, modified_time, format,
            width, height, capture_time, preview_path, preview_status,
            last_scan_time, last_error, scan_status
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows_for_root(archive_root, 0, archive_count),
    )
    connection.executemany(
        """
        INSERT INTO files(
            path, path_key, size_bytes, modified_time, format,
            width, height, capture_time, preview_path, preview_status,
            last_scan_time, last_error, scan_status
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows_for_root(active_root, archive_count, _ACTIVE_ROOT_FILE_COUNT),
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
                index,
                float(index % 101),
                "topiq_nr",
                0.82,
                float(index % 101),
                91.0,
                1_700_000_000.0 + index - 1,
                2_000_000 + index - 1,
                "learned-only",
                "learned:benchmark",
                "2026-01-01T00:00:00+00:00",
            )
            for index in range(1, _CATALOG_FILE_COUNT + 1)
        ),
    )


def test_active_root_queries_remain_bounded_with_60000_cached_rows(tmp_path: Path) -> None:
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
        active_rows = _measure(
            timings_ms,
            "active_review_page",
            lambda: list_review_files(
                connection,
                root=str(active_root),
                sort="score_desc",
                limit=60,
                offset=0,
            ),
        )
        revision = _measure(
            timings_ms,
            "active_selection_revision",
            lambda: review_selection_revision(connection, scope="review-browser", root=str(active_root)),
        )
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
            (*root_params, 60),
        )

    print(json.dumps({"timings_ms": timings_ms, "query_plan": query_plan}, sort_keys=True))

    summary = catalog_overview["summary"]
    assert summary["total_files"] == _CATALOG_FILE_COUNT
    assert summary["scored_files"] == _CATALOG_FILE_COUNT
    assert active_total == _ACTIVE_ROOT_FILE_COUNT
    assert len(active_rows) == 60
    assert revision
    assert active_score_total == _ACTIVE_ROOT_FILE_COUNT
    assert len(score_rows) == _ACTIVE_ROOT_FILE_COUNT
    assert query_plan
