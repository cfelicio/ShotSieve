from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

from shotsieve.db import normalize_resolved_path
from shotsieve.performance import log_duration, monotonic_seconds

from shotsieve.review_filters import (
    SORT_ORDERS,
    VALID_DECISION_STATES,
    _build_after_id_filter,
    _build_file_filters,
    _build_format_filters,
    _build_issue_filters,
    _build_metadata_status_filters,
    _build_resolution_filters,
    _build_review_browser_where,
    _build_review_state_filters,
    _build_score_filters,
    _build_size_filters,
    _compile_where_clause,
)
from shotsieve.review_cache import (
    _PRUNE_MISSING_CACHE_BATCH_SIZE,
    _allow_legacy_preview_path_fallback,
    _is_within_dir,
    _resolve_ready_preview_path,
    _resolve_source_path_within_roots,
    _trusted_delete_roots,
    clear_cache_scope,
    delete_files,
    media_path_for_file,
    normalize_file_ids,
    prune_missing_cache_entries,
    remove_files_from_cache,
)

__all__ = [
    "SORT_ORDERS",
    "ThreadPoolExecutor",
    "VALID_DECISION_STATES",
    "_PRUNE_MISSING_CACHE_BATCH_SIZE",
    "_allow_legacy_preview_path_fallback",
    "_build_after_id_filter",
    "_build_file_filters",
    "_build_format_filters",
    "_build_issue_filters",
    "_build_metadata_status_filters",
    "_build_resolution_filters",
    "_build_review_browser_where",
    "_build_review_state_filters",
    "_build_score_filters",
    "_build_size_filters",
    "_compile_where_clause",
    "_is_within_dir",
    "_resolve_ready_preview_path",
    "_resolve_source_path_within_roots",
    "_review_summary",
    "_trusted_delete_roots",
    "clear_cache_scope",
    "count_review_files",
    "delete_files",
    "get_review_file_detail",
    "list_analysis_diagnostics",
    "list_review_browser_file_ids",
    "list_review_files",
    "list_review_state_file_ids",
    "list_roots",
    "list_scan_runs",
    "media_path_for_file",
    "normalize_file_ids",
    "prune_missing_cache_entries",
    "remove_files_from_cache",
    "review_overview",
    "review_selection_revision",
    "update_review_state",
    "update_review_state_batch",
]


def _review_summary(connection, *, root: str | None = None) -> dict[str, int]:
    where_clause, params = _compile_where_clause(_build_file_filters(root=root))
    counts = connection.execute(
        f"""
        SELECT
            COUNT(files.id) AS total_files,
            COUNT(scores.file_id) AS scored_files,
            SUM(COALESCE(review_state.delete_marked, 0)) AS delete_marked,
            SUM(COALESCE(review_state.export_marked, 0)) AS export_marked
        FROM files
        LEFT JOIN scores ON scores.file_id = files.id
        LEFT JOIN review_state ON review_state.file_id = files.id
        {where_clause}
        """,
        tuple(params),
    ).fetchone()

    return {
        "total_files": int(counts["total_files"] or 0),
        "scored_files": int(counts["scored_files"] or 0),
        "delete_marked": int(counts["delete_marked"] or 0),
        "export_marked": int(counts["export_marked"] or 0),
    }


def review_overview(connection, *, root: str | None = None) -> dict[str, object]:
    """Return explicit active-library and catalog totals without splitting the cache."""
    started_at = monotonic_seconds()
    catalog = _review_summary(connection)
    root_stripped = root.strip() if root else None
    if root_stripped:
        active_library = _review_summary(connection, root=root_stripped)
        resolved_roots = [str(Path(p).expanduser().resolve()) for p in root_stripped.split("|") if p.strip()]
        active_root = "|".join(resolved_roots) if resolved_roots else None
    else:
        active_library = {
            "total_files": 0,
            "scored_files": 0,
            "delete_marked": 0,
            "export_marked": 0,
        }
        active_root = None

    log_duration(
        "review.overview",
        started_at,
        scope="root" if root_stripped else "catalog",
        total_files=active_library["total_files"],
    )

    return {
        # Keep summary for existing callers. New clients should use the explicit
        # active_library and catalog fields so their labels cannot be ambiguous.
        "summary": active_library if root_stripped else catalog,
        "active_library": {"root": active_root, **active_library},
        "catalog": catalog,
        "roots": list_roots(connection),
        "scan_runs": list_scan_runs(connection),
    }


def list_roots(connection) -> list[str]:
    rows = connection.execute(
        "SELECT DISTINCT root_path FROM scan_runs ORDER BY root_path ASC"
    ).fetchall()
    return [row["root_path"] for row in rows]


def list_scan_runs(connection, *, limit: int = 6) -> list[dict[str, object]]:
    rows = connection.execute(
        """
        SELECT root_path, started_time, completed_time, status,
               files_seen, files_added, files_updated, files_unchanged, files_removed, error_text
        FROM scan_runs
        ORDER BY started_time DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [dict(row) for row in rows]


def list_analysis_diagnostics(
    connection,
    *,
    root: str | None = None,
    limit: int = 100,
) -> dict[str, object]:
    """Return actionable reasons for files that do not currently have a score."""
    conditions, params = _build_file_filters(root=root)
    conditions.append("scores.file_id IS NULL")
    where_clause = "WHERE " + " AND ".join(conditions)
    bounded_limit = max(1, min(int(limit), 500))

    rows = connection.execute(
        f"""
        SELECT files.path, files.format, files.preview_status, files.last_error,
               files.analysis_status, files.analysis_error, files.last_analysis_time
        FROM files
        LEFT JOIN scores ON scores.file_id = files.id
        {where_clause}
        ORDER BY files.last_analysis_time DESC, files.last_scan_time DESC, files.id ASC
        LIMIT ?
        """,
        (*params, bounded_limit),
    ).fetchall()
    total_row = connection.execute(
        f"""
        SELECT COUNT(files.id) AS row_count
        FROM files
        LEFT JOIN scores ON scores.file_id = files.id
        {where_clause}
        """,
        tuple(params),
    ).fetchone()

    items: list[dict[str, object]] = []
    for row in rows:
        status = str(row["analysis_status"] or "").strip() or "pending"
        error = str(row["analysis_error"] or "").strip()
        if row["preview_status"] == "failed":
            detail = str(row["last_error"] or "").strip() or "Preview generation failed."
            error = f"Preview generation failed: {detail}"
            status = "failed"
        if not error and status == "skipped":
            error = "No compatible analysis image is available."
        if not error and status == "failed":
            error = "Analysis failed without an error detail."
        if not error:
            error = "Not analyzed yet. Run Analyze or Re-Score."
        items.append({
            "path": row["path"],
            "format": row["format"],
            "status": status,
            "error": error,
            "last_analysis_time": row["last_analysis_time"],
        })

    return {"total": int(total_row["row_count"] or 0), "items": items}


def count_review_files(
    connection,
    *,
    root: str | None = None,
    marked: str = "all",
    issues: str = "all",
    query: str | None = None,
    min_score: float | None = None,
    max_score: float | None = None,
    formats: list[str] | None = None,
    min_mp: float | None = None,
    max_mp: float | None = None,
    min_width: int | None = None,
    max_width: int | None = None,
    min_height: int | None = None,
    max_height: int | None = None,
    min_edge: int | None = None,
    max_edge: int | None = None,
    min_size: int | None = None,
    max_size: int | None = None,
    metadata: str = "all",
) -> int:
    """Return the total count of files matching the score-backed review browser filters."""
    started_at = monotonic_seconds()
    where_clause, params = _build_review_browser_where(
        root=root, marked=marked, issues=issues,
        query=query, min_score=min_score, max_score=max_score,
        formats=formats, min_mp=min_mp, max_mp=max_mp,
        min_width=min_width, max_width=max_width,
        min_height=min_height, max_height=max_height,
        min_edge=min_edge, max_edge=max_edge,
        min_size=min_size, max_size=max_size,
        metadata=metadata,
    )
    sql = f"""
        SELECT COUNT(*) AS total
        FROM files
        LEFT JOIN scores ON scores.file_id = files.id
        LEFT JOIN review_state ON review_state.file_id = files.id
        {where_clause}
    """
    row = connection.execute(sql, tuple(params)).fetchone()
    total = int(row["total"] or 0)
    log_duration(
        "review.count",
        started_at,
        scope="root" if root else "catalog",
        marked=marked,
        issues=issues,
        has_query=bool(query),
        total=total,
    )
    return total


def review_selection_revision(
    connection,
    *,
    scope: str,
    root: str | None = None,
    marked: str = "all",
    issues: str = "all",
    query: str | None = None,
    min_score: float | None = None,
    max_score: float | None = None,
    formats: list[str] | None = None,
    min_mp: float | None = None,
    max_mp: float | None = None,
    min_width: int | None = None,
    max_width: int | None = None,
    min_height: int | None = None,
    max_height: int | None = None,
    min_edge: int | None = None,
    max_edge: int | None = None,
    min_size: int | None = None,
    max_size: int | None = None,
    metadata: str = "all",
) -> str:
    started_at = monotonic_seconds()
    if scope == "review-browser":
        where_clause, params = _build_review_browser_where(
            root=root,
            marked=marked,
            issues=issues,
            query=query,
            min_score=min_score,
            max_score=max_score,
            formats=formats,
            min_mp=min_mp,
            max_mp=max_mp,
            min_width=min_width,
            max_width=max_width,
            min_height=min_height,
            max_height=max_height,
            min_edge=min_edge,
            max_edge=max_edge,
            min_size=min_size,
            max_size=max_size,
            metadata=metadata,
        )
        joins = """
            LEFT JOIN scores ON scores.file_id = files.id
            LEFT JOIN review_state ON review_state.file_id = files.id
        """
    elif scope == "review-state":
        where_clause, params = _compile_where_clause(
            _build_file_filters(root=root, query=query),
            _build_review_state_filters(marked=marked),
        )
        joins = "LEFT JOIN review_state ON review_state.file_id = files.id"
    else:
        raise ValueError("scope must be one of: review-browser, review-state")

    row = connection.execute(
        f"""
        SELECT
            COUNT(*) AS total,
            COALESCE(MIN(files.id), 0) AS min_id,
            COALESCE(MAX(files.id), 0) AS max_id,
            COALESCE(SUM(files.id), 0) AS sum_id,
            COALESCE(SUM(files.id * files.id), 0) AS sum_sq_id
        FROM files
        {joins}
        {where_clause}
        """,
        tuple(params),
    ).fetchone()
    scope_key = "catalog"
    if root:
        resolved_roots = [normalize_resolved_path(Path(p).expanduser().resolve()) for p in root.split("|") if p.strip()]
        scope_key = "|".join(resolved_roots) if resolved_roots else "catalog"
    revision = "|".join(
        str(value or 0)
        for value in (
            scope,
            scope_key,
            row["total"],
            row["min_id"],
            row["max_id"],
            row["sum_id"],
            row["sum_sq_id"],
        )
    )
    log_duration(
        "review.selection_revision",
        started_at,
        scope=scope,
        root_scoped=bool(root),
        marked=marked,
    )
    return revision


def list_review_files(
    connection,
    *,
    root: str | None = None,
    sort: str = "score_asc",
    marked: str = "all",
    issues: str = "all",
    query: str | None = None,
    min_score: float | None = None,
    max_score: float | None = None,
    formats: list[str] | None = None,
    min_mp: float | None = None,
    max_mp: float | None = None,
    min_width: int | None = None,
    max_width: int | None = None,
    min_height: int | None = None,
    max_height: int | None = None,
    min_edge: int | None = None,
    max_edge: int | None = None,
    min_size: int | None = None,
    max_size: int | None = None,
    metadata: str = "all",
    limit: int = 60,
    offset: int = 0,
) -> list[dict[str, object]]:
    started_at = monotonic_seconds()
    order_by = SORT_ORDERS.get(sort, SORT_ORDERS["score_asc"])
    where_clause, params = _build_review_browser_where(
        root=root, marked=marked, issues=issues,
        query=query, min_score=min_score, max_score=max_score,
        formats=formats, min_mp=min_mp, max_mp=max_mp,
        min_width=min_width, max_width=max_width,
        min_height=min_height, max_height=max_height,
        min_edge=min_edge, max_edge=max_edge,
        min_size=min_size, max_size=max_size,
        metadata=metadata,
    )
    sql_parts = [
        """
        SELECT files.id, files.path, files.format, files.preview_status, files.preview_path,
               files.width, files.height, files.size_bytes, files.capture_time, files.last_error,
               scores.overall_score,
               scores.learned_backend, scores.learned_score_normalized, scores.learned_confidence,
               COALESCE(review_state.decision_state, 'pending') AS decision_state,
               COALESCE(review_state.delete_marked, 0) AS delete_marked,
               COALESCE(review_state.export_marked, 0) AS export_marked,
               review_state.updated_time
        FROM files
        LEFT JOIN scores ON scores.file_id = files.id
        LEFT JOIN review_state ON review_state.file_id = files.id
        """,
        where_clause,
    ]

    sql_parts.append(f"ORDER BY {order_by}")
    sql_parts.append("LIMIT ? OFFSET ?")
    params.extend([limit, offset])

    rows = connection.execute(" ".join(sql_parts), tuple(params)).fetchall()
    payload = [dict(row) for row in rows]
    log_duration(
        "review.list",
        started_at,
        scope="root" if root else "catalog",
        sort=sort if sort in SORT_ORDERS else "score_asc",
        offset=offset,
        limit=limit,
        result_count=len(payload),
    )
    return payload


def list_review_browser_file_ids(
    connection,
    *,
    root: str | None = None,
    marked: str = "all",
    issues: str = "all",
    query: str | None = None,
    min_score: float | None = None,
    max_score: float | None = None,
    formats: list[str] | None = None,
    min_mp: float | None = None,
    max_mp: float | None = None,
    min_width: int | None = None,
    max_width: int | None = None,
    min_height: int | None = None,
    max_height: int | None = None,
    min_edge: int | None = None,
    max_edge: int | None = None,
    min_size: int | None = None,
    max_size: int | None = None,
    metadata: str = "all",
    limit: int | None = None,
    after_id: int | None = None,
) -> list[int]:
    """Return score-backed review browser ids in ascending keyset order."""
    where_clause, params = _compile_where_clause(
        _build_file_filters(root=root, query=query),
        _build_score_filters(require_scored=True, min_score=min_score, max_score=max_score),
        _build_review_state_filters(marked=marked),
        _build_issue_filters(issues=issues),
        _build_format_filters(formats=formats),
        _build_resolution_filters(
            min_mp=min_mp, max_mp=max_mp,
            min_width=min_width, max_width=max_width,
            min_height=min_height, max_height=max_height,
            min_edge=min_edge, max_edge=max_edge,
        ),
        _build_size_filters(min_size=min_size, max_size=max_size),
        _build_metadata_status_filters(metadata=metadata),
        _build_after_id_filter(after_id=after_id),
    )
    sql = f"""
        SELECT files.id
        FROM files
        LEFT JOIN scores ON scores.file_id = files.id
        LEFT JOIN review_state ON review_state.file_id = files.id
        {where_clause}
        ORDER BY files.id ASC
    """
    if limit is not None:
        sql = f"{sql}\n        LIMIT ?"
        params.append(limit)
    rows = connection.execute(sql, tuple(params)).fetchall()
    return [int(row["id"]) for row in rows]


def list_review_state_file_ids(
    connection,
    *,
    marked: str,
    root: str | None = None,
    query: str | None = None,
    limit: int | None = None,
    offset: int = 0,
    after_id: int | None = None,
) -> list[int]:
    """Return file ids filtered by user review-state only, without requiring score rows."""
    where_clause, params = _compile_where_clause(
        _build_file_filters(root=root, query=query),
        _build_review_state_filters(marked=marked),
        _build_after_id_filter(after_id=after_id),
    )
    sql = f"""
        SELECT files.id
        FROM files
        LEFT JOIN review_state ON review_state.file_id = files.id
        {where_clause}
        ORDER BY files.id ASC
    """
    if limit is not None:
        if after_id is not None and after_id > 0:
            sql = f"{sql}\n        LIMIT ?"
            params.append(limit)
        else:
            sql = f"{sql}\n        LIMIT ? OFFSET ?"
            params.extend([limit, offset])
    rows = connection.execute(sql, tuple(params)).fetchall()
    return [int(row["id"]) for row in rows]


def get_review_file_detail(connection, file_id: int) -> dict[str, object] | None:
    row = connection.execute(
        """
        SELECT files.id, files.path, files.format, files.preview_status, files.preview_path,
               files.width, files.height, files.size_bytes, files.capture_time, files.last_error,
               scores.overall_score,
               scores.learned_backend, scores.learned_raw_score, scores.learned_score_normalized,
               scores.learned_confidence,
               COALESCE(review_state.decision_state, 'pending') AS decision_state,
               COALESCE(review_state.delete_marked, 0) AS delete_marked,
               COALESCE(review_state.export_marked, 0) AS export_marked,
               review_state.updated_time
        FROM files
        LEFT JOIN scores ON scores.file_id = files.id
        LEFT JOIN review_state ON review_state.file_id = files.id
        WHERE files.id = ?
        """,
        (file_id,),
    ).fetchone()
    return dict(row) if row is not None else None


def update_review_state(
    connection,
    *,
    file_id: int,
    decision_state: str | None = None,
    delete_marked: bool | None = None,
    export_marked: bool | None = None,
    updated_time: str,
) -> None:
    if file_id <= 0:
        raise ValueError("file_id must be a positive integer")

    if decision_state is not None and decision_state not in VALID_DECISION_STATES:
        raise ValueError(f"decision_state must be one of: {', '.join(sorted(VALID_DECISION_STATES))}")

    if delete_marked is True and export_marked is True:
        raise ValueError("delete_marked and export_marked cannot both be true")

    file_row = connection.execute(
        "SELECT 1 FROM files WHERE id = ?",
        (file_id,),
    ).fetchone()
    if file_row is None:
        raise ValueError("file_id does not exist in the cache")

    existing = connection.execute(
        "SELECT decision_state, delete_marked, export_marked FROM review_state WHERE file_id = ?",
        (file_id,),
    ).fetchone()

    final_decision_state = decision_state if decision_state is not None else (existing["decision_state"] if existing else "pending")
    final_delete_marked = int(delete_marked if delete_marked is not None else (existing["delete_marked"] if existing else 0))
    final_export_marked = int(export_marked if export_marked is not None else (existing["export_marked"] if existing else 0))

    if final_delete_marked and final_export_marked:
        raise ValueError("delete_marked and export_marked cannot both be true")

    connection.execute(
        """
        INSERT INTO review_state(file_id, decision_state, delete_marked, export_marked, updated_time)
        VALUES(?, ?, ?, ?, ?)
        ON CONFLICT(file_id) DO UPDATE SET
            decision_state = excluded.decision_state,
            delete_marked = excluded.delete_marked,
            export_marked = excluded.export_marked,
            updated_time = excluded.updated_time
        """,
        (
            file_id,
            final_decision_state,
            final_delete_marked,
            final_export_marked,
            updated_time,
        ),
    )


def update_review_state_batch(
    connection,
    *,
    file_ids: Iterable[int],
    decision_state: str | None = None,
    delete_marked: bool | None = None,
    export_marked: bool | None = None,
    updated_time: str,
) -> int:
    normalized_ids = normalize_file_ids(file_ids)
    if decision_state is not None and decision_state not in VALID_DECISION_STATES:
        raise ValueError(f"decision_state must be one of: {', '.join(sorted(VALID_DECISION_STATES))}")

    if delete_marked is True and export_marked is True:
        raise ValueError("delete_marked and export_marked cannot both be true")

    placeholders = ",".join("?" for _ in normalized_ids)

    existing_file_ids = {
        int(row["id"])
        for row in connection.execute(
            f"SELECT id FROM files WHERE id IN ({placeholders})",
            tuple(normalized_ids),
        ).fetchall()
    }
    if len(existing_file_ids) != len(normalized_ids):
        raise ValueError("One or more file_ids do not exist in the cache")

    existing_rows = connection.execute(
        f"SELECT file_id, decision_state, delete_marked, export_marked FROM review_state WHERE file_id IN ({placeholders})",
        tuple(normalized_ids),
    ).fetchall()
    existing_by_id = {int(row["file_id"]): row for row in existing_rows}

    upserts: list[tuple[int, str, int, int, str]] = []
    for file_id in normalized_ids:
        existing = existing_by_id.get(file_id)
        final_decision_state = decision_state if decision_state is not None else (existing["decision_state"] if existing else "pending")
        final_delete_marked = int(delete_marked if delete_marked is not None else (existing["delete_marked"] if existing else 0))
        final_export_marked = int(export_marked if export_marked is not None else (existing["export_marked"] if existing else 0))

        if final_delete_marked and final_export_marked:
            raise ValueError("delete_marked and export_marked cannot both be true")

        upserts.append((file_id, final_decision_state, final_delete_marked, final_export_marked, updated_time))

    connection.executemany(
        """
        INSERT INTO review_state(file_id, decision_state, delete_marked, export_marked, updated_time)
        VALUES(?, ?, ?, ?, ?)
        ON CONFLICT(file_id) DO UPDATE SET
            decision_state = excluded.decision_state,
            delete_marked = excluded.delete_marked,
            export_marked = excluded.export_marked,
            updated_time = excluded.updated_time
        """,
        upserts,
    )

    return len(normalized_ids)