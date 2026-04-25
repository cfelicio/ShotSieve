from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor

from shotsieve.config import BROWSER_SAFE_EXTENSIONS, PREVIEW_PRIORITY_EXTENSIONS, RAW_CAMERA_EXTENSIONS
from shotsieve.db import escape_like, infer_preview_cache_roots, normalize_resolved_path, preview_cache_root_is_claimed, root_path_filter
from shotsieve.preview import clear_preview_cache_dir, delete_managed_preview_file


VALID_DECISION_STATES = {"pending", "delete", "export"}


_PRUNE_MISSING_CACHE_BATCH_SIZE = 5000


SORT_ORDERS = {
    "score_asc": "scores.overall_score ASC, files.id ASC",
    "score_desc": "scores.overall_score DESC, files.id ASC",
    "learned_asc": "scores.learned_score_normalized ASC, scores.overall_score ASC, files.id ASC",
    "learned_desc": "scores.learned_score_normalized DESC, files.id ASC",
    "date_asc": "files.capture_time ASC, files.id ASC",
    "date_desc": "files.capture_time DESC, files.id ASC",
    "recent": "files.last_scan_time DESC, files.id DESC",
    "path": "files.path ASC",
}


def review_overview(connection) -> dict[str, object]:
    counts = connection.execute(
        """
        SELECT
            COUNT(files.id) AS total_files,
            COUNT(scores.file_id) AS scored_files,
            SUM(COALESCE(review_state.delete_marked, 0)) AS delete_marked,
            SUM(COALESCE(review_state.export_marked, 0)) AS export_marked
        FROM files
        LEFT JOIN scores ON scores.file_id = files.id
        LEFT JOIN review_state ON review_state.file_id = files.id
        """
    ).fetchone()

    return {
        "summary": {
            "total_files": counts["total_files"] or 0,
            "scored_files": counts["scored_files"] or 0,
            "delete_marked": counts["delete_marked"] or 0,
            "export_marked": counts["export_marked"] or 0,
        },
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



def _build_file_filters(
    *,
    root: str | None = None,
    query: str | None = None,
) -> tuple[list[str], list[object]]:
    path_query = query.casefold() if query else None
    conditions: list[str] = []
    params: list[object] = []

    if root:
        root_path = Path(root).expanduser().resolve()
        root_clause, root_params = root_path_filter("files.path_key", root_path)
        conditions.append(root_clause)
        params.extend(root_params)

    if path_query:
        conditions.append("unicode_casefold(files.path) LIKE ? ESCAPE '\\'")
        params.append(f"%{escape_like(path_query)}%")

    return conditions, params


def _build_score_filters(
    *,
    require_scored: bool,
    min_score: float | None = None,
    max_score: float | None = None,
) -> tuple[list[str], list[object]]:
    conditions: list[str] = []
    params: list[object] = []

    if require_scored:
        conditions.append("scores.overall_score IS NOT NULL")

    if min_score is not None:
        conditions.append("scores.overall_score >= ?")
        params.append(min_score)

    if max_score is not None:
        conditions.append("scores.overall_score <= ?")
        params.append(max_score)

    return conditions, params


def _build_review_state_filters(*, marked: str = "all") -> tuple[list[str], list[object]]:
    conditions: list[str] = []
    params: list[object] = []

    if marked not in {"all", "delete", "export", "none"}:
        raise ValueError("marked must be one of: all, delete, export, none")

    if marked == "delete":
        conditions.append("COALESCE(review_state.delete_marked, 0) = 1")
    elif marked == "export":
        conditions.append("COALESCE(review_state.export_marked, 0) = 1")
    elif marked == "none":
        conditions.append("COALESCE(review_state.delete_marked, 0) = 0")
        conditions.append("COALESCE(review_state.export_marked, 0) = 0")

    return conditions, params


def _build_issue_filters(*, issues: str = "all") -> tuple[list[str], list[object]]:
    conditions: list[str] = []
    params: list[object] = []

    if issues not in {"all", "issues"}:
        raise ValueError("issues must be one of: all, issues")

    if issues == "issues":
        conditions.append("COALESCE(TRIM(files.last_error), '') <> ''")

    return conditions, params


def _compile_where_clause(*filter_groups: tuple[list[str], list[object]]) -> tuple[str, list[object]]:
    conditions: list[str] = []
    params: list[object] = []
    for group_conditions, group_params in filter_groups:
        conditions.extend(group_conditions)
        params.extend(group_params)

    where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
    return where_clause, params


def _build_after_id_filter(*, after_id: int | None = None) -> tuple[list[str], list[object]]:
    if after_id is None or after_id <= 0:
        return [], []
    return ["files.id > ?"], [after_id]


def _build_review_browser_where(
    *,
    root: str | None = None,
    marked: str = "all",
    issues: str = "all",
    query: str | None = None,
    min_score: float | None = None,
    max_score: float | None = None,
) -> tuple[str, list[object]]:
    """Build the score-backed WHERE clause for the review browser queue."""
    return _compile_where_clause(
        _build_file_filters(root=root, query=query),
        _build_score_filters(require_scored=True, min_score=min_score, max_score=max_score),
        _build_review_state_filters(marked=marked),
        _build_issue_filters(issues=issues),
    )


def count_review_files(
    connection,
    *,
    root: str | None = None,
    marked: str = "all",
    issues: str = "all",
    query: str | None = None,
    min_score: float | None = None,
    max_score: float | None = None,
) -> int:
    """Return the total count of files matching the score-backed review browser filters."""
    where_clause, params = _build_review_browser_where(
        root=root, marked=marked, issues=issues,
        query=query, min_score=min_score, max_score=max_score,
    )
    sql = f"""
        SELECT COUNT(*) AS total
        FROM files
        LEFT JOIN scores ON scores.file_id = files.id
        LEFT JOIN review_state ON review_state.file_id = files.id
        {where_clause}
    """
    row = connection.execute(sql, tuple(params)).fetchone()
    return row["total"] or 0


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
) -> str:
    if scope == "review-browser":
        where_clause, params = _build_review_browser_where(
            root=root,
            marked=marked,
            issues=issues,
            query=query,
            min_score=min_score,
            max_score=max_score,
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
    return "|".join(
        str(value or 0)
        for value in (
            row["total"],
            row["min_id"],
            row["max_id"],
            row["sum_id"],
            row["sum_sq_id"],
        )
    )


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
    limit: int = 60,
    offset: int = 0,
) -> list[dict[str, object]]:
    order_by = SORT_ORDERS.get(sort, SORT_ORDERS["score_asc"])
    where_clause, params = _build_review_browser_where(
        root=root, marked=marked, issues=issues,
        query=query, min_score=min_score, max_score=max_score,
    )
    sql_parts = [
        """
        SELECT files.id, files.path, files.format, files.preview_status, files.preview_path,
             files.width, files.height, files.capture_time, files.last_error,
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
    return [dict(row) for row in rows]


def list_review_browser_file_ids(
    connection,
    *,
    root: str | None = None,
    marked: str = "all",
    issues: str = "all",
    query: str | None = None,
    min_score: float | None = None,
    max_score: float | None = None,
    limit: int | None = None,
    after_id: int | None = None,
) -> list[int]:
    """Return score-backed review browser ids in ascending keyset order."""
    where_clause, params = _compile_where_clause(
        _build_file_filters(root=root, query=query),
        _build_score_filters(require_scored=True, min_score=min_score, max_score=max_score),
        _build_review_state_filters(marked=marked),
        _build_issue_filters(issues=issues),
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
             files.width, files.height, files.capture_time, files.last_error,
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


def remove_files_from_cache(
    connection,
    *,
    file_ids: Iterable[int],
    preview_cache_root: Path | None = None,
) -> int:
    normalized_ids = normalize_file_ids(file_ids)
    allow_preview_path_fallback = _allow_legacy_preview_path_fallback(connection, preview_cache_root)
    rows = connection.execute(
        f"SELECT path, preview_path FROM files WHERE id IN ({','.join('?' for _ in normalized_ids)})",
        tuple(normalized_ids),
    ).fetchall()

    for row in rows:
        delete_managed_preview_file(
            row["preview_path"],
            source_path=row["path"],
            preview_cache_root=preview_cache_root,
            allow_path_parent_fallback=allow_preview_path_fallback,
            suppress_errors=True,
        )

    connection.executemany(
        "DELETE FROM files WHERE id = ?",
        [(file_id,) for file_id in normalized_ids],
    )
    return len(normalized_ids)


def prune_missing_cache_entries(connection, *, preview_cache_root: Path | None = None) -> int:
    """Remove cached file entries whose source files no longer exist on disk.

    THREAD-SAFETY NOTE: The ThreadPoolExecutor is used ONLY for filesystem
    existence checks (Path.exists). All database operations must happen on the
    calling thread. Do NOT add connection.execute() calls inside the executor
    - SQLite connections are not safe to share across threads.
    """
    allow_preview_path_fallback = _allow_legacy_preview_path_fallback(connection, preview_cache_root)
    removed_count = 0
    last_seen_id = 0

    def check_exists(row):
        return row if not Path(row["path"]).exists() else None

    with ThreadPoolExecutor(max_workers=16) as executor:
        while True:
            rows = connection.execute(
                """
                SELECT id, path, preview_path
                FROM files
                WHERE id > ?
                ORDER BY id ASC
                LIMIT ?
                """,
                (last_seen_id, _PRUNE_MISSING_CACHE_BATCH_SIZE),
            ).fetchall()
            if not rows:
                break

            last_seen_id = rows[-1]["id"]
            missing_rows = list(filter(None, executor.map(check_exists, rows)))
            if not missing_rows:
                continue

            for row in missing_rows:
                delete_managed_preview_file(
                    row["preview_path"],
                    source_path=row["path"],
                    preview_cache_root=preview_cache_root,
                    allow_path_parent_fallback=allow_preview_path_fallback,
                    suppress_errors=True,
                )

            connection.executemany(
                "DELETE FROM files WHERE id = ?",
                [(row["id"],) for row in missing_rows],
            )
            removed_count += len(missing_rows)

    return removed_count


def clear_cache_scope(
    connection,
    *,
    scope: str,
    preview_cache_root: Path | None = None,
) -> dict[str, int]:
    if scope == "scores":
        removed = connection.execute("SELECT COUNT(*) AS count FROM scores").fetchone()["count"]
        connection.execute("DELETE FROM scores")
        return {"files": 0, "scores": removed, "review": 0, "scan_runs": 0}

    if scope == "review":
        removed = connection.execute("SELECT COUNT(*) AS count FROM review_state").fetchone()["count"]
        connection.execute("DELETE FROM review_state")
        return {"files": 0, "scores": 0, "review": removed, "scan_runs": 0}

    if scope == "all":
        allow_preview_path_fallback = _allow_legacy_preview_path_fallback(connection, preview_cache_root)
        review_count = connection.execute("SELECT COUNT(*) AS count FROM review_state").fetchone()["count"]
        score_count = connection.execute("SELECT COUNT(*) AS count FROM scores").fetchone()["count"]
        file_count = connection.execute("SELECT COUNT(*) AS count FROM files").fetchone()["count"]
        scan_run_count = connection.execute("SELECT COUNT(*) AS count FROM scan_runs").fetchone()["count"]
        preview_rows = connection.execute(
            "SELECT path, preview_path FROM files WHERE preview_path IS NOT NULL"
        ).fetchall()

        for row in preview_rows:
            delete_managed_preview_file(
                row["preview_path"],
                source_path=row["path"],
                preview_cache_root=preview_cache_root,
                allow_path_parent_fallback=allow_preview_path_fallback,
                suppress_errors=True,
            )

        cleanup_roots = []
        if preview_cache_root is not None:
            cleanup_roots.append(preview_cache_root.expanduser().resolve())
        cleanup_roots.extend(infer_preview_cache_roots(connection))

        for cleanup_root in dict.fromkeys(cleanup_roots):
            if preview_cache_root_is_claimed(cleanup_root):
                clear_preview_cache_dir(cleanup_root, suppress_errors=True)

        connection.execute("DELETE FROM review_state")
        connection.execute("DELETE FROM scores")
        connection.execute("DELETE FROM files")
        connection.execute("DELETE FROM scan_runs")
        return {"files": file_count, "scores": score_count, "review": review_count, "scan_runs": scan_run_count}

    raise ValueError("scope must be one of: scores, review, all")


def delete_files(
    connection,
    *,
    file_ids: Iterable[int],
    delete_from_disk: bool,
    preview_cache_root: Path | None = None,
) -> dict[str, object]:
    normalized_ids = normalize_file_ids(file_ids)
    allow_preview_path_fallback = _allow_legacy_preview_path_fallback(connection, preview_cache_root)
    trusted_roots = _trusted_delete_roots(connection) if delete_from_disk else ()
    rows = connection.execute(
        f"SELECT id, path, path_key, preview_path FROM files WHERE id IN ({','.join('?' for _ in normalized_ids)})",
        tuple(normalized_ids),
    ).fetchall()

    if len(rows) != len(normalized_ids):
        raise ValueError("One or more file_ids do not exist in the cache")

    deleted_ids: list[int] = []
    failed: list[dict[str, object]] = []

    for row in rows:
        try:
            if delete_from_disk:
                resolved_source_path = _resolve_source_path_within_roots(
                    row["path"],
                    row["path_key"],
                    trusted_roots,
                )
                delete_managed_preview_file(
                    row["preview_path"],
                    source_path=resolved_source_path,
                    preview_cache_root=preview_cache_root,
                    allow_path_parent_fallback=allow_preview_path_fallback,
                )

                resolved_source_path.unlink(missing_ok=True)

            connection.execute("DELETE FROM files WHERE id = ?", (row["id"],))
            deleted_ids.append(row["id"])
        except (OSError, ValueError) as exc:
            failed.append({"id": row["id"], "path": row["path"], "error": str(exc)})

    return {
        "deleted_ids": deleted_ids,
        "deleted_count": len(deleted_ids),
        "failed": failed,
        "failed_count": len(failed),
        "delete_from_disk": delete_from_disk,
    }


def normalize_file_ids(file_ids: Iterable[int]) -> list[int]:
    normalized = sorted({int(file_id) for file_id in file_ids})
    if not normalized:
        raise ValueError("At least one file_id is required")
    if any(file_id <= 0 for file_id in normalized):
        raise ValueError("file_ids must all be positive integers")
    return normalized


def _allow_legacy_preview_path_fallback(connection, preview_cache_root: Path | None) -> bool:
    if preview_cache_root is None:
        return False
    return len(infer_preview_cache_roots(connection)) > 1


def _trusted_delete_roots(connection) -> tuple[Path, ...]:
    rows = connection.execute(
        """
        SELECT DISTINCT root_path
        FROM scan_runs
        WHERE COALESCE(TRIM(root_path), '') != ''
        ORDER BY root_path ASC
        """
    ).fetchall()
    roots: list[Path] = []
    for row in rows:
        try:
            roots.append(Path(row["root_path"]).expanduser().resolve())
        except OSError:
            continue
    return tuple(dict.fromkeys(roots))


def _resolve_source_path_within_roots(
    path_value: str | Path,
    expected_path_key: str,
    trusted_roots: Sequence[Path],
) -> Path:
    resolved_path = Path(path_value).expanduser().resolve()
    if not trusted_roots:
        raise OSError(
            f"Refusing to delete file outside tracked scan roots: {resolved_path}"
        )

    resolved_path_key = normalize_resolved_path(resolved_path)
    if resolved_path_key != expected_path_key:
        raise OSError(
            f"Refusing to delete file with mismatched path key: {resolved_path}"
        )

    for root in trusted_roots:
        if _is_within_dir(resolved_path, root):
            return resolved_path

    raise OSError(f"Refusing to delete file outside tracked scan roots: {resolved_path}")


def _is_within_dir(path: Path, candidate_dir: Path) -> bool:
    try:
        path.relative_to(candidate_dir)
        return True
    except ValueError:
        return False


_RAW_EXTENSIONS = RAW_CAMERA_EXTENSIONS
_BROWSER_SAFE_SOURCE_EXTENSIONS = BROWSER_SAFE_EXTENSIONS
_STRONGLY_PREFER_GENERATED_PREVIEW_EXTENSIONS = PREVIEW_PRIORITY_EXTENSIONS


def _resolve_ready_preview_path(raw_preview_path: str | None, preview_status: str | None) -> Path | None:
    if not raw_preview_path or preview_status != "ready":
        return None
    preview_path = Path(raw_preview_path)
    return preview_path if preview_path.exists() else None


def _should_strongly_prefer_generated_preview(source_ext: str) -> bool:
    return source_ext.casefold() in _STRONGLY_PREFER_GENERATED_PREVIEW_EXTENSIONS


def media_path_for_file(connection, *, file_id: int, variant: str) -> Path | None:
    row = connection.execute(
        "SELECT path, preview_path, preview_status FROM files WHERE id = ?",
        (file_id,),
    ).fetchone()
    if row is None:
        return None

    source_path = Path(row["path"])
    preview_path = _resolve_ready_preview_path(row["preview_path"], row["preview_status"])

    # /api/media/source must return the original file path when available.
    if variant == "source":
        return source_path if source_path.exists() else None

    # /api/media/preview prefers generated previews whenever available.
    if preview_path is not None:
        return preview_path

    if not source_path.exists():
        return None

    # Generated preview is preferred for browser-fragile formats, but the
    # source is a deterministic last-resort fallback for any format.
    return source_path