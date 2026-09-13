from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from shotsieve.db import (
    infer_preview_cache_roots,
    normalize_resolved_path,
    preview_cache_root_is_claimed,
    root_path_filter,
)
from shotsieve.models import (
    FilesystemObservation,
    FileOperationResult,
    FileOperationSummary,
    attach_file_operation_summary,
    observe_filesystem_path,
)
from shotsieve.preview import clear_preview_cache_dir, delete_managed_preview_file

_PRUNE_MISSING_CACHE_BATCH_SIZE = 5000
_MISSING_CACHE_TOKEN_VERSION = "missing-cache-v1"


def _rattr(name: str, fallback: Any) -> Any:
    mod = sys.modules.get("shotsieve.review")
    if mod is not None and hasattr(mod, name):
        return getattr(mod, name)
    return fallback


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


def _is_within_dir(path: Path, candidate_dir: Path) -> bool:
    try:
        path.relative_to(candidate_dir)
        return True
    except ValueError:
        return False


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


def _resolve_ready_preview_path(raw_preview_path: str | None, preview_status: str | None) -> Path | None:
    if not raw_preview_path or preview_status != "ready":
        return None
    preview_path = Path(raw_preview_path)
    return preview_path if preview_path.exists() else None


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

    batch_size = _rattr("_PRUNE_MISSING_CACHE_BATCH_SIZE", _PRUNE_MISSING_CACHE_BATCH_SIZE)
    executor_cls = _rattr("ThreadPoolExecutor", ThreadPoolExecutor)

    def check_exists(row):
        return row if not Path(row["path"]).exists() else None

    with executor_cls(max_workers=16) as executor:
        while True:
            rows = connection.execute(
                """
                SELECT id, path, preview_path
                FROM files
                WHERE id > ?
                ORDER BY id ASC
                LIMIT ?
                """,
                (last_seen_id, batch_size),
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


def _check_missing_cleanup_root(root: Path) -> tuple[Path | None, str | None]:
    """Confirm that a cleanup root is a readable directory tree.

    ``Path.exists()`` is deliberately not used here: it turns permission and
    other filesystem failures into false negatives.  A cleanup preview may
    report only entries proven absent after the complete tree walk succeeds.
    """
    try:
        resolved_root = root.expanduser().resolve()
        root_stat = os.stat(resolved_root)
        if not stat.S_ISDIR(root_stat.st_mode):
            return None, f"Cleanup root is not a directory: {resolved_root}"

        def raise_walk_error(error: OSError) -> None:
            raise error

        for _directory, _directories, _files in os.walk(resolved_root, onerror=raise_walk_error):
            pass
        os.stat(resolved_root)
    except OSError as exc:
        return None, f"Unable to verify cleanup root '{root}': {exc}"

    return resolved_root, None


def _missing_cleanup_rows(connection, root: Path):
    where_clause, params = root_path_filter("files.path_key", root)
    return connection.execute(
        f"""
        SELECT
            files.id,
            files.path,
            files.path_key,
            files.preview_path,
            CASE WHEN review_state.file_id IS NULL THEN 0 ELSE 1 END AS review_count,
            COALESCE(review_state.decision_state, 'pending') AS decision_state,
            COALESCE(review_state.delete_marked, 0) AS delete_marked,
            COALESCE(review_state.export_marked, 0) AS export_marked
        FROM files
        LEFT JOIN review_state ON review_state.file_id = files.id
        WHERE {where_clause}
        ORDER BY files.id ASC
        """,
        tuple(params),
    ).fetchall()


def _missing_source_state(path_value: str) -> tuple[bool, str | None]:
    try:
        os.stat(Path(path_value))
    except (FileNotFoundError, NotADirectoryError):
        return True, None
    except OSError as exc:
        return False, str(exc)
    return False, None


def _missing_candidate_token(root: Path, candidates: Sequence[dict[str, object]]) -> str:
    token_payload = {
        "version": _MISSING_CACHE_TOKEN_VERSION,
        "root": normalize_resolved_path(root),
        "candidates": candidates,
    }
    serialized = json.dumps(token_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _inspect_missing_cache_entries(connection, *, root: Path) -> tuple[dict[str, object], list[Any]]:
    resolved_root, error = _check_missing_cleanup_root(root)
    if resolved_root is None:
        return {
            "status": "unknown",
            "root": str(root),
            "candidate_count": 0,
            "affected_review_count": 0,
            "candidates": [],
            "error": error or "Unable to verify cleanup root.",
        }, []

    missing_rows = []
    for row in _missing_cleanup_rows(connection, resolved_root):
        is_missing, source_error = _missing_source_state(row["path"])
        if source_error is not None:
            return {
                "status": "unknown",
                "root": str(resolved_root),
                "candidate_count": 0,
                "affected_review_count": 0,
                "candidates": [],
                "error": f"Unable to verify cached source '{row['path']}': {source_error}",
            }, []
        if is_missing:
            missing_rows.append(row)

    candidates = [
        {
            "id": int(row["id"]),
            "path": str(row["path"]),
            "review_count": int(row["review_count"] or 0),
            "decision_state": str(row["decision_state"] or "pending"),
            "delete_marked": int(row["delete_marked"] or 0),
            "export_marked": int(row["export_marked"] or 0),
        }
        for row in missing_rows
    ]
    token = _missing_candidate_token(resolved_root, candidates)
    payload: dict[str, object] = {
        "status": "ready" if candidates else "empty",
        "root": str(resolved_root),
        "candidate_count": len(candidates),
        "affected_review_count": sum(int(candidate["review_count"]) for candidate in candidates),
        "candidates": candidates,
        "revision": token,
        "token": token,
    }
    return payload, missing_rows


def preview_missing_cache_entries(connection, *, root: Path) -> dict[str, object]:
    """Preview missing catalog entries below one explicitly selected root."""
    payload, _rows = _inspect_missing_cache_entries(connection, root=root)
    return payload


def apply_missing_cache_entries(
    connection,
    *,
    root: Path,
    token: str,
    candidate_ids: Iterable[int],
    preview_cache_root: Path | None = None,
) -> dict[str, object]:
    """Apply an exact, previously previewed missing-entry cleanup.

    The database write is serialized while the root and candidate set are
    rechecked.  Any changed source, scope, review state, or candidate set
    returns ``refresh_required`` without deleting anything.
    """
    normalized_ids = sorted({int(file_id) for file_id in candidate_ids})
    if any(file_id <= 0 for file_id in normalized_ids):
        raise ValueError("candidate_ids must all be positive integers")
    if not token:
        raise ValueError("token is required")

    savepoint_name = "missing_cache_cleanup"
    uses_savepoint = connection.in_transaction
    if uses_savepoint:
        connection.execute(f"SAVEPOINT {savepoint_name}")
    else:
        connection.execute("BEGIN IMMEDIATE")

    def rollback_cleanup() -> None:
        if uses_savepoint:
            connection.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
            connection.execute(f"RELEASE SAVEPOINT {savepoint_name}")
        else:
            connection.rollback()

    def finish_cleanup() -> None:
        if uses_savepoint:
            connection.execute(f"RELEASE SAVEPOINT {savepoint_name}")
        else:
            connection.commit()

    try:
        payload, missing_rows = _inspect_missing_cache_entries(connection, root=root)
        expected_ids = [int(candidate["id"]) for candidate in payload.get("candidates", [])]
        if (
            payload.get("status") == "unknown"
            or payload.get("token") != token
            or expected_ids != normalized_ids
        ):
            rollback_cleanup()
            return {
                "status": "refresh_required" if payload.get("status") != "unknown" else "unknown",
                "root": payload.get("root", str(root)),
                "candidate_count": len(expected_ids),
                "affected_review_count": int(payload.get("affected_review_count", 0) or 0),
                "candidates": payload.get("candidates", []),
                "error": payload.get("error") or "The catalog changed after the preview. Refresh the preview and confirm again.",
            }

        if not expected_ids:
            finish_cleanup()
            return {
                "status": "empty",
                "root": payload["root"],
                "candidate_count": 0,
                "affected_review_count": 0,
                "removed_count": 0,
                "review_removed_count": 0,
            }

        for row in missing_rows:
            is_missing, source_error = _missing_source_state(row["path"])
            if source_error is not None or not is_missing:
                rollback_cleanup()
                return {
                    "status": "unknown" if source_error is not None else "refresh_required",
                    "root": payload["root"],
                    "candidate_count": len(expected_ids),
                    "affected_review_count": int(payload["affected_review_count"]),
                    "candidates": payload["candidates"],
                    "error": (
                        f"Unable to verify cached source '{row['path']}': {source_error}"
                        if source_error is not None
                        else "A source file reappeared after the preview. Refresh the preview and confirm again."
                    ),
                }

        allow_preview_path_fallback = _allow_legacy_preview_path_fallback(connection, preview_cache_root)
        affected_review_count = int(payload["affected_review_count"])
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
            [(file_id,) for file_id in normalized_ids],
        )
        finish_cleanup()
        return {
            "status": "applied",
            "root": payload["root"],
            "candidate_count": len(normalized_ids),
            "affected_review_count": affected_review_count,
            "removed_count": len(normalized_ids),
            "review_removed_count": affected_review_count,
        }
    except Exception:
        rollback_cleanup()
        raise


def clear_cache_scope(
    connection,
    *,
    scope: str,
    preview_cache_root: Path | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
    cancel_check: Callable[[], None] | None = None,
) -> dict[str, int]:
    def emit_progress(processed: int, total: int, phase: str = "clearing_cache") -> None:
        if progress_callback is not None:
            progress_callback(processed, total, phase)

    if scope == "scores":
        emit_progress(0, 1)
        if cancel_check is not None:
            cancel_check()
        removed = connection.execute("SELECT COUNT(*) AS count FROM scores").fetchone()["count"]
        connection.execute("DELETE FROM scores")
        emit_progress(1, 1)
        return {"files": 0, "scores": removed, "review": 0, "scan_runs": 0}

    if scope == "review":
        emit_progress(0, 1)
        if cancel_check is not None:
            cancel_check()
        removed = connection.execute("SELECT COUNT(*) AS count FROM review_state").fetchone()["count"]
        connection.execute("DELETE FROM review_state")
        emit_progress(1, 1)
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
        total_steps = max(1, len(preview_rows) + 1)

        emit_progress(0, total_steps)

        for index, row in enumerate(preview_rows, start=1):
            if cancel_check is not None:
                cancel_check()
            delete_managed_preview_file(
                row["preview_path"],
                source_path=row["path"],
                preview_cache_root=preview_cache_root,
                allow_path_parent_fallback=allow_preview_path_fallback,
                suppress_errors=True,
            )
            emit_progress(index, total_steps)

        cleanup_roots = []
        if preview_cache_root is not None:
            cleanup_roots.append(preview_cache_root.expanduser().resolve())
        cleanup_roots.extend(infer_preview_cache_roots(connection))

        for cleanup_root in dict.fromkeys(cleanup_roots):
            if cancel_check is not None:
                cancel_check()
            if preview_cache_root_is_claimed(cleanup_root):
                clear_preview_cache_dir(cleanup_root, suppress_errors=True)

        if cancel_check is not None:
            cancel_check()
        connection.execute("DELETE FROM review_state")
        connection.execute("DELETE FROM scores")
        connection.execute("DELETE FROM files")
        connection.execute("DELETE FROM scan_runs")
        emit_progress(total_steps, total_steps)
        return {"files": file_count, "scores": score_count, "review": review_count, "scan_runs": scan_run_count}

    raise ValueError("scope must be one of: scores, review, all")


def delete_files(
    connection,
    *,
    file_ids: Iterable[int],
    delete_from_disk: bool,
    preview_cache_root: Path | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    cancel_check: Callable[[], None] | None = None,
) -> dict[str, object]:
    normalized_ids = normalize_file_ids(file_ids)
    allow_preview_path_fallback = _allow_legacy_preview_path_fallback(connection, preview_cache_root)
    trusted_roots = _trusted_delete_roots(connection) if delete_from_disk else ()
    rows = connection.execute(
        f"SELECT id, path, path_key, preview_path FROM files WHERE id IN ({','.join('?' for _ in normalized_ids)}) ORDER BY id",
        tuple(normalized_ids),
    ).fetchall()

    if len(rows) != len(normalized_ids):
        raise ValueError("One or more file_ids do not exist in the cache")

    summary = FileOperationSummary(action="delete", delete_from_disk=delete_from_disk)
    total_files = len(rows)

    if progress_callback is not None:
        try:
            progress_callback(0, total_files)
        except BaseException as exc:
            _stop_delete_with_unprocessed(
                summary,
                rows,
                0,
                exc,
                cancelled=isinstance(exc, InterruptedError),
            )

    for index, row in enumerate(rows, start=1):
        if cancel_check is not None:
            try:
                cancel_check()
            except BaseException as exc:
                _stop_delete_with_unprocessed(
                    summary,
                    rows,
                    index - 1,
                    exc,
                    cancelled=isinstance(exc, InterruptedError),
                )

        source = Path(row["path"])
        source_observation: FilesystemObservation | None = None
        if delete_from_disk:
            try:
                resolved_source_path = _resolve_source_path_within_roots(
                    row["path"],
                    row["path_key"],
                    trusted_roots,
                )
            except (OSError, ValueError) as exc:
                summary.add(
                    _delete_result_from_error(
                        row,
                        stage="source_check",
                        error=exc,
                        retry_safe=True,
                    )
                )
                _emit_delete_progress(progress_callback, index, total_files, summary, rows, index)
                continue

            source_observation = observe_filesystem_path(resolved_source_path)
            if source_observation.state != "present":
                source_error = source_observation.error_text or "Source file not found"
                summary.add(
                    _delete_result_from_error(
                        row,
                        stage="source_check",
                        error=source_error,
                        retry_safe=source_observation.state == "missing",
                        source_observation=source_observation,
                    )
                )
                _emit_delete_progress(progress_callback, index, total_files, summary, rows, index)
                continue

            try:
                resolved_source_path.unlink()
            except BaseException as exc:
                source_after = observe_filesystem_path(resolved_source_path)
                outcome = "failed" if source_after.state == "present" else "uncertain"
                summary.add(
                    _delete_result_from_error(
                        row,
                        stage="source_removal",
                        error=exc,
                        retry_safe=outcome == "failed",
                        outcome=outcome,
                        source_observation=source_after,
                    )
                )
                _emit_delete_progress(progress_callback, index, total_files, summary, rows, index)
                continue

        try:
            connection.execute("DELETE FROM files WHERE id = ?", (row["id"],))
            # A completed deletion is an independent unit of work.  Commit it
            # before cleanup so cancellation or a later row cannot resurrect it.
            connection.commit()
        except BaseException as exc:
            result, catalog_deleted, needs_rollback = _reconcile_delete_catalog_failure(
                connection,
                row=row,
                source=source,
                delete_from_disk=delete_from_disk,
                catalog_error=exc,
                source_observation=source_observation,
            )
            if catalog_deleted:
                summary.deleted_ids.append(int(row["id"]))
            summary.add(result)
            _append_delete_unprocessed_rows(summary, rows[index:], error=exc)
            attach_file_operation_summary(exc, summary, needs_rollback=needs_rollback)
            raise

        summary.deleted_ids.append(int(row["id"]))
        summary.add(
            FileOperationResult(
                file_id=int(row["id"]),
                source=str(source),
                destination=None,
                action="delete",
                outcome="success",
                stage="catalog_update" if not delete_from_disk else "source_removal",
                source_state=(
                    source_observation.state
                    if source_observation is not None
                    else "unknown"
                ),
            )
        )

        if delete_from_disk:
            try:
                delete_managed_preview_file(
                    row["preview_path"],
                    source_path=source,
                    preview_cache_root=preview_cache_root,
                    allow_path_parent_fallback=allow_preview_path_fallback,
                )
            except (OSError, ValueError) as exc:
                summary.add_warning(
                    file_id=int(row["id"]),
                    source=str(source),
                    stage="preview_cleanup",
                    error=exc,
                )

        _emit_delete_progress(progress_callback, index, total_files, summary, rows, index)

    return summary.to_dict()


def _delete_result_from_error(
    row,
    *,
    stage: str,
    error: BaseException,
    retry_safe: bool,
    outcome: str = "failed",
    source_observation: FilesystemObservation | None = None,
    destination_observation: FilesystemObservation | None = None,
) -> FileOperationResult:
    return FileOperationResult(
        file_id=int(row["id"]),
        source=str(row["path"]),
        destination=None,
        action="delete",
        outcome=outcome,
        stage=stage,
        error_text=str(error),
        errno=getattr(error, "errno", None),
        winerror=getattr(error, "winerror", None),
        retry_safe=retry_safe,
        source_state=source_observation.state if source_observation is not None else "unknown",
        destination_state=(
            destination_observation.state if destination_observation is not None else None
        ),
        observation_errors=_delete_observation_errors(
            row,
            source_observation=source_observation,
            destination_observation=destination_observation,
        ),
    )


def _delete_observation_errors(
    row,
    *,
    source_observation: FilesystemObservation | None,
    destination_observation: FilesystemObservation | None,
) -> list[str]:
    errors: list[str] = []
    if source_observation is not None and source_observation.error_text:
        errors.append(f"source '{row['path']}': {source_observation.error_text}")
    if destination_observation is not None and destination_observation.error_text:
        errors.append(f"destination: {destination_observation.error_text}")
    return errors


def _reconcile_delete_catalog_failure(
    connection,
    *,
    row,
    source: Path,
    delete_from_disk: bool,
    catalog_error: BaseException,
    source_observation: FilesystemObservation | None,
) -> tuple[FileOperationResult, bool, bool]:
    """Read back a failed catalog delete before reporting its outcome."""
    rollback_error: str | None = None
    try:
        connection.rollback()
    except BaseException as exc:
        rollback_error = str(exc) or exc.__class__.__name__

    catalog_read_error: str | None = None
    catalog_row = None
    try:
        catalog_row = connection.execute(
            "SELECT id FROM files WHERE id = ?",
            (row["id"],),
        ).fetchone()
    except BaseException as exc:
        catalog_read_error = str(exc) or exc.__class__.__name__

    if delete_from_disk:
        source_observation = observe_filesystem_path(source)
    error_parts = [str(catalog_error) or catalog_error.__class__.__name__]
    if rollback_error:
        error_parts.append(f"catalog rollback failed: {rollback_error}")
    if catalog_read_error:
        error_parts.append(f"catalog observation failed: {catalog_read_error}")
    error_text = "; ".join(error_parts)

    if catalog_row is None and catalog_read_error is None:
        return (
            _delete_result_from_error(
                row,
                stage="catalog_update",
                error=error_text,
                retry_safe=False,
                outcome="uncertain",
                source_observation=source_observation,
            ),
            True,
            False,
        )

    return (
        _delete_result_from_error(
            row,
            stage="catalog_update",
            error=error_text,
            retry_safe=(not delete_from_disk and rollback_error is None and catalog_read_error is None),
            outcome="uncertain" if delete_from_disk or catalog_read_error else "failed",
            source_observation=source_observation,
        ),
        False,
        rollback_error is not None or catalog_read_error is not None,
    )


def _append_delete_unprocessed_rows(summary: FileOperationSummary, rows, *, error: BaseException) -> None:
    accounted_ids = {item.file_id for item in summary.items}
    for row in rows:
        file_id = int(row["id"])
        if file_id in accounted_ids:
            continue
        summary.add(
            FileOperationResult(
                file_id=file_id,
                source=str(row["path"]),
                destination=None,
                action="delete",
                outcome="unprocessed",
                stage="not_started",
                error_text=str(error),
                retry_safe=True,
            )
        )
        accounted_ids.add(file_id)


def _emit_delete_progress(
    progress_callback: Callable[[int, int], None] | None,
    processed: int,
    total: int,
    summary: FileOperationSummary,
    rows,
    next_index: int,
) -> None:
    if progress_callback is None:
        return
    try:
        progress_callback(processed, total)
    except BaseException as exc:
        _stop_delete_with_unprocessed(
            summary,
            rows,
            next_index,
            exc,
            cancelled=isinstance(exc, InterruptedError),
        )


def _stop_delete_with_unprocessed(
    summary: FileOperationSummary,
    rows,
    start_index: int,
    error: BaseException,
    *,
    cancelled: bool,
) -> None:
    accounted_ids = {item.file_id for item in summary.items}
    for row in rows[start_index:]:
        file_id = int(row["id"])
        if file_id in accounted_ids:
            continue
        summary.add(
            FileOperationResult(
                file_id=file_id,
                source=str(row["path"]),
                destination=None,
                action="delete",
                outcome="unprocessed",
                stage="not_started",
                error_text=str(error),
                retry_safe=True,
            )
        )
        accounted_ids.add(file_id)
    attach_file_operation_summary(error, summary, cancelled=cancelled)
    raise error
