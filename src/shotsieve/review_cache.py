from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Sequence, Callable, Any
from concurrent.futures import ThreadPoolExecutor

from shotsieve.db import infer_preview_cache_roots, normalize_resolved_path, preview_cache_root_is_claimed
from shotsieve.preview import clear_preview_cache_dir, delete_managed_preview_file


_PRUNE_MISSING_CACHE_BATCH_SIZE = 5000


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
        f"SELECT id, path, path_key, preview_path FROM files WHERE id IN ({','.join('?' for _ in normalized_ids)})",
        tuple(normalized_ids),
    ).fetchall()

    if len(rows) != len(normalized_ids):
        raise ValueError("One or more file_ids do not exist in the cache")

    deleted_ids: list[int] = []
    failed: list[dict[str, object]] = []
    total_files = len(rows)

    if progress_callback is not None:
        progress_callback(0, total_files)

    for index, row in enumerate(rows, start=1):
        if cancel_check is not None:
            cancel_check()
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
        finally:
            if progress_callback is not None:
                progress_callback(index, total_files)

    return {
        "deleted_ids": deleted_ids,
        "deleted_count": len(deleted_ids),
        "failed": failed,
        "failed_count": len(failed),
        "delete_from_disk": delete_from_disk,
    }
