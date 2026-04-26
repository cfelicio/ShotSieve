import concurrent.futures
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Iterable, Sequence

from shotsieve.config import DEFAULT_RAW_PREVIEW_MODE
from shotsieve.db import PREVIEW_CACHE_ROOT_METADATA_KEY, get_metadata_value, infer_preview_cache_roots, normalize_resolved_path, root_path_filter, set_preview_cache_root
from shotsieve.models import ScanSummary
from shotsieve.preview import generate_preview



def canonical_path_key(path: Path) -> str:
    return normalize_resolved_path(path)


def discover_files(
    root: Path,
    *,
    recursive: bool = True,
    extensions: Sequence[str],
    excluded_dirs: Sequence[Path] = (),
) -> Iterable[Path]:
    walker = root.rglob("*") if recursive else root.glob("*")
    allowed_extensions = {extension.casefold() for extension in extensions}
    excluded = [path.resolve() for path in excluded_dirs]
    claimed_preview_roots: set[Path] = set()

    for path in walker:
        if not path.is_file():
            continue

        resolved_path = path.resolve()
        if any(is_within_dir(resolved_path, excluded_dir) for excluded_dir in excluded):
            continue
        if _is_within_claimed_preview_root(resolved_path, claimed_preview_roots):
            continue

        if resolved_path.suffix.casefold() in allowed_extensions:
            yield path


def _run_cancel_check(cancel_check: Callable[[], None] | None) -> None:
    if cancel_check is not None:
        cancel_check()


class ScanBatchInterrupted(InterruptedError):
    def __init__(self, message: str, *, attempted_count: int) -> None:
        super().__init__(message)
        self.attempted_count = attempted_count


class ScanInterrupted(InterruptedError):
    def __init__(self, message: str, *, processed_count: int) -> None:
        super().__init__(message)
        self.processed_count = processed_count


def scan_root(
    connection,
    *,
    root: Path,
    recursive: bool = True,
    limit: int | None = None,
    offset: int = 0,
    extensions: Sequence[str],
    preview_dir: Path,
    rescan_all: bool = False,
    generate_previews: bool = True,
    raw_preview_mode: str = DEFAULT_RAW_PREVIEW_MODE,
    resource_profile: str | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
    files_total_hint: int | None = None,
    cancel_check: Callable[[], None] | None = None,
) -> ScanSummary:
    summary = ScanSummary()
    started_time = utc_now()
    requested_offset = max(0, int(offset))
    remaining_offset = requested_offset
    last_error: str | None = None
    seen_path_keys: set[str] = set()

    stored_preview_root = get_metadata_value(connection, PREVIEW_CACHE_ROOT_METADATA_KEY)
    existing_preview_roots = []
    if stored_preview_root:
        existing_preview_roots.append(Path(stored_preview_root).expanduser().resolve())
    existing_preview_roots.extend(infer_preview_cache_roots(connection))
    existing_preview_roots = list(dict.fromkeys(existing_preview_roots))

    preview_root_claimed = False
    processed_count = 0
    total_hint = max(0, int(files_total_hint or 0))
    pending_paths: list[Path] = []
    if not generate_previews and stored_preview_root is None and not existing_preview_roots:
        try:
            set_preview_cache_root(connection, preview_dir)
        except ValueError:
            pass

    excluded_preview_dirs = [preview_dir]
    excluded_preview_dirs.extend(existing_preview_roots)

    cursor = connection.execute(
        """
        INSERT INTO scan_runs(started_time, root_path, status)
        VALUES(?, ?, 'running')
        """,
        (started_time, str(root.resolve())),
    )
    scan_run_id = cursor.lastrowid
    shared_executor: concurrent.futures.ProcessPoolExecutor | None = None

    try:
        if progress_callback is not None:
            progress_callback(0, total_hint, "scanning")

        file_stream = discover_files(
            root,
            recursive=recursive,
            extensions=extensions,
            excluded_dirs=tuple(excluded_preview_dirs),
        )
        
        # Scale workers with CPU cores and available RAM.
        # recommended_cpu_workers() caps workers based on RAM to prevent OOM
        # on machines with many cores but limited memory.
        from shotsieve.learned_iqa import recommended_cpu_workers
        max_workers = recommended_cpu_workers(resource_profile)
        if generate_previews and max_workers > 1:
            shared_executor = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)

        # Collect files first so we know the count before deciding on strategy.
        for path in file_stream:
            _run_cancel_check(cancel_check)
            if remaining_offset > 0:
                remaining_offset -= 1
                summary.offset_consumed += 1
                continue
            if limit is not None and processed_count >= limit:
                break
            processed_count += 1
            summary.files_seen += 1
            if progress_callback is not None:
                try:
                    progress_callback(processed_count, total_hint, "scanning")
                except InterruptedError:
                    processed_count = max(0, processed_count - 1)
                    summary.files_seen = max(0, summary.files_seen - 1)
                    raise
            seen_path_keys.add(canonical_path_key(path))
            pending_paths.append(path)

            # Flush in batches to keep memory low.  The batch is processed
            # either inline or via the pool depending on its size.
            if len(pending_paths) >= 100:
                _run_cancel_check(cancel_check)
                existing_rows = _load_existing_rows(connection, pending_paths)
                if generate_previews and not preview_root_claimed and _batch_requires_preview_generation(
                    pending_paths,
                    existing_rows=existing_rows,
                    rescan_all=rescan_all,
                ):
                    try:
                        set_preview_cache_root(connection, preview_dir)
                        preview_root_claimed = True
                    except ValueError:
                        preview_root_claimed = False
                queued_count = len(pending_paths)
                try:
                    _process_scan_batch(
                        pending_paths, connection, summary, max_workers,
                        preview_dir=preview_dir, rescan_all=rescan_all,
                        generate_previews=generate_previews,
                        raw_preview_mode=raw_preview_mode,
                        executor=shared_executor,
                        existing_rows=existing_rows,
                        cancel_check=cancel_check,
                    )
                except ScanBatchInterrupted as exc:
                    skipped_count = max(0, queued_count - exc.attempted_count)
                    processed_count = max(0, processed_count - skipped_count)
                    summary.files_seen = max(0, summary.files_seen - skipped_count)
                    pending_paths = []
                    raise InterruptedError(str(exc)) from exc
                last_error = last_error or summary.last_batch_error
                pending_paths = []

        # Handle remaining files.
        if pending_paths:
            _run_cancel_check(cancel_check)
            existing_rows = _load_existing_rows(connection, pending_paths)
            if generate_previews and not preview_root_claimed and _batch_requires_preview_generation(
                pending_paths,
                existing_rows=existing_rows,
                rescan_all=rescan_all,
            ):
                try:
                    set_preview_cache_root(connection, preview_dir)
                    preview_root_claimed = True
                except ValueError:
                    preview_root_claimed = False
            queued_count = len(pending_paths)
            try:
                _process_scan_batch(
                    pending_paths, connection, summary, max_workers,
                    preview_dir=preview_dir, rescan_all=rescan_all,
                    generate_previews=generate_previews,
                    raw_preview_mode=raw_preview_mode,
                    executor=shared_executor,
                    existing_rows=existing_rows,
                    cancel_check=cancel_check,
                )
            except ScanBatchInterrupted as exc:
                skipped_count = max(0, queued_count - exc.attempted_count)
                processed_count = max(0, processed_count - skipped_count)
                summary.files_seen = max(0, summary.files_seen - skipped_count)
                pending_paths = []
                raise InterruptedError(str(exc)) from exc
            last_error = last_error or summary.last_batch_error

        if progress_callback is not None:
            final_total = total_hint or processed_count
            progress_callback(processed_count, final_total, "scanning")

        if limit is None and requested_offset == 0:
            _run_cancel_check(cancel_check)
            summary.files_removed += purge_missing_files(connection, root=root, seen_path_keys=seen_path_keys)

        connection.execute(
            """
            UPDATE scan_runs
            SET completed_time = ?,
                files_seen = ?,
                files_added = ?,
                files_updated = ?,
                files_unchanged = ?,
                files_removed = ?,
                status = ?,
                error_text = ?
            WHERE id = ?
            """,
            (
                utc_now(),
                summary.files_seen,
                summary.files_added,
                summary.files_updated,
                summary.files_unchanged,
                summary.files_removed,
                "completed_with_errors" if summary.files_failed else "completed",
                last_error,
                scan_run_id,
            ),
        )
    except Exception as exc:
        if pending_paths:
            unflushed_count = len(pending_paths)
            processed_count = max(0, processed_count - unflushed_count)
            summary.files_seen = max(0, summary.files_seen - unflushed_count)
            pending_paths = []
        if progress_callback is not None:
            final_total = max(total_hint, processed_count)
            try:
                progress_callback(processed_count, final_total, "failed")
            except InterruptedError:
                pass
        connection.execute(
            """
            UPDATE scan_runs
            SET completed_time = ?,
                files_seen = ?,
                files_added = ?,
                files_updated = ?,
                files_unchanged = ?,
                files_removed = ?,
                status = 'failed',
                error_text = ?
            WHERE id = ?
            """,
            (
                utc_now(),
                summary.files_seen,
                summary.files_added,
                summary.files_updated,
                summary.files_unchanged,
                summary.files_removed,
                str(exc),
                scan_run_id,
            ),
        )
        if isinstance(exc, InterruptedError):
            raise ScanInterrupted(str(exc), processed_count=processed_count) from exc
        raise
    finally:
        if shared_executor is not None:
            shared_executor.shutdown(wait=True)

    return summary


_POOL_THRESHOLD = 4  # Use inline processing for batches smaller than this.


def _process_scan_batch(
    paths: list[Path],
    connection,
    summary,
    max_workers: int,
    *,
    preview_dir: Path,
    rescan_all: bool,
    generate_previews: bool,
    raw_preview_mode: str = DEFAULT_RAW_PREVIEW_MODE,
    executor: concurrent.futures.ProcessPoolExecutor | None = None,
    existing_rows: dict[str, dict] | None = None,
    cancel_check: Callable[[], None] | None = None,
) -> None:
    """Process a batch of scan paths, choosing inline or parallel execution.

    For small batches (< _POOL_THRESHOLD files), processes inline to avoid
    the ~200-500ms Windows cost of creating a ProcessPoolExecutor (each
    worker re-imports PIL, rawpy, pillow_heif into its own address space).

    When *executor* is provided, it is reused across batches to amortize
    pool startup cost over the entire scan.  When omitted, a temporary pool
    is created for this batch only (backward-compatible fallback).
    """
    # Prefetch existing DB metadata for the batch so we can pass it to
    # gather_file_metadata() for the rescan short-circuit and to
    # commit_batch() for accurate updated-vs-unchanged accounting.
    if existing_rows is None:
        existing_rows = _load_existing_rows(connection, paths)

    batch_results: list[dict] = []
    attempted_count = 0

    if not generate_previews or len(paths) < _POOL_THRESHOLD:
        # Inline path — skip pool overhead for trivially small batches.
        try:
            for path in paths:
                _run_cancel_check(cancel_check)
                try:
                    attempted_count += 1
                    pk = canonical_path_key(path)
                    batch_results.append(
                        gather_file_metadata(
                            path,
                            preview_dir=preview_dir,
                            rescan_all=rescan_all,
                            generate_previews=generate_previews,
                            raw_preview_mode=raw_preview_mode,
                            existing_metadata=existing_rows.get(pk),
                        )
                    )
                except Exception as exc:
                    summary.files_failed += 1
                    summary.last_batch_error = str(exc)
        except InterruptedError:
            if batch_results:
                commit_batch(connection, batch_results, summary, existing_rows=existing_rows)
            raise ScanBatchInterrupted("Scan job was cancelled by user.", attempted_count=attempted_count)
    else:
        # Parallel path — use provided executor or create a temporary one.
        pool = executor
        owns_pool = pool is None
        if owns_pool:
            pool = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)
        assert pool is not None
        cancel_error: str | None = None
        try:
            futures: set[concurrent.futures.Future] = set()
            path_iter = iter(paths)

            def submit_until_full() -> None:
                nonlocal cancel_error
                while cancel_error is None and len(futures) < max_workers:
                    _run_cancel_check(cancel_check)
                    try:
                        path = next(path_iter)
                    except StopIteration:
                        return
                    futures.add(
                        pool.submit(
                            gather_file_metadata,
                            path,
                            preview_dir=preview_dir,
                            rescan_all=rescan_all,
                            generate_previews=generate_previews,
                            raw_preview_mode=raw_preview_mode,
                            existing_metadata=existing_rows.get(canonical_path_key(path)),
                        )
                    )

            try:
                submit_until_full()
            except InterruptedError as exc:
                cancel_error = str(exc)

            while futures:
                done, still_pending = concurrent.futures.wait(
                    futures,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                futures = set(still_pending)

                if cancel_error is None:
                    try:
                        _run_cancel_check(cancel_check)
                    except InterruptedError as exc:
                        cancel_error = str(exc)
                        for pending_future in futures:
                            pending_future.cancel()

                for future in done:
                    if future.cancelled():
                        continue

                    attempted_count += 1
                    try:
                        batch_results.append(future.result())
                    except concurrent.futures.CancelledError:
                        continue
                    except Exception as exc:
                        summary.files_failed += 1
                        summary.last_batch_error = str(exc)

                if cancel_error is None:
                    try:
                        submit_until_full()
                    except InterruptedError as exc:
                        cancel_error = str(exc)
                        for pending_future in futures:
                            pending_future.cancel()

            if cancel_error is not None:
                if batch_results:
                    commit_batch(connection, batch_results, summary, existing_rows=existing_rows)
                raise ScanBatchInterrupted(cancel_error, attempted_count=attempted_count)
        finally:
            if owns_pool:
                pool.shutdown(wait=True)

    if batch_results:
        commit_batch(connection, batch_results, summary, existing_rows=existing_rows)


def _load_existing_rows(connection, paths: Sequence[Path]) -> dict[str, dict]:
    path_keys = [canonical_path_key(p) for p in paths]
    if not path_keys:
        return {}

    placeholders = ",".join("?" for _ in path_keys)
    return {
        row["path_key"]: {
            "modified_time": row["modified_time"],
            "size_bytes": row["size_bytes"],
            "width": row["width"],
            "height": row["height"],
            "capture_time": row["capture_time"],
            "preview_status": row["preview_status"],
            "preview_path": row["preview_path"],
            "last_error": row["last_error"],
        }
        for row in connection.execute(
            f"SELECT path_key, modified_time, size_bytes, width, height, capture_time, preview_status, preview_path, last_error FROM files WHERE path_key IN ({placeholders})",
            path_keys,
        ).fetchall()
    }


def _batch_requires_preview_generation(
    paths: Sequence[Path],
    *,
    existing_rows: dict[str, dict],
    rescan_all: bool,
) -> bool:
    return any(
        _file_requires_preview_generation(
            path,
            existing_rows.get(canonical_path_key(path)),
            rescan_all=rescan_all,
        )
        for path in paths
    )


def _file_requires_preview_generation(
    path: Path,
    existing_metadata: dict | None,
    *,
    rescan_all: bool,
) -> bool:
    if rescan_all or existing_metadata is None:
        return True

    stat = path.stat()
    if existing_metadata.get("modified_time") != stat.st_mtime:
        return True
    if existing_metadata.get("size_bytes") != stat.st_size:
        return True

    existing_preview_path = existing_metadata.get("preview_path")
    existing_preview_exists = bool(existing_preview_path) and Path(str(existing_preview_path)).exists()
    return existing_metadata.get("preview_status") != "ready" or not existing_preview_exists


def gather_file_metadata(
    path: Path,
    *,
    preview_dir: Path,
    rescan_all: bool,
    generate_previews: bool = True,
    raw_preview_mode: str = DEFAULT_RAW_PREVIEW_MODE,
    existing_metadata: dict | None = None,
) -> dict:
    stat = path.stat()
    format_name = path.suffix.casefold().lstrip('.')
    path_key = canonical_path_key(path)
    metadata_changed = (
        existing_metadata is None
        or existing_metadata.get("modified_time") != stat.st_mtime
        or existing_metadata.get("size_bytes") != stat.st_size
    )
    existing_preview_path = existing_metadata.get("preview_path") if existing_metadata else None
    existing_preview_exists = bool(existing_preview_path) and Path(str(existing_preview_path)).exists()
    base_scan_status = "new" if existing_metadata is None else ("updated" if metadata_changed else "unchanged")

    # Short-circuit: if the file hasn't changed and its preview is ready,
    # skip the expensive preview generation entirely.
    if (
        not rescan_all
        and existing_metadata is not None
        and existing_metadata.get("modified_time") == stat.st_mtime
        and existing_metadata.get("size_bytes") == stat.st_size
        and existing_metadata.get("preview_status") == "ready"
        and existing_preview_exists
    ):
        return {
            "path": str(path),
            "path_key": path_key,
            "size_bytes": stat.st_size,
            "modified_time": stat.st_mtime,
            "format": format_name,
            "last_scan_time": utc_now(),
            "width": None,
            "height": None,
            "capture_time": None,
            "preview_path": existing_metadata.get("preview_path"),
            "preview_status": "ready",
            "last_error": existing_metadata.get("last_error"),
            "scan_status": "unchanged",
            "preserve_metadata": True,
        }

    metadata = {
        "path": str(path),
        "path_key": path_key,
        "size_bytes": stat.st_size,
        "modified_time": stat.st_mtime,
        "format": format_name,
        "last_scan_time": utc_now(),
        "width": None,
        "height": None,
        "capture_time": None,
        "preview_path": None,
        "preview_status": "pending",
        "last_error": None,
        "scan_status": base_scan_status,
        "preserve_metadata": False,
    }

    if generate_previews:
        # Previews are CPU intensive, but generate_preview handles its own errors
        preview = generate_preview(path, preview_dir, raw_preview_mode=raw_preview_mode)
        metadata.update({
            "preview_path": preview.path,
            "preview_status": preview.status,
            "width": preview.width,
            "height": preview.height,
            "capture_time": preview.capture_time,
            "last_error": preview.error_text,
            "scan_status": "error" if preview.status == "failed" else base_scan_status,
        })
    
    return metadata


def commit_batch(connection, batch: list[dict], summary: ScanSummary, *, existing_rows: dict[str, dict] | None = None):
    if not batch:
        return

    # Determine which path_keys already exist and their metadata for
    # accurate added / updated / unchanged accounting.
    if existing_rows is None:
        path_keys = [item["path_key"] for item in batch]
        placeholders = ",".join("?" for _ in path_keys)
        existing_rows = {
            row["path_key"]: {
                "modified_time": row["modified_time"],
                "size_bytes": row["size_bytes"],
            }
            for row in connection.execute(
                f"SELECT path_key, modified_time, size_bytes FROM files WHERE path_key IN ({placeholders})",
                path_keys,
            ).fetchall()
        }

    connection.executemany(
        """
        INSERT INTO files(
            path, path_key, size_bytes, modified_time, format, 
            width, height, capture_time, preview_path, preview_status,
            last_scan_time, last_error, scan_status
        )
        VALUES(
            :path, :path_key, :size_bytes, :modified_time, :format, 
            :width, :height, :capture_time, :preview_path, :preview_status,
            :last_scan_time, :last_error, :scan_status
        )
        ON CONFLICT(path_key) DO UPDATE SET
            path = excluded.path,
            size_bytes = excluded.size_bytes,
            modified_time = excluded.modified_time,
            format = excluded.format,
            width = CASE
                WHEN :preserve_metadata THEN COALESCE(excluded.width, width)
                    ELSE excluded.width
                END,
                height = CASE
                WHEN :preserve_metadata THEN COALESCE(excluded.height, height)
                    ELSE excluded.height
                END,
                capture_time = CASE
                WHEN :preserve_metadata THEN COALESCE(excluded.capture_time, capture_time)
                    ELSE excluded.capture_time
                END,
                preview_path = CASE
                WHEN :preserve_metadata THEN COALESCE(excluded.preview_path, preview_path)
                    ELSE excluded.preview_path
                END,
                preview_status = CASE
                WHEN :preserve_metadata THEN COALESCE(excluded.preview_status, preview_status)
                    ELSE excluded.preview_status
                END,
            last_scan_time = excluded.last_scan_time,
                last_error = CASE
                WHEN :preserve_metadata THEN COALESCE(excluded.last_error, last_error)
                    ELSE excluded.last_error
                END,
            scan_status = excluded.scan_status
        """,
        batch,
    )

    for item in batch:
        existing = existing_rows.get(item["path_key"])
        if existing is None:
            summary.files_added += 1
        elif (
            existing.get("modified_time") != item["modified_time"]
            or existing.get("size_bytes") != item["size_bytes"]
        ):
            summary.files_updated += 1
        else:
            summary.files_unchanged += 1
        if item.get("scan_status") == "error":
            summary.files_failed += 1
            if item.get("last_error"):
                summary.last_batch_error = item["last_error"]


def purge_missing_files(connection, *, root: Path, seen_path_keys: set[str]) -> int:
    root_path = root.resolve()
    where_clause, params = root_path_filter("path_key", root_path)
    rows = connection.execute(
        f"SELECT id, path_key FROM files WHERE {where_clause}",
        tuple(params),
    ).fetchall()

    removed_ids = [
        row["id"]
        for row in rows
        if row["path_key"] not in seen_path_keys
    ]

    if not removed_ids:
        return 0

    connection.executemany(
        "DELETE FROM files WHERE id = ?",
        [(file_id,) for file_id in removed_ids],
    )
    return len(removed_ids)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def is_within_dir(path: Path, candidate_dir: Path) -> bool:
    try:
        path.relative_to(candidate_dir)
        return True
    except ValueError:
        return False


def _is_within_claimed_preview_root(path: Path, claimed_preview_roots: set[Path]) -> bool:
    for parent in path.parents:
        if parent in claimed_preview_roots:
            return True
        if (parent / ".shotsieve-preview-root").exists():
            claimed_preview_roots.add(parent)
            return True
    return False
