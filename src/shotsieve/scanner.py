import concurrent.futures
import fnmatch
import os
import stat
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from shotsieve.config import DEFAULT_RAW_PREVIEW_MODE
from shotsieve.db import (
    PREVIEW_CACHE_ROOT_METADATA_KEY,
    attach_scan_run_diagnostic,
    get_metadata_value,
    infer_preview_cache_roots,
    normalize_resolved_path,
    set_preview_cache_root,
)
from shotsieve.image_conversion import DEFAULT_MAX_DECODE_PIXELS, IMAGE_CONVERSION_VERSION
from shotsieve.models import ScanRunDiagnostic, ScanSummary
from shotsieve.preview import generate_preview


def canonical_path_key(path: Path) -> str:
    return normalize_resolved_path(path)


def _filesystem_name_sort_key(name: str) -> tuple[str, str]:
    """Return a stable cross-platform ordering for one filesystem name."""
    return name.casefold(), name


class IgnoreMatcher:
    def __init__(self, root: Path, rules: Sequence[str]) -> None:
        self.root = root.expanduser().resolve()
        self.rules: list[str] = []
        self.absolute_rules: list[Path] = []
        for r in rules:
            r = r.strip()
            if not r:
                continue
            try:
                p = Path(r).expanduser()
                if p.is_absolute():
                    p_res = p.resolve()
                    if p_res == self.root:
                        continue
                    self.absolute_rules.append(p_res)
                    r = p_res.relative_to(self.root).as_posix()
            except ValueError:
                if p.is_absolute():
                    continue
            self.rules.append(r)

    def should_ignore(self, path: Path) -> bool:
        resolved = path.expanduser().resolve()
        if resolved == self.root:
            return False

        if _is_excluded(resolved, set(self.absolute_rules)):
            return True

        try:
            rel_path = resolved.relative_to(self.root)
        except ValueError:
            return False

        rel_posix = rel_path.as_posix()
        parts = rel_path.parts

        for rule in self.rules:
            r_posix = rule.replace("\\", "/").strip("/")
            r_parts = r_posix.split("/")
            r_parts_no_wild = [p for p in r_parts if p != "**"]

            if len(r_parts_no_wild) == 1:
                pattern = r_parts_no_wild[0]
                if any(fnmatch.fnmatchcase(part, pattern) for part in parts):
                    return True
                continue

            glob_pattern = r_posix.replace("**", "*")
            if fnmatch.fnmatchcase(rel_posix, glob_pattern):
                return True
            if fnmatch.fnmatchcase(rel_posix, glob_pattern + "/*"):
                return True
            if rel_posix == r_posix or rel_posix.startswith(r_posix + "/"):
                return True

        return False


def _is_excluded(path: Path, excluded: set[Path]) -> bool:
    for exc in excluded:
        if path == exc:
            return True
        try:
            path.relative_to(exc)
            return True
        except ValueError:
            pass
    return False


class FileDiscoveryError(OSError):
    """Raised when a scan cannot establish complete filesystem coverage."""

    def __init__(self, path: Path, cause: OSError) -> None:
        self.path = path
        self.cause = cause
        super().__init__(f"Unable to enumerate '{path}': {cause}")


def _raise_discovery_error(path: Path, cause: OSError) -> None:
    if isinstance(cause, FileDiscoveryError):
        raise cause
    raise FileDiscoveryError(path, cause) from cause


def _preview_marker_exists(path: Path) -> bool:
    """Check a preview marker without hiding access errors as absence."""
    try:
        os.stat(path)
    except (FileNotFoundError, NotADirectoryError):
        return False
    except OSError as exc:
        _raise_discovery_error(path, exc)
    return True


def discover_files(
    root: Path,
    *,
    recursive: bool = True,
    extensions: Sequence[str],
    excluded_dirs: Sequence[Path] = (),
    ignore_rules: Sequence[str] = (),
) -> Iterable[Path]:
    allowed_extensions = {ext.casefold() if ext.startswith('.') else f".{ext.casefold()}" for ext in extensions}
    excluded = {path.expanduser().resolve() for path in excluded_dirs}
    claimed_preview_roots: set[Path] = set()
    
    root_resolved = root.expanduser().resolve()
    ignore_matcher = IgnoreMatcher(root_resolved, ignore_rules)

    try:
        root_stat = os.stat(root_resolved)
    except OSError as exc:
        _raise_discovery_error(root_resolved, exc)
    if not stat.S_ISDIR(root_stat.st_mode):
        _raise_discovery_error(root_resolved, NotADirectoryError(str(root_resolved)))

    if not recursive:
        try:
            with os.scandir(root_resolved) as entries:
                for entry in sorted(entries, key=lambda item: _filesystem_name_sort_key(item.name)):
                    if entry.is_file():
                        path = Path(entry.path)
                        resolved_path = path.resolve()
                        if (
                            path.suffix.casefold() in allowed_extensions
                            and not _is_excluded(resolved_path, excluded)
                            and not _is_within_claimed_preview_root(resolved_path, claimed_preview_roots)
                            and not ignore_matcher.should_ignore(resolved_path)
                        ):
                            yield path
        except FileDiscoveryError:
            raise
        except OSError as exc:
            _raise_discovery_error(root_resolved, exc)
        return

    def on_walk_error(error: OSError) -> None:
        error_path = Path(getattr(error, "filename", None) or root_resolved)
        _raise_discovery_error(error_path, error)

    for dirpath, dirnames, filenames in os.walk(root_resolved, topdown=True, onerror=on_walk_error):
        current_dir = Path(dirpath)
        dirnames.sort(key=_filesystem_name_sort_key)
        filenames.sort(key=_filesystem_name_sort_key)
        
        for dname in list(dirnames):
            dpath = current_dir / dname
            try:
                resolved_dpath = dpath.resolve()
                
                if _is_excluded(resolved_dpath, excluded):
                    dirnames.remove(dname)
                    continue
                    
                if _preview_marker_exists(resolved_dpath / ".shotsieve-preview-root"):
                    dirnames.remove(dname)
                    continue
                    
                if ignore_matcher.should_ignore(resolved_dpath):
                    dirnames.remove(dname)
                    continue
            except FileDiscoveryError:
                raise
            except OSError as exc:
                _raise_discovery_error(dpath, exc)

        for fname in filenames:
            fpath = current_dir / fname
            if fpath.suffix.casefold() not in allowed_extensions:
                continue
            try:
                resolved_fpath = fpath.resolve()
                if _is_excluded(resolved_fpath, excluded):
                    continue
                if _is_within_claimed_preview_root(resolved_fpath, claimed_preview_roots):
                    continue
                if ignore_matcher.should_ignore(resolved_fpath):
                    continue
            except FileDiscoveryError:
                raise
            except OSError as exc:
                _raise_discovery_error(fpath, exc)
            yield fpath



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


@dataclass(slots=True)
class _ScanIterationState:
    """Mutable state shared by discovery, batching, and failure finalization."""

    remaining_offset: int
    processed_count: int = 0
    preview_root_claimed: bool = False
    last_error: str | None = None
    pending_paths: list[Path] = field(default_factory=list)


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
    ignore_rules: Sequence[str] = (),
    max_decode_pixels: int = DEFAULT_MAX_DECODE_PIXELS,
) -> ScanSummary:
    summary = ScanSummary()
    started_time = utc_now()
    scan_state = _ScanIterationState(remaining_offset=max(0, int(offset)))

    cursor = connection.execute(
        """
        INSERT INTO scan_runs(started_time, root_path, status)
        VALUES(?, ?, 'running')
        """,
        (started_time, str(root.resolve())),
    )
    scan_run_id = cursor.lastrowid

    stored_preview_root = get_metadata_value(connection, PREVIEW_CACHE_ROOT_METADATA_KEY)
    existing_preview_roots = []
    if stored_preview_root:
        existing_preview_roots.append(Path(stored_preview_root).expanduser().resolve())
    existing_preview_roots.extend(infer_preview_cache_roots(connection))
    existing_preview_roots = list(dict.fromkeys(existing_preview_roots))

    total_hint = max(0, int(files_total_hint or 0))
    if not generate_previews and stored_preview_root is None and not existing_preview_roots:
        try:
            set_preview_cache_root(connection, preview_dir)
        except ValueError:
            pass

    excluded_preview_dirs = [preview_dir]
    excluded_preview_dirs.extend(existing_preview_roots)

    shared_executor: concurrent.futures.ProcessPoolExecutor | None = None

    try:
        if progress_callback is not None:
            progress_callback(0, total_hint, "scanning")

        file_stream = discover_files(
            root,
            recursive=recursive,
            extensions=extensions,
            excluded_dirs=tuple(excluded_preview_dirs),
            ignore_rules=ignore_rules,
        )

        
        from shotsieve.learned_iqa import recommended_cpu_workers

        # Scale workers with CPU cores and available RAM.
        # recommended_cpu_workers() caps workers based on RAM to prevent OOM
        # on machines with many cores but limited memory.
        max_workers = recommended_cpu_workers(resource_profile)
        shared_executor = _create_scan_executor(generate_previews, max_workers)
        _scan_discovered_files(
            file_stream,
            connection,
            summary,
            scan_state,
            max_workers,
            limit=limit,
            total_hint=total_hint,
            progress_callback=progress_callback,
            preview_dir=preview_dir,
            rescan_all=rescan_all,
            generate_previews=generate_previews,
            raw_preview_mode=raw_preview_mode,
            max_decode_pixels=max_decode_pixels,
            executor=shared_executor,
            cancel_check=cancel_check,
        )

        if progress_callback is not None:
            final_total = total_hint or scan_state.processed_count
            progress_callback(scan_state.processed_count, final_total, "scanning")

        _finalize_completed_scan(
            connection,
            scan_run_id,
            summary,
            last_error=scan_state.last_error,
        )
    except Exception as exc:
        if scan_state.pending_paths:
            unflushed_count = len(scan_state.pending_paths)
            scan_state.processed_count = max(0, scan_state.processed_count - unflushed_count)
            summary.files_seen = max(0, summary.files_seen - unflushed_count)
            scan_state.pending_paths.clear()
        _finalize_failed_scan(
            connection,
            scan_run_id,
            root=root,
            started_time=started_time,
            summary=summary,
            exc=exc,
            progress_callback=progress_callback,
            processed_count=scan_state.processed_count,
            total_hint=total_hint,
        )
    finally:
        if shared_executor is not None:
            shared_executor.shutdown(wait=True)

    return summary


_POOL_THRESHOLD = 4  # Use inline processing for batches smaller than this.


def _create_scan_executor(
    generate_previews: bool,
    max_workers: int,
) -> concurrent.futures.ProcessPoolExecutor | None:
    """Create the shared preview executor when the scan can use one."""
    if generate_previews and max_workers > 1:
        return concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)
    return None


def _scan_discovered_files(
    file_stream: Iterable[Path],
    connection,
    summary: ScanSummary,
    state: _ScanIterationState,
    max_workers: int,
    *,
    limit: int | None,
    total_hint: int,
    progress_callback: Callable[[int, int, str], None] | None,
    preview_dir: Path,
    rescan_all: bool,
    generate_previews: bool,
    raw_preview_mode: str,
    max_decode_pixels: int,
    executor: concurrent.futures.ProcessPoolExecutor | None,
    cancel_check: Callable[[], None] | None,
) -> None:
    """Collect discovered paths and flush both full and tail batches."""
    for path in file_stream:
        _run_cancel_check(cancel_check)
        if state.remaining_offset > 0:
            state.remaining_offset -= 1
            summary.offset_consumed += 1
            continue
        if limit is not None and state.processed_count >= limit:
            break

        state.processed_count += 1
        summary.files_seen += 1
        if progress_callback is not None:
            try:
                progress_callback(state.processed_count, total_hint, "scanning")
            except InterruptedError:
                state.processed_count = max(0, state.processed_count - 1)
                summary.files_seen = max(0, summary.files_seen - 1)
                raise
        state.pending_paths.append(path)

        if len(state.pending_paths) >= 100:
            _flush_pending_batch(
                connection,
                summary,
                state,
                max_workers,
                preview_dir=preview_dir,
                rescan_all=rescan_all,
                generate_previews=generate_previews,
                raw_preview_mode=raw_preview_mode,
                max_decode_pixels=max_decode_pixels,
                executor=executor,
                cancel_check=cancel_check,
            )

    if state.pending_paths:
        _flush_pending_batch(
            connection,
            summary,
            state,
            max_workers,
            preview_dir=preview_dir,
            rescan_all=rescan_all,
            generate_previews=generate_previews,
            raw_preview_mode=raw_preview_mode,
            max_decode_pixels=max_decode_pixels,
            executor=executor,
            cancel_check=cancel_check,
        )


def _flush_pending_batch(
    connection,
    summary: ScanSummary,
    state: _ScanIterationState,
    max_workers: int,
    *,
    preview_dir: Path,
    rescan_all: bool,
    generate_previews: bool,
    raw_preview_mode: str,
    max_decode_pixels: int,
    executor: concurrent.futures.ProcessPoolExecutor | None,
    cancel_check: Callable[[], None] | None,
) -> None:
    """Prepare, process, and clear one pending scan batch."""
    _run_cancel_check(cancel_check)
    existing_rows = _load_existing_rows(connection, state.pending_paths)
    if generate_previews and not state.preview_root_claimed and _batch_requires_preview_generation(
        state.pending_paths,
        existing_rows=existing_rows,
        rescan_all=rescan_all,
    ):
        try:
            set_preview_cache_root(connection, preview_dir)
            state.preview_root_claimed = True
        except ValueError:
            state.preview_root_claimed = False

    queued_count = len(state.pending_paths)
    try:
        _process_scan_batch(
            state.pending_paths,
            connection,
            summary,
            max_workers,
            preview_dir=preview_dir,
            rescan_all=rescan_all,
            generate_previews=generate_previews,
            raw_preview_mode=raw_preview_mode,
            max_decode_pixels=max_decode_pixels,
            executor=executor,
            existing_rows=existing_rows,
            cancel_check=cancel_check,
        )
    except ScanBatchInterrupted as exc:
        skipped_count = max(0, queued_count - exc.attempted_count)
        state.processed_count = max(0, state.processed_count - skipped_count)
        summary.files_seen = max(0, summary.files_seen - skipped_count)
        state.pending_paths.clear()
        raise InterruptedError(str(exc)) from exc

    state.last_error = state.last_error or summary.last_batch_error
    state.pending_paths.clear()


def _finalize_completed_scan(
    connection,
    scan_run_id: int,
    summary: ScanSummary,
    *,
    last_error: str | None,
) -> None:
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


def _finalize_failed_scan(
    connection,
    scan_run_id: int,
    *,
    root: Path,
    started_time: str,
    summary: ScanSummary,
    exc: Exception,
    progress_callback: Callable[[int, int, str], None] | None,
    processed_count: int,
    total_hint: int,
) -> None:
    """Persist failed-scan diagnostics before the caller's transaction rolls back."""
    if progress_callback is not None:
        final_total = max(total_hint, processed_count)
        try:
            progress_callback(processed_count, final_total, "failed")
        except InterruptedError:
            pass

    completed_time = utc_now()
    error_text = str(exc) or exc.__class__.__name__
    diagnostic = ScanRunDiagnostic(
        root_path=str(root.resolve()),
        started_time=started_time,
        completed_time=completed_time,
        status="failed",
        files_seen=summary.files_seen,
        files_added=summary.files_added,
        files_updated=summary.files_updated,
        files_unchanged=summary.files_unchanged,
        files_removed=summary.files_removed,
        error_text=error_text,
    )
    try:
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
                completed_time,
                summary.files_seen,
                summary.files_added,
                summary.files_updated,
                summary.files_unchanged,
                summary.files_removed,
                error_text,
                scan_run_id,
            ),
        )
    except Exception as diagnostic_update_error:
        attach_scan_run_diagnostic(diagnostic_update_error, diagnostic)
        raise exc from diagnostic_update_error
    finally:
        attach_scan_run_diagnostic(exc, diagnostic)
    if isinstance(exc, InterruptedError):
        interrupted = ScanInterrupted(error_text, processed_count=processed_count)
        attach_scan_run_diagnostic(interrupted, diagnostic)
        raise interrupted from exc
    raise exc


def _process_scan_batch(
    paths: list[Path],
    connection,
    summary: ScanSummary,
    max_workers: int,
    *,
    preview_dir: Path,
    rescan_all: bool,
    generate_previews: bool,
    raw_preview_mode: str = DEFAULT_RAW_PREVIEW_MODE,
    max_decode_pixels: int = DEFAULT_MAX_DECODE_PIXELS,
    executor: concurrent.futures.ProcessPoolExecutor | None = None,
    existing_rows: dict[str, dict] | None = None,
    cancel_check: Callable[[], None] | None = None,
) -> None:
    """Process a batch through an inline or bounded parallel strategy."""
    if existing_rows is None:
        existing_rows = _load_existing_rows(connection, paths)

    if not generate_previews or len(paths) < _POOL_THRESHOLD:
        outcome = _process_inline_batch(
            paths,
            preview_dir=preview_dir,
            rescan_all=rescan_all,
            generate_previews=generate_previews,
            raw_preview_mode=raw_preview_mode,
            max_decode_pixels=max_decode_pixels,
            existing_rows=existing_rows,
            cancel_check=cancel_check,
        )
    else:
        outcome = _process_parallel_batch(
            paths,
            max_workers,
            preview_dir=preview_dir,
            rescan_all=rescan_all,
            generate_previews=generate_previews,
            raw_preview_mode=raw_preview_mode,
            max_decode_pixels=max_decode_pixels,
            existing_rows=existing_rows,
            executor=executor,
            cancel_check=cancel_check,
        )

    _account_batch_outcome(connection, summary, outcome, existing_rows=existing_rows)


@dataclass(slots=True)
class _BatchProcessingOutcome:
    """Results and accounting signals produced by one execution strategy."""

    results: list[dict] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    attempted_count: int = 0
    cancel_error: str | None = None


def _process_inline_batch(
    paths: Sequence[Path],
    *,
    preview_dir: Path,
    rescan_all: bool,
    generate_previews: bool,
    raw_preview_mode: str,
    max_decode_pixels: int,
    existing_rows: dict[str, dict],
    cancel_check: Callable[[], None] | None,
) -> _BatchProcessingOutcome:
    """Gather a batch in the caller thread, preserving per-file failures."""
    outcome = _BatchProcessingOutcome()
    try:
        for path in paths:
            _run_cancel_check(cancel_check)
            try:
                outcome.attempted_count += 1
                outcome.results.append(
                    gather_file_metadata(
                        path,
                        preview_dir=preview_dir,
                        rescan_all=rescan_all,
                        generate_previews=generate_previews,
                        raw_preview_mode=raw_preview_mode,
                        max_decode_pixels=max_decode_pixels,
                        existing_metadata=existing_rows.get(canonical_path_key(path)),
                    )
                )
            except Exception as exc:
                outcome.failures.append(str(exc))
    except InterruptedError:
        outcome.cancel_error = "Scan job was cancelled by user."
    return outcome


def _process_parallel_batch(
    paths: Sequence[Path],
    max_workers: int,
    *,
    preview_dir: Path,
    rescan_all: bool,
    generate_previews: bool,
    raw_preview_mode: str,
    max_decode_pixels: int,
    existing_rows: dict[str, dict],
    executor: concurrent.futures.ProcessPoolExecutor | None,
    cancel_check: Callable[[], None] | None,
) -> _BatchProcessingOutcome:
    """Gather a batch with bounded in-flight work and cancellation support."""
    pool = executor
    owns_pool = pool is None
    if owns_pool:
        pool = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)
    assert pool is not None

    outcome = _BatchProcessingOutcome()
    futures: dict[concurrent.futures.Future, int] = {}
    result_by_index: dict[int, dict] = {}
    failure_by_index: dict[int, str] = {}
    next_path_index = 0
    path_iter = iter(paths)

    def submit_until_full() -> None:
        nonlocal next_path_index
        while outcome.cancel_error is None and len(futures) < max_workers:
            _run_cancel_check(cancel_check)
            try:
                path = next(path_iter)
            except StopIteration:
                return
            future = pool.submit(
                gather_file_metadata,
                path,
                preview_dir=preview_dir,
                rescan_all=rescan_all,
                generate_previews=generate_previews,
                raw_preview_mode=raw_preview_mode,
                max_decode_pixels=max_decode_pixels,
                existing_metadata=existing_rows.get(canonical_path_key(path)),
            )
            futures[future] = next_path_index
            next_path_index += 1

    try:
        try:
            submit_until_full()
        except InterruptedError as exc:
            outcome.cancel_error = str(exc)

        while futures:
            done, still_pending = concurrent.futures.wait(
                futures,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            done_indices = {future: futures[future] for future in done}
            futures = {future: futures[future] for future in still_pending}

            if outcome.cancel_error is None:
                try:
                    _run_cancel_check(cancel_check)
                except InterruptedError as exc:
                    outcome.cancel_error = str(exc)
                    for pending_future in futures:
                        pending_future.cancel()

            for future in done:
                if future.cancelled():
                    continue

                path_index = done_indices[future]
                outcome.attempted_count += 1
                try:
                    result_by_index[path_index] = future.result()
                except concurrent.futures.CancelledError:
                    continue
                except Exception as exc:
                    failure_by_index[path_index] = str(exc)

            if outcome.cancel_error is None:
                try:
                    submit_until_full()
                except InterruptedError as exc:
                    outcome.cancel_error = str(exc)
                    for pending_future in futures:
                        pending_future.cancel()
    finally:
        if owns_pool:
            pool.shutdown(wait=True)

    outcome.results.extend(result_by_index[index] for index in sorted(result_by_index))
    outcome.failures.extend(failure_by_index[index] for index in sorted(failure_by_index))

    return outcome


def _account_batch_outcome(
    connection,
    summary: ScanSummary,
    outcome: _BatchProcessingOutcome,
    *,
    existing_rows: dict[str, dict],
) -> None:
    """Apply shared failure, persistence, and cancellation accounting."""
    for error_text in outcome.failures:
        summary.files_failed += 1
        summary.last_batch_error = error_text

    if outcome.results:
        commit_batch(connection, outcome.results, summary, existing_rows=existing_rows)

    if outcome.cancel_error is not None:
        raise ScanBatchInterrupted(outcome.cancel_error, attempted_count=outcome.attempted_count)


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
            "preview_conversion_version": row["preview_conversion_version"],
            "preview_path": row["preview_path"],
            "last_error": row["last_error"],
            "analysis_status": row["analysis_status"],
            "analysis_error": row["analysis_error"],
            "last_analysis_time": row["last_analysis_time"],
        }
        for row in connection.execute(
            f"SELECT path_key, modified_time, size_bytes, width, height, capture_time, preview_status, preview_conversion_version, preview_path, last_error, analysis_status, analysis_error, last_analysis_time FROM files WHERE path_key IN ({placeholders})",
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
    return (
        existing_metadata.get("preview_status") != "ready"
        or existing_metadata.get("preview_conversion_version") != IMAGE_CONVERSION_VERSION
        or not existing_preview_exists
    )


def gather_file_metadata(
    path: Path,
    *,
    preview_dir: Path,
    rescan_all: bool,
    generate_previews: bool = True,
    raw_preview_mode: str = DEFAULT_RAW_PREVIEW_MODE,
    max_decode_pixels: int = DEFAULT_MAX_DECODE_PIXELS,
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
        and existing_metadata.get("preview_conversion_version") == IMAGE_CONVERSION_VERSION
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
            "preview_conversion_version": IMAGE_CONVERSION_VERSION,
            "last_error": existing_metadata.get("last_error"),
            "scan_status": "unchanged",
            "analysis_status": None,
            "analysis_error": None,
            "last_analysis_time": None,
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
        "preview_conversion_version": None,
        "last_error": None,
        "scan_status": base_scan_status,
        "analysis_status": "pending" if metadata_changed or existing_metadata is None else existing_metadata.get("analysis_status"),
        "analysis_error": None if metadata_changed or existing_metadata is None else existing_metadata.get("analysis_error"),
        "last_analysis_time": None if metadata_changed or existing_metadata is None else existing_metadata.get("last_analysis_time"),
        "preserve_metadata": False,
    }

    if generate_previews:
        # Previews are CPU intensive, but generate_preview handles its own errors
        preview = generate_preview(
            path,
            preview_dir,
            raw_preview_mode=raw_preview_mode,
            max_decode_pixels=max_decode_pixels,
        )
        metadata.update({
            "preview_path": preview.path,
            "preview_status": preview.status,
            "preview_conversion_version": IMAGE_CONVERSION_VERSION if preview.status == "ready" else None,
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
            preview_conversion_version,
            last_scan_time, last_error, scan_status,
            analysis_status, analysis_error, last_analysis_time
        )
        VALUES(
            :path, :path_key, :size_bytes, :modified_time, :format, 
            :width, :height, :capture_time, :preview_path, :preview_status,
            :preview_conversion_version,
            :last_scan_time, :last_error, :scan_status,
            :analysis_status, :analysis_error, :last_analysis_time
        )
        ON CONFLICT(path_key) DO UPDATE SET
            path = excluded.path,
            move_managed = 0,
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
                preview_conversion_version = CASE
                WHEN :preserve_metadata THEN COALESCE(excluded.preview_conversion_version, preview_conversion_version)
                    ELSE excluded.preview_conversion_version
                END,
            last_scan_time = excluded.last_scan_time,
                last_error = CASE
                WHEN :preserve_metadata THEN COALESCE(excluded.last_error, last_error)
                    ELSE excluded.last_error
                END,
                scan_status = excluded.scan_status,
                analysis_status = CASE
                    WHEN :preserve_metadata THEN analysis_status
                        ELSE excluded.analysis_status
                    END,
                analysis_error = CASE
                    WHEN :preserve_metadata THEN analysis_error
                        ELSE excluded.analysis_error
                    END,
                last_analysis_time = CASE
                    WHEN :preserve_metadata THEN last_analysis_time
                        ELSE excluded.last_analysis_time
                    END
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
        if _preview_marker_exists(parent / ".shotsieve-preview-root"):
            claimed_preview_roots.add(parent)
            return True
    return False


def check_overlapping_roots(roots: Sequence[Path]) -> list[tuple[Path, Path]]:
    """Return list of overlapping root pairs (parent, child)."""
    resolved_paths = sorted([r.expanduser().resolve() for r in roots], key=lambda p: len(p.parts))
    overlaps = []
    for i, path in enumerate(resolved_paths):
        for other in resolved_paths[i + 1:]:
            try:
                other.relative_to(path)
                overlaps.append((path, other))
            except ValueError:
                pass
    return overlaps
