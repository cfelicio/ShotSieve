"""File export operations - copy or move selected files to a destination folder."""
from __future__ import annotations

import errno
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from shotsieve.db import infer_preview_cache_roots, normalize_path_case
from shotsieve.models import (
    FilesystemObservation,
    FileOperationResult,
    FileOperationSummary,
    attach_file_operation_summary,
    observe_filesystem_path,
)
from shotsieve.preview import delete_managed_preview_file
from shotsieve.scanner import canonical_path_key


@dataclass(slots=True)
class ExportSummary(FileOperationSummary):
    action: str = "export"
    contract_enabled: bool = True


class _PathObservationError(OSError):
    """Raised when a target cannot be safely classified before transfer."""

    def __init__(self, path: Path, observation: FilesystemObservation) -> None:
        super().__init__(observation.error_text or f"Unable to inspect {path}")
        self.path = path
        self.observation = observation


class _TransferCleanupError(OSError):
    """A transfer failed and its best-effort destination cleanup also failed."""

    def __init__(self, transfer_error: BaseException, cleanup_error: BaseException) -> None:
        super().__init__(f"{transfer_error}; destination cleanup failed: {cleanup_error}")
        self.transfer_error = transfer_error
        self.cleanup_error = cleanup_error


def export_files(
    connection,
    *,
    file_ids: list[int],
    destination: str,
    mode: str = "copy",
    preview_cache_root: Path | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    cancel_check: Callable[[], None] | None = None,
) -> ExportSummary:
    """Copy or move files to a destination directory.

    Args:
        connection: SQLite database connection.
        file_ids: List of file IDs to export.
        destination: Destination directory path.
        mode: "copy" or "move".
        preview_cache_root: Root directory of preview cache. Used to safely
            clean up old preview files after a move operation.

    Returns:
        ExportSummary with counts and any failures.

    Raises:
        ValueError: If mode is invalid, destination doesn't exist, or any
            file_ids are not found in the database.
    """
    if mode not in ("copy", "move"):
        raise ValueError(f"Export mode must be 'copy' or 'move', got '{mode}'")

    dest_path = Path(destination).expanduser().resolve()
    if not dest_path.is_absolute():
        dest_path = Path.cwd() / dest_path
    if not dest_path.is_dir():
        raise ValueError(f"Destination directory does not exist: {dest_path}")

    # Defense-in-depth: refuse system-critical directories.
    _reject_system_directory(dest_path)

    if not file_ids:
        return ExportSummary()

    # Deduplicate and normalize IDs.
    unique_ids = sorted(set(file_ids))

    placeholders = ",".join("?" for _ in unique_ids)
    rows = connection.execute(
        f"SELECT id, path, preview_path FROM files WHERE id IN ({placeholders}) ORDER BY id",
        unique_ids,
    ).fetchall()

    # Validate: all requested IDs must exist in the database.
    found_ids = {row["id"] for row in rows}
    missing_ids = [fid for fid in unique_ids if fid not in found_ids]
    if missing_ids:
        raise ValueError(f"File IDs not found in database: {missing_ids}")

    allow_preview_path_fallback = (
        preview_cache_root is not None
        and len(infer_preview_cache_roots(connection)) > 1
    )

    summary = ExportSummary()
    total_files = len(rows)

    if progress_callback is not None:
        try:
            progress_callback(0, total_files)
        except BaseException as exc:
            _stop_with_unprocessed(summary, rows, 0, exc, mode=mode, cancelled=isinstance(exc, InterruptedError))

    for index, row in enumerate(rows, start=1):
        if cancel_check is not None:
            try:
                cancel_check()
            except BaseException as exc:
                _stop_with_unprocessed(
                    summary,
                    rows,
                    index - 1,
                    exc,
                    mode=mode,
                    cancelled=isinstance(exc, InterruptedError),
                )

        source = Path(row["path"])
        source_observation = observe_filesystem_path(source)
        if source_observation.state != "present":
            source_error = source_observation.error_text or "Source file not found"
            summary.add(
                _result_from_error(
                    row,
                    action=mode,
                    destination=None,
                    stage="source_check",
                    error=source_error,
                    retry_safe=source_observation.state == "missing",
                    fallback="Source file not found",
                    source_observation=source_observation,
                )
            )
            _emit_progress(progress_callback, index, total_files, summary, rows, index, mode)
            continue

        try:
            target = _resolve_target(dest_path, source.name)
        except _PathObservationError as exc:
            summary.add(
                _result_from_error(
                    row,
                    action=mode,
                    destination=str(exc.path),
                    stage="destination_selection",
                    error=exc,
                    retry_safe=True,
                    source_observation=source_observation,
                    destination_observation=exc.observation,
                )
            )
            _emit_progress(progress_callback, index, total_files, summary, rows, index, mode)
            continue
        except (OSError, ValueError) as exc:
            summary.add(
                _result_from_error(
                    row,
                    action=mode,
                    destination=None,
                    stage="destination_selection",
                    error=exc,
                    retry_safe=True,
                    source_observation=source_observation,
                )
            )
            _emit_progress(progress_callback, index, total_files, summary, rows, index, mode)
            continue

        if mode == "copy":
            try:
                _copy_without_overwrite(source, target)
                summary.copied += 1
                summary.add(
                    FileOperationResult(
                        file_id=int(row["id"]),
                        source=str(source),
                        destination=str(target),
                        action=mode,
                        outcome="success",
                        stage="transfer",
                    )
                )
            except OSError as exc:
                collision = isinstance(exc, FileExistsError)
                destination_observation = observe_filesystem_path(target)
                if collision:
                    outcome = "failed"
                    retry_safe = True
                else:
                    outcome = (
                        "failed"
                        if destination_observation.state == "missing"
                        else "uncertain"
                    )
                    retry_safe = outcome == "failed"
                summary.add(
                    _result_from_error(
                        row,
                        action=mode,
                        destination=str(target),
                        stage="destination_selection" if collision else "transfer",
                        error=exc,
                        retry_safe=retry_safe,
                        outcome=outcome,
                        source_observation=observe_filesystem_path(source),
                        destination_observation=destination_observation,
                    )
                )
            except Exception as exc:
                destination_observation = observe_filesystem_path(target)
                summary.add(
                    _result_from_error(
                        row,
                        action=mode,
                        destination=str(target),
                        stage="transfer",
                        error=exc,
                        retry_safe=False,
                        outcome="uncertain",
                        source_observation=observe_filesystem_path(source),
                        destination_observation=destination_observation,
                    )
                )
                _stop_with_unprocessed(summary, rows, index, exc, mode=mode, cancelled=False)
            _emit_progress(progress_callback, index, total_files, summary, rows, index, mode)
            continue

        try:
            _move_without_overwrite(source, target)
        except OSError as exc:
            collision = isinstance(exc, FileExistsError)
            source_after = observe_filesystem_path(source)
            destination_after = observe_filesystem_path(target)
            if collision:
                outcome = "failed"
                retry_safe = True
            else:
                outcome = (
                    "failed"
                    if source_after.state == "present" and destination_after.state == "missing"
                    else "uncertain"
                )
                retry_safe = outcome == "failed"
            summary.add(
                _result_from_error(
                    row,
                    action=mode,
                    destination=str(target),
                    stage="destination_selection" if collision else "transfer",
                    error=exc,
                    retry_safe=retry_safe,
                    outcome=outcome,
                    source_observation=source_after,
                    destination_observation=destination_after,
                )
            )
            _emit_progress(progress_callback, index, total_files, summary, rows, index, mode)
            continue
        except Exception as exc:
            source_after = observe_filesystem_path(source)
            destination_after = observe_filesystem_path(target)
            summary.add(
                _result_from_error(
                    row,
                    action=mode,
                    destination=str(target),
                    stage="transfer",
                    error=exc,
                    retry_safe=False,
                    outcome="uncertain",
                    source_observation=source_after,
                    destination_observation=destination_after,
                )
            )
            _stop_with_unprocessed(summary, rows, index, exc, mode=mode, cancelled=False)

        try:
            # Update the cached path and clear preview so it gets regenerated.
            connection.execute(
                "UPDATE files SET path = ?, path_key = ?, preview_path = NULL, preview_status = 'missing' WHERE id = ?",
                (str(target), canonical_path_key(target), row["id"]),
            )

            # Persist successful move rows immediately so a later row's
            # database failure cannot roll back paths for files that have
            # already been moved on disk.
            connection.commit()
        except BaseException as exc:
            try:
                result, needs_rollback = _reconcile_move_catalog_failure(
                    connection,
                    row=row,
                    source=source,
                    target=target,
                    catalog_error=exc,
                )
            except BaseException as restore_error:
                summary.add(_result_from_error(
                    row,
                    action=mode,
                    destination=str(target),
                    stage="catalog_update",
                    error=f"{exc}; compensation failed: {restore_error}",
                    retry_safe=False,
                    outcome="catalog_failed",
                    source_observation=observe_filesystem_path(source),
                    destination_observation=observe_filesystem_path(target),
                ))
                _append_unprocessed_rows(
                    summary,
                    rows[index:],
                    action=mode,
                    error=exc,
                )
                attach_file_operation_summary(restore_error, summary, needs_rollback=True)
                try:
                    setattr(restore_error, "catalog_error", exc)
                except Exception:
                    pass
                raise restore_error from exc

            summary.add(result)
            _append_unprocessed_rows(
                summary,
                rows[index:],
                action=mode,
                error=exc,
            )
            attach_file_operation_summary(exc, summary, needs_rollback=needs_rollback)
            raise

        summary.moved += 1
        summary.add(
            FileOperationResult(
                file_id=int(row["id"]),
                source=str(source),
                destination=str(target),
                action=mode,
                outcome="success",
                stage="catalog_update",
            )
        )

        try:
            # Clean up old preview file only after the database update succeeds.
            delete_managed_preview_file(
                row["preview_path"],
                source_path=row["path"],
                preview_cache_root=preview_cache_root,
                allow_path_parent_fallback=allow_preview_path_fallback,
            )
        except Exception as exc:
            summary.add_warning(
                file_id=int(row["id"]),
                source=str(source),
                stage="preview_cleanup",
                error=exc,
            )

        _emit_progress(progress_callback, index, total_files, summary, rows, index, mode)

    return summary


def _append_unprocessed_rows(
    summary: ExportSummary,
    rows,
    *,
    action: str,
    error: BaseException,
) -> None:
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
                action=action,
                outcome="unprocessed",
                stage="not_started",
                error_text=str(error),
                retry_safe=True,
            )
        )
        accounted_ids.add(file_id)


def _rollback_catalog_transaction(connection) -> str | None:
    try:
        connection.rollback()
    except BaseException as exc:
        return str(exc) or exc.__class__.__name__
    return None


def _read_catalog_row(connection, file_id: int):
    try:
        row = connection.execute(
            "SELECT path, preview_path FROM files WHERE id = ?",
            (file_id,),
        ).fetchone()
    except BaseException as exc:
        return None, str(exc) or exc.__class__.__name__
    return row, None


def _same_catalog_path(value: object, path: Path) -> bool:
    if not isinstance(value, str):
        return False
    try:
        return normalize_path_case(str(Path(value).resolve())) == normalize_path_case(str(path.resolve()))
    except OSError:
        return normalize_path_case(value) == normalize_path_case(str(path))


def _catalog_failure_text(
    catalog_error: BaseException,
    *,
    rollback_error: str | None,
    catalog_read_error: str | None,
) -> str:
    details = [str(catalog_error) or catalog_error.__class__.__name__]
    if rollback_error:
        details.append(f"catalog rollback failed: {rollback_error}")
    if catalog_read_error:
        details.append(f"catalog observation failed: {catalog_read_error}")
    return "; ".join(details)


def _reconcile_move_catalog_failure(
    connection,
    *,
    row,
    source: Path,
    target: Path,
    catalog_error: BaseException,
) -> tuple[FileOperationResult, bool]:
    """Reconcile a move after an UPDATE/COMMIT error.

    A move changes the filesystem before the catalog.  The transaction is
    rolled back first, then the row is read back before any compensation is
    attempted.  This avoids restoring the file underneath a catalog update
    that may have committed despite reporting an error.
    """
    rollback_error = _rollback_catalog_transaction(connection)
    catalog_row, catalog_read_error = _read_catalog_row(connection, int(row["id"]))
    source_observation = observe_filesystem_path(source)
    destination_observation = observe_filesystem_path(target)
    error_text = _catalog_failure_text(
        catalog_error,
        rollback_error=rollback_error,
        catalog_read_error=catalog_read_error,
    )

    if catalog_row is not None and _same_catalog_path(catalog_row["path"], target):
        # The catalog agrees with the moved bytes.  The caller still receives
        # the original error, but must not retry or compensate this file.
        return (
            _result_from_error(
                row,
                action="move",
                destination=str(target),
                stage="catalog_update",
                error=error_text,
                retry_safe=False,
                outcome="uncertain",
                source_observation=source_observation,
                destination_observation=destination_observation,
            ),
            False,
        )

    if catalog_row is not None and _same_catalog_path(catalog_row["path"], Path(row["path"])):
        try:
            _restore_moved_source(source, target)
        except BaseException as restore_error:
            try:
                setattr(restore_error, "file_operation_recovery_error", str(restore_error))
            except Exception:
                pass
            raise

        source_observation = observe_filesystem_path(source)
        destination_observation = observe_filesystem_path(target)
        return (
            _result_from_error(
                row,
                action="move",
                destination=None,
                stage="catalog_update",
                error=error_text,
                retry_safe=(
                    source_observation.state == "present"
                    and destination_observation.state == "missing"
                    and rollback_error is None
                ),
                outcome="failed",
                source_observation=source_observation,
                destination_observation=destination_observation,
            ),
            rollback_error is not None,
        )

    # We cannot prove which catalog state won, so leave both paths visible in
    # the result and force manual inspection.  The outer owner must roll back
    # again rather than committing a transaction whose state was not read.
    return (
        _result_from_error(
            row,
            action="move",
            destination=str(target),
            stage="catalog_update",
            error=error_text,
            retry_safe=False,
            outcome="uncertain",
            source_observation=source_observation,
            destination_observation=destination_observation,
        ),
        True,
    )


def _result_from_error(
    row,
    *,
    action: str,
    destination: str | None,
    stage: str,
    error: BaseException | str,
    retry_safe: bool,
    fallback: str | None = None,
    outcome: str = "failed",
    source_observation: FilesystemObservation | None = None,
    destination_observation: FilesystemObservation | None = None,
) -> FileOperationResult:
    error_text = str(error) or fallback
    if fallback and isinstance(error, FileNotFoundError):
        error_text = fallback
    return FileOperationResult(
        file_id=int(row["id"]),
        source=str(row["path"]),
        destination=destination,
        action=action,
        outcome=outcome,
        stage=stage,
        error_text=error_text,
        errno=getattr(error, "errno", None),
        winerror=getattr(error, "winerror", None),
        retry_safe=retry_safe,
        source_state=source_observation.state if source_observation is not None else "unknown",
        destination_state=(
            destination_observation.state if destination_observation is not None else None
        ),
        observation_errors=_observation_errors(
            source_observation=source_observation,
            destination_observation=destination_observation,
            source=str(row["path"]),
            destination=destination,
        ),
    )


def _observation_errors(
    *,
    source_observation: FilesystemObservation | None,
    destination_observation: FilesystemObservation | None,
    source: str,
    destination: str | None,
) -> list[str]:
    errors: list[str] = []
    if source_observation is not None and source_observation.error_text:
        errors.append(f"source '{source}': {source_observation.error_text}")
    if destination_observation is not None and destination_observation.error_text:
        target = destination or "<unknown destination>"
        errors.append(f"destination '{target}': {destination_observation.error_text}")
    return errors


def _emit_progress(
    progress_callback: Callable[[int, int], None] | None,
    processed: int,
    total: int,
    summary: ExportSummary,
    rows,
    next_index: int,
    mode: str,
) -> None:
    if progress_callback is None:
        return
    try:
        progress_callback(processed, total)
    except BaseException as exc:
        _stop_with_unprocessed(
            summary,
            rows,
            next_index,
            exc,
            mode=mode,
            cancelled=isinstance(exc, InterruptedError),
        )


def _stop_with_unprocessed(
    summary: ExportSummary,
    rows,
    start_index: int,
    error: BaseException,
    *,
    mode: str,
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
                action=mode,
                outcome="unprocessed",
                stage="not_started",
                error_text=str(error),
                retry_safe=True,
            )
        )
        accounted_ids.add(file_id)
    attach_file_operation_summary(error, summary, cancelled=cancelled)
    raise error


def _copy_without_overwrite(source: Path, target: Path) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    descriptor = os.open(target, flags, 0o666)
    created = True
    try:
        with os.fdopen(descriptor, "wb") as destination_stream, source.open("rb") as source_stream:
            shutil.copyfileobj(source_stream, destination_stream)
        shutil.copystat(source, target)
    except BaseException as transfer_error:
        if created:
            try:
                target.unlink()
            except FileNotFoundError:
                pass
            except BaseException as cleanup_error:
                raise _TransferCleanupError(transfer_error, cleanup_error) from transfer_error
        raise


def _move_without_overwrite(source: Path, target: Path) -> None:
    try:
        os.link(source, target)
    except OSError as exc:
        if exc.errno not in {errno.EXDEV, errno.ENOSYS, errno.EOPNOTSUPP} and getattr(exc, "winerror", None) not in {1, 50}:
            raise
        _copy_without_overwrite(source, target)
    try:
        source.unlink()
    except OSError:
        # A cross-device move can leave both paths.  The caller records this
        # as partial/uncertain and deliberately does not retry it implicitly.
        raise


def _resolve_target(dest_dir: Path, filename: str) -> Path:
    """Find a non-colliding target path, adding _2, _3, etc. if needed."""
    target = dest_dir / filename
    observation = observe_filesystem_path(target)
    if observation.state == "missing":
        return target
    if observation.state == "unknown":
        raise _PathObservationError(target, observation)

    stem = target.stem
    suffix = target.suffix
    max_attempts = 10_000
    for counter in range(2, 2 + max_attempts):
        candidate = dest_dir / f"{stem}_{counter}{suffix}"
        observation = observe_filesystem_path(candidate)
        if observation.state == "missing":
            return candidate
        if observation.state == "unknown":
            raise _PathObservationError(candidate, observation)
    raise ValueError(f"Could not find a unique filename for {filename} after {max_attempts} attempts")


def _restore_moved_source(source: Path, target: Path) -> None:
    target_observation = observe_filesystem_path(target)
    if target_observation.state == "missing":
        return
    if target_observation.state == "unknown":
        raise _PathObservationError(target, target_observation)

    source_observation = observe_filesystem_path(source)
    if source_observation.state == "present":
        raise FileExistsError(f"Cannot restore moved file without replacing existing source: {source}")
    if source_observation.state == "unknown":
        raise _PathObservationError(source, source_observation)
    _move_without_overwrite(target, source)


def _reject_system_directory(dest: Path) -> None:
    """Raise ValueError if *dest* is inside a system-critical directory."""
    import platform

    system = platform.system()
    resolved = normalize_path_case(str(dest.resolve()))

    if system == "Windows":
        blocked = (
            normalize_path_case("c:\\windows"),
            normalize_path_case("c:\\program files"),
            normalize_path_case("c:\\program files (x86)"),
        )
    else:
        blocked = (
            normalize_path_case("/bin"),
            normalize_path_case("/sbin"),
            normalize_path_case("/usr/bin"),
            normalize_path_case("/usr/sbin"),
            normalize_path_case("/boot"),
            normalize_path_case("/proc"),
            normalize_path_case("/sys"),
        )

    for prefix in blocked:
        if resolved == prefix or resolved.startswith(prefix + ("/" if system != "Windows" else "\\")):
            raise ValueError(f"Cannot export to system directory: {dest}")
