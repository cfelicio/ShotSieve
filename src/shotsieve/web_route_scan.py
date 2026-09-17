"""Scan-job orchestration used by the web route adapters."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from shotsieve.db import (
    attach_scan_run_diagnostic,
    mark_scan_run_diagnostic_persisted,
    persist_scan_run_diagnostic,
    scan_run_diagnostic_from_exception,
    scan_run_diagnostic_was_persisted,
)
from shotsieve.job_registry import JobRegistry
from shotsieve.models import ScanRunDiagnostic
from shotsieve.web_request import ScanRequest
from shotsieve.web_route_common import WebRouteContext, WebRouteDependencies


@dataclass(frozen=True, slots=True)
class _ScanJobRequest:
    """Immutable scan inputs handed from the HTTP adapter to the worker."""

    roots: tuple[Path, ...]
    preview_dir: str | None
    extensions: str | None
    limit: int | None
    offset: int
    recursive: bool
    rescan_all: bool
    generate_previews: bool
    preview_mode: str
    files_total_hint: int
    resource_profile: str | None
    ignore_rules: tuple[str, ...]
    max_decode_pixels: int

    @classmethod
    def from_scan_request(cls, scan_request: ScanRequest) -> _ScanJobRequest:
        """Copy the parsed request before the asynchronous worker starts."""
        return cls(
            roots=tuple(scan_request["roots"]),
            preview_dir=scan_request["preview_dir"],
            extensions=scan_request["extensions"],
            limit=scan_request["limit"],
            offset=max(0, scan_request["offset"]),
            recursive=scan_request["recursive"],
            rescan_all=scan_request["rescan_all"],
            generate_previews=scan_request["generate_previews"],
            preview_mode=scan_request["preview_mode"],
            files_total_hint=max(0, scan_request["files_total_hint"]),
            resource_profile=scan_request["resource_profile"],
            ignore_rules=tuple(scan_request["ignore_rules"]),
            max_decode_pixels=scan_request["max_decode_pixels"],
        )


@dataclass(frozen=True, slots=True)
class _ScanRootAttempt:
    """Outcome of one root, including the context needed for failure reports."""

    root: Path
    started_time: str
    processed_before_root: int
    summary: Any = None
    exception: BaseException | None = None
    not_processed_reason: str | None = None


@dataclass(frozen=True, slots=True)
class _ScanProgressPublisher:
    """Translate one-root scanner progress into job-level progress."""

    registry: JobRegistry
    job_id: str
    processed_before_root: int
    total_hint: int

    def __call__(self, processed_in_root: int, _root_total: int, phase: str) -> None:
        files_total = self.total_hint if self.total_hint > 0 else 0
        self.registry.update_progress(
            self.job_id,
            {
                "phase": phase,
                "files_processed": max(0, self.processed_before_root + processed_in_root),
                "files_total": files_total,
            },
        )


@dataclass(frozen=True, slots=True)
class _ScanCancellationCheck:
    """Adapt the route-level cancellation helper to the scanner callback API."""

    registry: JobRegistry
    job_id: str
    raise_if_cancelled: Callable[[JobRegistry, str], None]

    def __call__(self) -> None:
        self.raise_if_cancelled(self.registry, self.job_id)


def _scan_offset_consumed(summary: Any, *, requested_offset: int) -> int:
    consumed = getattr(summary, "offset_consumed", None)
    if isinstance(consumed, int):
        return max(0, min(requested_offset, consumed))

    files_seen = int(getattr(summary, "files_seen", 0) or 0)
    if requested_offset > 0 and files_seen > 0:
        return requested_offset
    return 0


def _raise_if_scan_cancelled(registry: JobRegistry, job_id: str) -> None:
    if registry.is_cancelled(job_id):
        raise InterruptedError("Scan job was cancelled by user.")


def _resolved_scan_root(root: Path) -> str:
    try:
        return str(root.expanduser().resolve())
    except OSError:
        return str(root.expanduser())


def _scan_root_report(
    root: Path,
    *,
    status: str,
    summary: Any = None,
    error_text: str | None = None,
) -> dict[str, object]:
    return {
        "root_path": _resolved_scan_root(root),
        "status": status,
        "files_seen": int(getattr(summary, "files_seen", 0) or 0),
        "files_added": int(getattr(summary, "files_added", 0) or 0),
        "files_updated": int(getattr(summary, "files_updated", 0) or 0),
        "files_unchanged": int(getattr(summary, "files_unchanged", 0) or 0),
        "files_removed": int(getattr(summary, "files_removed", 0) or 0),
        "files_failed": int(getattr(summary, "files_failed", 0) or 0),
        "error_text": error_text,
    }


def _scan_root_report_from_exception(
    root: Path,
    exc: BaseException,
    *,
    started_time: str,
    db_path: Path,
) -> dict[str, object]:
    diagnostic = scan_run_diagnostic_from_exception(exc)
    if diagnostic is None:
        diagnostic = ScanRunDiagnostic(
            root_path=_resolved_scan_root(root),
            started_time=started_time,
            completed_time=started_time,
            status="failed",
            files_seen=max(0, int(getattr(exc, "processed_count", 0) or 0)),
            error_text=str(exc) or exc.__class__.__name__,
        )
        attach_scan_run_diagnostic(exc, diagnostic)

    persistence_error: str | None = None
    if not scan_run_diagnostic_was_persisted(exc):
        try:
            persist_scan_run_diagnostic(db_path, diagnostic)
        except Exception as exc_persist:
            persistence_error = str(exc_persist) or exc_persist.__class__.__name__
        else:
            mark_scan_run_diagnostic_persisted(exc)

    error_text = diagnostic.error_text
    if persistence_error:
        error_text = f"{error_text or exc.__class__.__name__}; failed to persist scan diagnostic: {persistence_error}"

    return {
        "root_path": diagnostic.root_path,
        "status": "cancelled" if isinstance(exc, InterruptedError) else diagnostic.status,
        "files_seen": diagnostic.files_seen,
        "files_added": diagnostic.files_added,
        "files_updated": diagnostic.files_updated,
        "files_unchanged": diagnostic.files_unchanged,
        "files_removed": diagnostic.files_removed,
        "files_failed": 0,
        "error_text": error_text,
    }


def _scan_job_summary(
    aggregated: dict[str, int],
    root_results: list[dict[str, object]],
    *,
    overall_status: str,
) -> dict[str, object]:
    return {
        **aggregated,
        "root_results": root_results,
        "overall_status": overall_status,
    }


def _scan_one_root(
    root: Path,
    *,
    context: WebRouteContext,
    request: _ScanJobRequest,
    config: Any,
    registry: JobRegistry,
    job_id: str,
    processed_before_root: int,
    remaining_offset: int,
    remaining_limit: int | None,
    raise_if_cancelled: Callable[[JobRegistry, str], None] = _raise_if_scan_cancelled,
) -> _ScanRootAttempt:
    """Run one root in its own transaction and return its explicit outcome."""
    deps = cast(WebRouteDependencies, context.dependencies)
    started_time = deps.utc_now()
    try:
        raise_if_cancelled(registry, job_id)
        root_total_hint = None
        if request.files_total_hint > 0:
            root_total_hint = max(0, request.files_total_hint - processed_before_root)

        root_exception: BaseException | None = None
        with deps.database(config.db_path) as connection:
            try:
                summary = deps.scan_root(
                    connection,
                    root=root,
                    recursive=request.recursive,
                    limit=remaining_limit,
                    offset=remaining_offset,
                    extensions=config.supported_extensions,
                    preview_dir=config.preview_dir,
                    rescan_all=request.rescan_all,
                    generate_previews=request.generate_previews,
                    raw_preview_mode=config.raw_preview_mode,
                    max_decode_pixels=request.max_decode_pixels,
                    resource_profile=request.resource_profile,
                    progress_callback=_ScanProgressPublisher(
                        registry=registry,
                        job_id=job_id,
                        processed_before_root=processed_before_root,
                        total_hint=request.files_total_hint,
                    ),
                    files_total_hint=root_total_hint,
                    cancel_check=_ScanCancellationCheck(
                        registry=registry,
                        job_id=job_id,
                        raise_if_cancelled=raise_if_cancelled,
                    ),
                    # Keep the scanner's historical mutable-sequence payload
                    # contract at this boundary; the worker request itself
                    # remains immutable.
                    ignore_rules=list(request.ignore_rules),
                )
            except InterruptedError as exc:
                # Let the database context finish its rollback-safe diagnostic
                # persistence before the worker finalizer sees the exception.
                root_exception = exc
        if root_exception is not None:
            raise root_exception
    except Exception as exc:
        return _ScanRootAttempt(
            root=root,
            started_time=started_time,
            processed_before_root=processed_before_root,
            exception=exc,
        )

    return _ScanRootAttempt(
        root=root,
        started_time=started_time,
        processed_before_root=processed_before_root,
        summary=summary,
    )


def _not_processed_root(root: Path, *, reason: str) -> _ScanRootAttempt:
    return _ScanRootAttempt(
        root=root,
        started_time="",
        processed_before_root=0,
        not_processed_reason=reason,
    )


def _finalize_scan_job(
    context: WebRouteContext,
    request: _ScanJobRequest,
    registry: JobRegistry,
    job_id: str,
    aggregated: dict[str, int],
    attempts: list[_ScanRootAttempt],
    *,
    unexpected_error: BaseException | None = None,
) -> None:
    """Build root diagnostics and complete or fail the public scan job."""
    root_results: list[dict[str, object]] = []
    failed_attempt: _ScanRootAttempt | None = None
    failed_report: dict[str, object] | None = None

    for attempt in attempts:
        if attempt.exception is not None:
            failed_report = _scan_root_report_from_exception(
                attempt.root,
                attempt.exception,
                started_time=attempt.started_time,
                db_path=context.db_path,
            )
            root_results.append(failed_report)
            failed_attempt = attempt
            continue
        if attempt.not_processed_reason is not None:
            root_results.append(
                _scan_root_report(
                    attempt.root,
                    status="not_processed",
                    error_text=attempt.not_processed_reason,
                )
            )
            continue

        summary = attempt.summary
        root_results.append(
            _scan_root_report(
                attempt.root,
                status="completed_with_errors" if getattr(summary, "files_failed", 0) else "completed",
                summary=summary,
                error_text=getattr(summary, "last_batch_error", None),
            )
        )

    if unexpected_error is not None:
        registry.fail(
            job_id,
            error=str(unexpected_error) or unexpected_error.__class__.__name__,
            summary=_scan_job_summary(aggregated, root_results, overall_status="failed"),
        )
        return

    if failed_attempt is not None and failed_report is not None:
        failure_error = str(failed_report.get("error_text") or failed_attempt.exception) or failed_attempt.exception.__class__.__name__
        files_processed = failed_attempt.processed_before_root + int(failed_report.get("files_seen", 0) or 0)
        failure_progress = {
            "phase": "failed",
            "files_processed": max(0, files_processed),
            "files_total": request.files_total_hint if request.files_total_hint > 0 else max(0, files_processed),
        }
        registry.fail(
            job_id,
            error=failure_error,
            progress=failure_progress,
            summary=_scan_job_summary(aggregated, root_results, overall_status="failed"),
        )
        return

    registry.update_progress(
        job_id,
        {
            "phase": "scanning",
            "files_processed": aggregated["files_seen"],
            "files_total": request.files_total_hint if request.files_total_hint > 0 else aggregated["files_seen"],
        },
    )
    overall_status = (
        "completed_with_errors"
        if any(item["status"] == "completed_with_errors" for item in root_results)
        else "completed"
    )
    registry.complete(
        job_id,
        summary=_scan_job_summary(aggregated, root_results, overall_status=overall_status),
    )


def _run_scan_job(
    context: WebRouteContext,
    request: _ScanJobRequest,
    registry: JobRegistry,
    job_id: str,
) -> None:
    """Run all roots while keeping sequencing and pagination explicit."""
    deps = cast(WebRouteDependencies, context.dependency_views.jobs)
    aggregated = {
        "files_seen": 0,
        "files_added": 0,
        "files_updated": 0,
        "files_unchanged": 0,
        "files_removed": 0,
        "files_failed": 0,
    }
    attempts: list[_ScanRootAttempt] = []

    try:
        config = deps.build_config(
            str(context.db_path),
            raw_preview_dir=request.preview_dir,
            raw_extensions=request.extensions,
            raw_preview_mode=request.preview_mode,
        )
        registry.update_progress(
            job_id,
            {
                "phase": "scanning",
                "files_processed": 0,
                "files_total": request.files_total_hint,
            },
        )

        processed_before_root = 0
        remaining_offset = request.offset
        remaining_limit = request.limit
        for root_index, root in enumerate(request.roots):
            if remaining_limit is not None and remaining_limit <= 0:
                attempts.extend(
                    _not_processed_root(
                        later_root,
                        reason="Not processed because the scan limit was reached.",
                    )
                    for later_root in request.roots[root_index:]
                )
                break

            attempted_offset = remaining_offset
            attempt = _scan_one_root(
                root,
                context=context,
                request=request,
                config=config,
                registry=registry,
                job_id=job_id,
                processed_before_root=processed_before_root,
                remaining_offset=remaining_offset,
                remaining_limit=remaining_limit,
                raise_if_cancelled=_raise_if_scan_cancelled,
            )
            attempts.append(attempt)
            if attempt.exception is not None:
                attempts.extend(
                    _not_processed_root(
                        later_root,
                        reason="Not processed because an earlier root failed.",
                    )
                    for later_root in request.roots[root_index + 1:]
                )
                break

            summary = attempt.summary
            aggregated["files_seen"] += summary.files_seen
            aggregated["files_added"] += summary.files_added
            aggregated["files_updated"] += summary.files_updated
            aggregated["files_unchanged"] += summary.files_unchanged
            aggregated["files_removed"] += summary.files_removed
            aggregated["files_failed"] += summary.files_failed
            processed_before_root += summary.files_seen
            remaining_offset = max(
                0,
                remaining_offset
                - _scan_offset_consumed(summary, requested_offset=attempted_offset),
            )
            if remaining_limit is not None:
                remaining_limit = max(0, remaining_limit - summary.files_seen)

        _finalize_scan_job(
            context,
            request,
            registry,
            job_id,
            aggregated,
            attempts,
        )
    except Exception as exc:
        _finalize_scan_job(
            context,
            request,
            registry,
            job_id,
            aggregated,
            attempts,
            unexpected_error=exc,
        )
    finally:
        context.operation_lock.release()
