from __future__ import annotations

import sys
from http import HTTPStatus
from pathlib import Path
from typing import Any, cast
from urllib.parse import parse_qs, urlparse

from shotsieve.config import normalize_raw_preview_mode
from shotsieve.learned_iqa_catalog import validate_model_name
from shotsieve.model_assets import classify_preparation_error, read_preparation_record
from shotsieve.db import (
    attach_scan_run_diagnostic,
    mark_scan_run_diagnostic_persisted,
    persist_scan_run_diagnostic,
    scan_run_diagnostic_from_exception,
    scan_run_diagnostic_was_persisted,
)
from shotsieve.job_registry import JobRegistry
from shotsieve.models import ScanRunDiagnostic
from shotsieve.scoring import AnalysisProgress
from shotsieve.web_route_common import (
    WebRouteContext,
    WebRouteDependencies,
)


def _get_web_routes() -> Any:
    return sys.modules["shotsieve.web_routes"]


def _operation_failure_summary(error: BaseException) -> dict[str, object] | None:
    raw_summary = getattr(error, "file_operation_summary", None)
    if not isinstance(raw_summary, dict):
        return None
    summary = dict(raw_summary)
    summary.setdefault("fatal_error", str(error))
    if isinstance(error, InterruptedError):
        summary["cancelled"] = True
        summary["outcome"] = "cancelled"
    return summary


def _handle_job_get_routes(handler: Any, context: WebRouteContext, parsed: Any) -> bool:
    routes = _get_web_routes()
    status_routes = {
        "/api/compare-models/status": (context.compare_registry, "Compare"),
        "/api/operations/status": (context.operation_registry, "Operation"),
        "/api/score/status": (context.score_registry, "Score"),
        "/api/scan/status": (context.scan_registry, "Scan"),
        "/api/models/prepare/status": (context.model_registry, "Model preparation"),
    }
    status_route = status_routes.get(parsed.path)
    if status_route is not None:
        registry, label = status_route
        routes.handle_job_status(handler, routes._require_registry(registry, label=label), label=label)
        return True

    result_routes = {
        "/api/compare-models/result": (context.compare_registry, "Compare"),
        "/api/operations/result": (context.operation_registry, "Operation"),
        "/api/score/result": (context.score_registry, "Score"),
        "/api/scan/result": (context.scan_registry, "Scan"),
        "/api/models/prepare/result": (context.model_registry, "Model preparation"),
    }
    result_route = result_routes.get(parsed.path)
    if result_route is None:
        return False

    registry, label = result_route
    routes.handle_job_result(handler, routes._require_registry(registry, label=label), label=label)
    return True


def _handle_analysis_post_routes(handler: Any, context: WebRouteContext, parsed: Any) -> bool:
    deps = cast(WebRouteDependencies, context.dependencies)
    routes = _get_web_routes()
    if parsed.path == "/api/scan/start":
        payload = deps.read_json_body(handler, max_body_size=context.max_request_body_size)
        routes.start_scan_job(handler, context, payload)
        return True

    if parsed.path == "/api/score/start":
        payload = deps.read_json_body(handler, max_body_size=context.max_request_body_size)
        routes.start_score_job(handler, context, payload)
        return True

    if parsed.path == "/api/compare-models/start":
        payload = deps.read_json_body(handler, max_body_size=context.max_request_body_size)
        routes.start_compare_job(handler, context, payload)
        return True

    if parsed.path == "/api/models/prepare/start":
        payload = deps.read_json_body(handler, max_body_size=context.max_request_body_size)
        routes.start_model_prepare_job(handler, context, payload)
        return True

    if parsed.path in {"/api/score-estimate", "/api/compare-estimate"}:
        routes._send_rows_total_estimate(handler, context)
        return True

    return False


def _handle_job_cancel_post_routes(handler: Any, context: WebRouteContext, parsed: Any) -> bool:
    routes = _get_web_routes()
    cancel_routes = {
        "/api/compare-models/cancel": context.compare_registry,
        "/api/operations/cancel": context.operation_registry,
        "/api/score/cancel": context.score_registry,
        "/api/scan/cancel": context.scan_registry,
        "/api/models/prepare/cancel": context.model_registry,
    }
    registry = cancel_routes.get(parsed.path)
    if registry is None:
        return False

    routes.handle_job_cancel(handler, routes._require_registry(registry, label="Job"), max_request_body_size=context.max_request_body_size)
    return True


def _handle_cache_post_routes(handler: Any, context: WebRouteContext, parsed: Any) -> bool:
    deps = cast(WebRouteDependencies, context.dependencies)
    routes = _get_web_routes()
    if parsed.path == "/api/cache/missing/apply":
        payload = deps.read_json_body(handler, max_body_size=context.max_request_body_size)
        result = routes._execute_missing_cache_apply_request(context, payload)
        routes.send_json(handler, result)
        return True

    if parsed.path == "/api/cache/clear/start":
        payload = deps.read_json_body(handler, max_body_size=context.max_request_body_size)
        routes.start_cache_clear_job(handler, context, payload)
        return True

    if parsed.path != "/api/cache/clear":
        return False

    payload = deps.read_json_body(handler, max_body_size=context.max_request_body_size)
    result = routes._execute_cache_clear_request(context, payload, progress_callback=None, cancel_check=None)
    routes.send_json(handler, result)
    return True


def start_delete_job(handler: Any, context: WebRouteContext, payload: dict[str, object]) -> None:
    routes = _get_web_routes()
    registry = routes._require_registry(context.operation_registry, label="Operation")
    deps = cast(WebRouteDependencies, context.dependencies)
    if not routes.try_acquire_operation_lock(handler, context):
        return

    total_hint = routes._progress_total_hint(deps, payload) or 0
    job_id = registry.create(initial_progress=routes._progress_payload("deleting_files", files_processed=0, files_total=total_hint))

    def run_job() -> None:
        try:
            def publish(processed: int, total: int, phase: str) -> None:
                registry.update_progress(job_id, routes._progress_payload(phase, files_processed=processed, files_total=total))

            def cancel_check() -> None:
                if registry.is_cancelled(job_id):
                    raise InterruptedError("Delete job was cancelled by user.")

            result = routes._execute_delete_request(context, payload, progress_callback=publish, cancel_check=cancel_check)
            registry.complete(job_id, summary=result)
        except Exception as exc:
            registry.fail(job_id, error=str(exc), summary=_operation_failure_summary(exc))
        finally:
            context.operation_lock.release()

    deps.thread_factory(target=run_job, daemon=True).start()
    routes.send_json(handler, {"job_id": job_id, "status": "running"})


def start_export_job(handler: Any, context: WebRouteContext, payload: dict[str, object]) -> None:
    routes = _get_web_routes()
    registry = routes._require_registry(context.operation_registry, label="Operation")
    deps = cast(WebRouteDependencies, context.dependencies)
    if not routes.try_acquire_operation_lock(handler, context):
        return

    phase = "moving_files" if str(payload.get("mode") or "copy") == "move" else "exporting_files"
    total_hint = routes._progress_total_hint(deps, payload) or 0
    job_id = registry.create(initial_progress=routes._progress_payload(phase, files_processed=0, files_total=total_hint))

    def run_job() -> None:
        try:
            def publish(processed: int, total: int, phase_name: str) -> None:
                registry.update_progress(job_id, routes._progress_payload(phase_name, files_processed=processed, files_total=total))

            def cancel_check() -> None:
                if registry.is_cancelled(job_id):
                    raise InterruptedError("Export job was cancelled by user.")

            result = routes._execute_export_request(context, payload, progress_callback=publish, cancel_check=cancel_check)
            registry.complete(job_id, summary=routes._export_result_payload(result))
        except Exception as exc:
            registry.fail(job_id, error=str(exc), summary=_operation_failure_summary(exc))
        finally:
            context.operation_lock.release()

    deps.thread_factory(target=run_job, daemon=True).start()
    routes.send_json(handler, {"job_id": job_id, "status": "running"})


def start_cache_clear_job(handler: Any, context: WebRouteContext, payload: dict[str, object]) -> None:
    routes = _get_web_routes()
    registry = routes._require_registry(context.operation_registry, label="Operation")
    deps = cast(WebRouteDependencies, context.dependencies)
    deps.required_choice(payload.get("scope"), name="scope", choices=("scores", "review", "all"))
    if not routes.try_acquire_operation_lock(handler, context):
        return

    job_id = registry.create(initial_progress=routes._progress_payload("clearing_cache", files_processed=0, files_total=1))

    def run_job() -> None:
        try:
            def publish(processed: int, total: int, phase_name: str) -> None:
                registry.update_progress(job_id, routes._progress_payload(phase_name, files_processed=processed, files_total=total))

            def cancel_check() -> None:
                if registry.is_cancelled(job_id):
                    raise InterruptedError("Cache clear job was cancelled by user.")

            result = routes._execute_cache_clear_request(context, payload, progress_callback=publish, cancel_check=cancel_check)
            registry.complete(job_id, summary=result)
        except Exception as exc:
            registry.fail(job_id, error=str(exc))
        finally:
            context.operation_lock.release()

    deps.thread_factory(target=run_job, daemon=True).start()
    routes.send_json(handler, {"job_id": job_id, "status": "running"})


def _send_rows_total_estimate(handler: Any, context: WebRouteContext) -> None:
    deps = cast(WebRouteDependencies, context.dependencies)
    routes = _get_web_routes()
    payload = deps.read_json_body(handler, max_body_size=context.max_request_body_size)
    with deps.database(context.db_path) as connection:
        rows_total = deps.count_score_rows(
            connection,
            raw_root=deps.optional_string(payload.get("root")),
        )
    routes.send_json(handler, {"rows_total": rows_total})


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


def start_scan_job(handler: Any, context: WebRouteContext, payload: dict[str, object]) -> None:
    deps = cast(WebRouteDependencies, context.dependencies)
    routes = _get_web_routes()
    scan_registry = routes._require_registry(context.scan_registry, label="Scan")
    scan_request = deps.parse_scan_request(payload)

    from shotsieve.scanner import check_overlapping_roots
    overlaps = check_overlapping_roots(scan_request["roots"])
    if overlaps:
        parent, child = overlaps[0]
        handler.send_error(HTTPStatus.BAD_REQUEST, f"Overlapping folders detected: '{child}' is a subfolder of '{parent}'. Please remove the subfolder.")
        return

    if not routes.try_acquire_operation_lock(handler, context):
        return

    total_hint = max(0, routes._scan_request_total_hint(scan_request))
    job_id = scan_registry.create(initial_progress={
        "phase": "indexing",
        "files_processed": 0,
        "files_total": total_hint,
    })

    def run_scan_job() -> None:
        aggregated = {
            "files_seen": 0,
            "files_added": 0,
            "files_updated": 0,
            "files_unchanged": 0,
            "files_removed": 0,
            "files_failed": 0,
        }
        root_results: list[dict[str, object]] = []
        try:
            config = deps.build_config(
                str(context.db_path),
                raw_preview_dir=scan_request["preview_dir"],
                raw_extensions=scan_request["extensions"],
                raw_preview_mode=scan_request["preview_mode"],
            )
            scan_registry.update_progress(job_id, {
                "phase": "scanning",
                "files_processed": 0,
                "files_total": total_hint,
            })

            processed_before_root = 0
            remaining_offset = max(0, routes._scan_request_offset(scan_request))
            remaining_limit = scan_request["limit"]

            def publish_progress(processed_in_root: int, _root_total: int, phase: str) -> None:
                files_total = total_hint if total_hint > 0 else 0
                scan_registry.update_progress(job_id, {
                    "phase": phase,
                    "files_processed": max(0, processed_before_root + processed_in_root),
                    "files_total": files_total,
                })

            roots = routes._scan_request_roots(scan_request)
            failure_error: str | None = None
            failure_progress: dict[str, object] | None = None
            for root_index, root in enumerate(roots):
                root_started_time = deps.utc_now()
                try:
                    routes._raise_if_scan_cancelled(scan_registry, job_id)
                    if remaining_limit is not None and remaining_limit <= 0:
                        root_results.extend(
                            _scan_root_report(
                                later_root,
                                status="not_processed",
                                error_text="Not processed because the scan limit was reached.",
                            )
                            for later_root in roots[root_index:]
                        )
                        break

                    root_total_hint = None
                    if total_hint > 0:
                        root_total_hint = max(0, total_hint - processed_before_root)

                    root_offset = remaining_offset
                    # Each root owns an independent transaction.  A failed
                    # root therefore rolls back only its own catalog work.
                    root_exception: BaseException | None = None
                    with deps.database(config.db_path) as connection:
                        try:
                            summary = deps.scan_root(
                                connection,
                                root=root,
                                recursive=scan_request["recursive"],
                                limit=remaining_limit,
                                offset=root_offset,
                                extensions=config.supported_extensions,
                                preview_dir=config.preview_dir,
                                rescan_all=scan_request["rescan_all"],
                                generate_previews=scan_request["generate_previews"],
                                raw_preview_mode=config.raw_preview_mode,
                                resource_profile=scan_request["resource_profile"],
                                progress_callback=publish_progress,
                                files_total_hint=root_total_hint,
                                cancel_check=lambda: routes._raise_if_scan_cancelled(scan_registry, job_id),
                                ignore_rules=scan_request["ignore_rules"],
                            )
                        except InterruptedError as exc:
                            # Preserve the established best-effort cancellation
                            # behavior: batches already committed by scan_root
                            # remain visible when this root's transaction exits.
                            root_exception = exc
                    if root_exception is not None:
                        raise root_exception
                    aggregated["files_seen"] += summary.files_seen
                    aggregated["files_added"] += summary.files_added
                    aggregated["files_updated"] += summary.files_updated
                    aggregated["files_unchanged"] += summary.files_unchanged
                    aggregated["files_removed"] += summary.files_removed
                    aggregated["files_failed"] += summary.files_failed
                    processed_before_root += summary.files_seen
                    remaining_offset = max(0, remaining_offset - routes._scan_offset_consumed(summary, requested_offset=root_offset))
                    if remaining_limit is not None:
                        remaining_limit = max(0, remaining_limit - summary.files_seen)
                    root_results.append(
                        _scan_root_report(
                            root,
                            status="completed_with_errors" if summary.files_failed else "completed",
                            summary=summary,
                            error_text=getattr(summary, "last_batch_error", None),
                        )
                    )
                except Exception as exc:
                    root_report = _scan_root_report_from_exception(
                        root,
                        exc,
                        started_time=root_started_time,
                        db_path=context.db_path,
                    )
                    root_results.append(root_report)
                    root_results.extend(
                        _scan_root_report(
                            later_root,
                            status="not_processed",
                            error_text="Not processed because an earlier root failed.",
                        )
                        for later_root in roots[root_index + 1:]
                    )
                    failure_error = str(root_report.get("error_text") or exc) or exc.__class__.__name__
                    files_processed = processed_before_root + int(root_report.get("files_seen", 0) or 0)
                    failure_progress = {
                        "phase": "failed",
                        "files_processed": max(0, files_processed),
                        "files_total": total_hint if total_hint > 0 else max(0, files_processed),
                    }
                    break

            if failure_error is not None:
                scan_registry.fail(
                    job_id,
                    error=failure_error,
                    progress=failure_progress,
                    summary=_scan_job_summary(aggregated, root_results, overall_status="failed"),
                )
                return

            scan_registry.update_progress(job_id, {
                "phase": "scanning",
                "files_processed": aggregated["files_seen"],
                "files_total": total_hint if total_hint > 0 else aggregated["files_seen"],
            })
            overall_status = (
                "completed_with_errors"
                if any(item["status"] == "completed_with_errors" for item in root_results)
                else "completed"
            )
            scan_registry.complete(
                job_id,
                summary=_scan_job_summary(aggregated, root_results, overall_status=overall_status),
            )
        except Exception as exc:
            scan_registry.fail(
                job_id,
                error=str(exc) or exc.__class__.__name__,
                summary=_scan_job_summary(aggregated, root_results, overall_status="failed"),
            )
        finally:
            context.operation_lock.release()

    deps.thread_factory(target=run_scan_job, daemon=True).start()
    routes.send_json(handler, {"job_id": job_id, "status": "running"})


def start_score_job(handler: Any, context: WebRouteContext, payload: dict[str, object]) -> None:
    deps = cast(WebRouteDependencies, context.dependencies)
    routes = _get_web_routes()
    score_registry = routes._require_registry(context.score_registry, label="Score")
    learned_device = deps.optional_string(payload.get("device"))
    resource_profile = deps.optional_string(payload.get("resource_profile"))
    raw_preview_mode = normalize_raw_preview_mode(deps.optional_string(payload.get("preview_mode")))
    requested_model = deps.optional_string(payload.get("learned_backend_name"))
    if requested_model:
        validate_model_name(requested_model)
    deps.require_learned_runtime(resource_profile=resource_profile, preferred_device=learned_device)

    if not routes.try_acquire_operation_lock(handler, context):
        return

    job_id = score_registry.create(initial_progress={
        "model_name": None,
        "model_index": 1,
        "model_count": 1,
        "files_processed": 0,
        "files_total": 0,
    })

    def run_score_job() -> None:
        try:
            def publish_progress(progress: AnalysisProgress) -> None:
                score_registry.update_progress(job_id, routes.progress_payload(progress))

            with deps.database(context.db_path) as connection:
                preview_dir = deps.get_preview_cache_root(connection, db_path=context.db_path, persist=False)
                summary = deps.score_files(
                    connection,
                    limit=deps.optional_int(payload.get("limit"), minimum=1),
                    offset=deps.optional_int(payload.get("offset"), minimum=0) or 0,
                    raw_root=deps.optional_string(payload.get("root")),
                    force=deps.coerce_bool(payload.get("force"), default=False),
                    learned_backend_name=deps.optional_string(payload.get("learned_backend_name")),
                    learned_device=learned_device,
                    learned_batch_size=deps.optional_int(payload.get("batch_size"), minimum=1) or deps.default_batch_size(),
                    preview_dir=preview_dir,
                    raw_preview_mode=raw_preview_mode,
                    progress_callback=publish_progress,
                    resource_profile=resource_profile,
                )

            score_registry.complete(job_id, summary={
                "rows_loaded": summary.rows_loaded,
                "files_considered": summary.files_considered,
                "files_scored": summary.files_scored,
                "learned_scored": summary.learned_scored,
                "files_skipped": summary.files_skipped,
                "files_failed": summary.files_failed,
            })
        except Exception as exc:
            score_registry.fail(job_id, error=str(exc))
        finally:
            context.operation_lock.release()

    deps.thread_factory(target=run_score_job, daemon=True).start()
    routes.send_json(handler, {"job_id": job_id, "status": "running"})


def _model_prepare_progress(record: dict[str, object]) -> dict[str, object]:
    phase = str(record.get("phase") or "preparing_model")
    phase_percent = {
        "checking_storage": 10,
        "preparing_model": 55,
        "validating_initialization": 85,
        "complete": 100,
    }.get(phase)
    return {
        "phase": phase,
        "model_name": record.get("model"),
        "percent": phase_percent,
        "processed_counts": record.get("processed_counts") or {},
    }


def start_model_prepare_job(handler: Any, context: WebRouteContext, payload: dict[str, object]) -> None:
    deps = cast(WebRouteDependencies, context.dependencies)
    routes = _get_web_routes()
    registry = routes._require_registry(context.model_registry, label="Model preparation")
    raw_model = deps.optional_string(payload.get("model"))
    if not raw_model:
        raise ValueError("model is required")
    model_name = validate_model_name(raw_model)
    prepare_fn = getattr(deps, "prepare_model", None)
    if not callable(prepare_fn):
        raise RuntimeError("Model preparation is unavailable")
    if not routes.try_acquire_operation_lock(handler, context):
        return

    job_id = registry.create(initial_progress=_model_prepare_progress({
        "phase": "checking_storage",
        "model": model_name,
        "processed_counts": {"validation_images": 0},
    }))

    def run_job() -> None:
        def failure_diagnostic(error: BaseException) -> dict[str, object]:
            diagnostic = read_preparation_record(context.db_path.parent)
            if diagnostic.get("state") in {"failed", "runtime_unavailable"}:
                return diagnostic
            if isinstance(error, InterruptedError):
                report = {
                    "category": "cancelled",
                    "phase": str(diagnostic.get("phase") or "preparing_model"),
                    "cause": "Model preparation was cancelled.",
                    "cause_chain": ["Model preparation was cancelled."],
                    "recovery_action": "Preparation was cancelled. Retry to complete model validation.",
                }
            else:
                report = classify_preparation_error(error, phase=str(diagnostic.get("phase") or "preparing_model"))
            return {
                **diagnostic,
                "state": "failed",
                "model": model_name,
                "error": report["cause"],
                "error_report": report,
                "recovery_action": report["recovery_action"],
                "cancelled": isinstance(error, InterruptedError),
            }

        try:
            def publish(record: dict[str, object]) -> None:
                registry.update_progress(job_id, _model_prepare_progress(record))

            def cancel_check() -> None:
                if registry.is_cancelled(job_id):
                    raise InterruptedError("Model preparation job was cancelled by user.")

            result = prepare_fn(
                model_name,
                data_dir=context.db_path.parent,
                progress_callback=publish,
                cancel_check=cancel_check,
            )
            registry.complete(job_id, summary=result)
        except InterruptedError as exc:
            diagnostic = failure_diagnostic(exc)
            registry.fail(
                job_id,
                error=str(diagnostic.get("error") or "Model preparation was cancelled."),
                progress=_model_prepare_progress(diagnostic),
                summary=diagnostic,
            )
        except Exception as exc:
            diagnostic = failure_diagnostic(exc)
            error_report = diagnostic.get("error_report")
            report_cause = error_report.get("cause") if isinstance(error_report, dict) else None
            registry.fail(
                job_id,
                error=str(diagnostic.get("error") or report_cause or "Model preparation failed."),
                progress=_model_prepare_progress(diagnostic),
                summary=diagnostic,
            )
        finally:
            context.operation_lock.release()

    deps.thread_factory(target=run_job, daemon=True).start()
    routes.send_json(handler, {"job_id": job_id, "status": "running"})


def start_compare_job(handler: Any, context: WebRouteContext, payload: dict[str, object]) -> None:
    deps = cast(WebRouteDependencies, context.dependencies)
    routes = _get_web_routes()
    compare_registry = routes._require_registry(context.compare_registry, label="Compare")
    compare_request = deps.parse_compare_request(payload, default_batch_size=deps.default_batch_size())
    raw_preview_mode = normalize_raw_preview_mode(deps.optional_string(payload.get("preview_mode")))
    deps.require_learned_runtime(
        resource_profile=compare_request.get("resource_profile"),
        preferred_device=compare_request.get("device"),
    )

    if not routes.try_acquire_operation_lock(handler, context):
        return

    job_id = compare_registry.create(initial_progress={
        "model_name": None,
        "model_index": 0,
        "model_count": len(routes._compare_request_models(compare_request)),
        "files_processed": 0,
        "files_total": 0,
    })

    def run_compare_job() -> None:
        try:
            def publish_progress(progress: AnalysisProgress) -> None:
                compare_registry.update_progress(job_id, routes.progress_payload(progress))

            with deps.database(context.db_path) as connection:
                preview_dir = deps.get_preview_cache_root(connection, db_path=context.db_path, persist=False)
                summary = deps.compare_learned_models(
                    connection,
                    model_names=routes._compare_request_models(compare_request),
                    limit=compare_request["limit"],
                    offset=compare_request["offset"],
                    raw_root=compare_request["root"],
                    learned_device=compare_request["device"],
                    learned_batch_size=compare_request["batch_size"],
                    compare_chunk_size=compare_request["compare_chunk_size"],
                    progress_callback=publish_progress,
                    preview_dir=preview_dir,
                    raw_preview_mode=raw_preview_mode,
                    resource_profile=compare_request.get("resource_profile"),
                )

            compare_registry.complete(job_id, summary=routes.comparison_summary_payload(summary))
        except Exception as exc:
            compare_registry.fail(job_id, error=str(exc))
        finally:
            context.operation_lock.release()

    deps.thread_factory(target=run_compare_job, daemon=True).start()
    routes.send_json(handler, {"job_id": job_id, "status": "running"})


def comparison_summary_payload(summary: Any) -> dict[str, object]:
    return {
        "model_names": summary.model_names,
        "rows": summary.rows,
        "compare_failures": getattr(summary, "compare_failures", []),
        "requested_rows_total": getattr(summary, "requested_rows_total", getattr(summary, "files_considered", 0)),
        "processed_rows_total": getattr(summary, "processed_rows_total", getattr(summary, "files_considered", 0)),
        "truncated": bool(getattr(summary, "truncated", False)),
        "max_rows": getattr(summary, "max_rows", None),
        "files_considered": summary.files_considered,
        "files_compared": summary.files_compared,
        "files_skipped": summary.files_skipped,
        "files_failed": summary.files_failed,
        "elapsed_seconds": summary.elapsed_seconds,
        "model_timings_seconds": summary.model_timings_seconds,
    }


def progress_payload(progress: AnalysisProgress) -> dict[str, object]:
    return {
        "model_name": progress.model_name,
        "model_index": progress.model_index,
        "model_count": progress.model_count,
        "files_processed": progress.files_processed,
        "files_total": progress.files_total,
        "phase": progress.phase,
    }


def try_acquire_operation_lock(handler: Any, context: WebRouteContext) -> bool:
    routes = _get_web_routes()
    if context.operation_lock.acquire(blocking=False):
        return True
    routes.send_json_error(
        handler,
        HTTPStatus.CONFLICT,
        "Another analysis operation is already running. Please wait for it to finish.",
    )
    return False


def handle_job_status(handler: Any, registry: JobRegistry, *, label: str) -> None:
    routes = _get_web_routes()
    deps = getattr(handler, "_shotsieve_route_dependencies", None)
    parsed = urlparse(handler.path)
    params = parse_qs(parsed.query)
    job_id = deps.first_value(params, "job_id", None) if deps is not None and hasattr(deps, "first_value") else (params.get("job_id") or [None])[0]
    if not job_id:
        raise ValueError("job_id is required")
    status_payload = registry.status(job_id)
    if status_payload is None:
        handler.send_error(HTTPStatus.NOT_FOUND, f"{label} job not found")
        return
    routes.send_json(handler, status_payload)


def handle_job_result(handler: Any, registry: JobRegistry, *, label: str) -> None:
    routes = _get_web_routes()
    deps = getattr(handler, "_shotsieve_route_dependencies", None)
    parsed = urlparse(handler.path)
    params = parse_qs(parsed.query)
    job_id = deps.first_value(params, "job_id", None) if deps is not None and hasattr(deps, "first_value") else (params.get("job_id") or [None])[0]
    if not job_id:
        raise ValueError("job_id is required")
    status_payload = registry.status(job_id)
    if status_payload is None:
        handler.send_error(HTTPStatus.NOT_FOUND, f"{label} job not found")
        return

    status_value = status_payload.get("status")
    if status_value == "completed":
        summary_payload = status_payload.get("summary")
        if isinstance(summary_payload, dict):
            routes.send_json(handler, summary_payload)
            return
        routes.send_json_error(handler, HTTPStatus.INTERNAL_SERVER_ERROR, f"{label} job completed without a summary payload")
        return

    if status_value == "failed":
        if label in {"Operation", "Model preparation"} and isinstance(status_payload.get("summary"), dict):
            routes.send_json(handler, status_payload["summary"])
            return
        routes.send_json_error(handler, HTTPStatus.BAD_REQUEST, str(status_payload.get("error") or f"{label} job failed"))
        return

    routes.send_json_error(handler, HTTPStatus.CONFLICT, f"{label} job is still running")


def handle_job_cancel(handler: Any, registry: JobRegistry, *, max_request_body_size: int) -> None:
    routes = _get_web_routes()
    deps = getattr(handler, "_shotsieve_route_dependencies", None)
    parsed = urlparse(handler.path)
    params = parse_qs(parsed.query)
    job_id = deps.first_value(params, "job_id", None) if deps is not None and hasattr(deps, "first_value") else (params.get("job_id") or [""])[0]
    if not job_id:
        content_length = int(getattr(handler, "headers", {}).get("Content-Length", "0") if hasattr(handler, "headers") else 0)
        payload = deps.read_json_body(handler, max_body_size=max_request_body_size) if (deps is not None and hasattr(deps, "read_json_body") and content_length > 0) else {}
        job_id = (deps.optional_string(payload.get("job_id")) if deps is not None and hasattr(deps, "optional_string") else payload.get("job_id")) or ""
    if not job_id:
        raise ValueError("job_id is required")
    cancelled = registry.cancel(job_id)
    routes.send_json(handler, {"job_id": job_id, "cancelled": cancelled})
