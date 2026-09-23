from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from http import HTTPStatus
from typing import Any, cast
from urllib.parse import parse_qs, urlparse

from shotsieve.config import normalize_raw_preview_mode
from shotsieve.job_registry import JobRegistry
from shotsieve.learned_iqa import DEFAULT_MODEL_NAME
from shotsieve.learned_iqa_catalog import validate_model_name
from shotsieve.model_assets import (
    build_model_diagnostic,
    classify_preparation_error,
    read_preparation_record,
)
from shotsieve.scoring import AnalysisProgress
from shotsieve.web_route_common import (
    WebRouteContext,
    WebRouteDependencies,
    _compare_request_models,
    _export_result_payload,
    get_default_route_adapters,
    _require_registry,
    route_callback,
    send_json,
    send_json_error,
)
from shotsieve.web_request import parse_max_decode_pixels
from shotsieve import web_route_scan as _scan_runner

# Keep the historical private helper imports available to web_routes and
# integrations while the implementation lives in the dedicated scan runner.
_ScanJobRequest = _scan_runner._ScanJobRequest
_finalize_scan_job = _scan_runner._finalize_scan_job
_raise_if_scan_cancelled = _scan_runner._raise_if_scan_cancelled
_resolved_scan_root = _scan_runner._resolved_scan_root
_run_scan_job = _scan_runner._run_scan_job
_scan_job_summary = _scan_runner._scan_job_summary
_scan_offset_consumed = _scan_runner._scan_offset_consumed
_scan_one_root = _scan_runner._scan_one_root
_scan_root_report = _scan_runner._scan_root_report
_scan_root_report_from_exception = _scan_runner._scan_root_report_from_exception


def _route(context: WebRouteContext, name: str, fallback: Callable[..., Any]) -> Callable[..., Any]:
    return route_callback(context, name, fallback)


def _handler_route(handler: Any, name: str, fallback: Callable[..., Any]) -> Callable[..., Any]:
    adapters = getattr(handler, "_shotsieve_route_adapters", None) or get_default_route_adapters()
    return adapters.get(name, fallback) if adapters is not None else fallback


def _progress_payload_for_operation(phase: str, *, files_processed: int = 0, files_total: int = 0) -> dict[str, object]:
    from shotsieve.web_route_files import _progress_payload

    return _progress_payload(phase, files_processed=files_processed, files_total=files_total)


def _progress_total_hint_for_operation(deps: WebRouteDependencies, payload: dict[str, object]) -> int | None:
    from shotsieve.web_route_files import _progress_total_hint

    return _progress_total_hint(deps, payload)


def _execute_delete_request(*args: Any, **kwargs: Any) -> Any:
    from shotsieve.web_route_files import _execute_delete_request as execute_delete_request

    return execute_delete_request(*args, **kwargs)


def _execute_export_request(*args: Any, **kwargs: Any) -> Any:
    from shotsieve.web_route_files import _execute_export_request as execute_export_request

    return execute_export_request(*args, **kwargs)


def _execute_cache_clear_request(*args: Any, **kwargs: Any) -> Any:
    from shotsieve.web_route_files import _execute_cache_clear_request as execute_cache_clear_request

    return execute_cache_clear_request(*args, **kwargs)


def _execute_missing_cache_apply_request(*args: Any, **kwargs: Any) -> Any:
    from shotsieve.web_route_files import _execute_missing_cache_apply_request as execute_missing_cache_apply_request

    return execute_missing_cache_apply_request(*args, **kwargs)


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


@dataclass(frozen=True, slots=True)
class _OperationJobFailure:
    error: str
    progress: dict[str, object] | None = None
    summary: dict[str, object] | None = None


def _default_operation_job_failure(error: BaseException) -> _OperationJobFailure:
    return _OperationJobFailure(
        error=str(error),
        summary=_operation_failure_summary(error),
    )


def _model_failure_summary(
    error: BaseException,
    *,
    model_name: object,
    requested_runtime: object,
    phase: str,
) -> dict[str, object]:
    existing = getattr(error, "model_diagnostic", None)
    diagnostic = dict(existing) if isinstance(existing, dict) else build_model_diagnostic(
        error,
        phase=phase,
        model_name=model_name,
        requested_runtime=requested_runtime,
    )
    diagnostic.setdefault("model", str(model_name) if model_name else None)
    diagnostic.setdefault("requested_runtime", str(requested_runtime or "auto").casefold())
    return {"diagnostic": diagnostic, "error_report": diagnostic}


def _handle_job_get_routes(handler: Any, context: WebRouteContext, parsed: Any) -> bool:
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
        _route(context, "handle_job_status", handle_job_status)(
            handler,
            _route(context, "_require_registry", _require_registry)(registry, label=label),
            label=label,
        )
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
    _route(context, "handle_job_result", handle_job_result)(
        handler,
        _route(context, "_require_registry", _require_registry)(registry, label=label),
        label=label,
    )
    return True


def _handle_analysis_post_routes(handler: Any, context: WebRouteContext, parsed: Any) -> bool:
    deps = cast(WebRouteDependencies, context.dependency_views.jobs)
    if parsed.path == "/api/scan/start":
        payload = deps.read_json_body(handler, max_body_size=context.max_request_body_size)
        _route(context, "start_scan_job", start_scan_job)(handler, context, payload)
        return True

    if parsed.path == "/api/score/start":
        payload = deps.read_json_body(handler, max_body_size=context.max_request_body_size)
        _route(context, "start_score_job", start_score_job)(handler, context, payload)
        return True

    if parsed.path == "/api/compare-models/start":
        payload = deps.read_json_body(handler, max_body_size=context.max_request_body_size)
        _route(context, "start_compare_job", start_compare_job)(handler, context, payload)
        return True

    if parsed.path == "/api/models/prepare/start":
        payload = deps.read_json_body(handler, max_body_size=context.max_request_body_size)
        _route(context, "start_model_prepare_job", start_model_prepare_job)(handler, context, payload)
        return True

    if parsed.path in {"/api/score-estimate", "/api/compare-estimate"}:
        _route(context, "_send_rows_total_estimate", _send_rows_total_estimate)(handler, context)
        return True

    return False


def _handle_job_cancel_post_routes(handler: Any, context: WebRouteContext, parsed: Any) -> bool:
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

    _route(context, "handle_job_cancel", handle_job_cancel)(
        handler,
        _route(context, "_require_registry", _require_registry)(registry, label="Job"),
        max_request_body_size=context.max_request_body_size,
    )
    return True


def _handle_cache_post_routes(handler: Any, context: WebRouteContext, parsed: Any) -> bool:
    deps = cast(WebRouteDependencies, context.dependency_views.jobs)
    emit_json = _route(context, "send_json", send_json)
    if parsed.path == "/api/cache/missing/apply":
        payload = deps.read_json_body(handler, max_body_size=context.max_request_body_size)
        result = _route(context, "_execute_missing_cache_apply_request", _execute_missing_cache_apply_request)(context, payload)
        emit_json(handler, result)
        return True

    if parsed.path == "/api/cache/clear/start":
        payload = deps.read_json_body(handler, max_body_size=context.max_request_body_size)
        _route(context, "start_cache_clear_job", start_cache_clear_job)(handler, context, payload)
        return True

    if parsed.path != "/api/cache/clear":
        return False

    payload = deps.read_json_body(handler, max_body_size=context.max_request_body_size)
    result = _route(context, "_execute_cache_clear_request", _execute_cache_clear_request)(
        context, payload, progress_callback=None, cancel_check=None
    )
    emit_json(handler, result)
    return True


def _start_operation_job(
    handler: Any,
    context: WebRouteContext,
    *,
    registry: JobRegistry,
    initial_progress: dict[str, object],
    progress_payload: Callable[..., dict[str, object]],
    worker: Callable[[Callable[..., None], Callable[[], None]], object],
    result_payload: Callable[[object], dict[str, object]],
    cancel_error: Callable[[], Exception],
    exception_payload: Callable[[BaseException], _OperationJobFailure] = _default_operation_job_failure,
    result_handler: Callable[[JobRegistry, str, dict[str, object]], None] | None = None,
) -> None:
    """Start an operation job with one shared lock/registry lifecycle."""
    deps = cast(WebRouteDependencies, context.dependency_views.jobs)
    server = getattr(handler, "server", None)
    lifecycle_lock = getattr(server, "operation_lifecycle_lock", None)

    def start_job() -> None:
        if server is not None and not getattr(server, "accepting_operations", True):
            _route(context, "send_json_error", send_json_error)(
                handler,
                HTTPStatus.SERVICE_UNAVAILABLE,
                "The server is shutting down and cannot start new work.",
            )
            return
        if not _route(context, "try_acquire_operation_lock", try_acquire_operation_lock)(handler, context):
            return

        try:
            job_id = registry.create(initial_progress=initial_progress)
        except Exception:
            context.operation_lock.release()
            raise

        def run_job() -> None:
            try:
                def publish(*args: object) -> None:
                    registry.update_progress(job_id, progress_payload(*args))

                def cancel_check() -> None:
                    if registry.is_cancelled(job_id):
                        raise cancel_error()

                result = worker(publish, cancel_check)
                summary = result_payload(result)
                if result_handler is None:
                    registry.complete(job_id, summary=summary)
                else:
                    result_handler(registry, job_id, summary)
            except Exception as exc:
                failure = exception_payload(exc)
                registry.fail(
                    job_id,
                    error=failure.error,
                    progress=failure.progress,
                    summary=failure.summary,
                )
            finally:
                context.operation_lock.release()

        try:
            deps.thread_factory(target=run_job, daemon=True).start()
        except Exception as exc:
            registry.fail(job_id, error=str(exc))
            context.operation_lock.release()
            raise
        _route(context, "send_json", send_json)(handler, {"job_id": job_id, "status": "running"})

    if lifecycle_lock is None:
        start_job()
    else:
        with lifecycle_lock:
            start_job()


def start_delete_job(handler: Any, context: WebRouteContext, payload: dict[str, object]) -> None:
    registry = _route(context, "_require_registry", _require_registry)(context.operation_registry, label="Operation")
    deps = cast(WebRouteDependencies, context.dependency_views.jobs)
    total_hint = _route(context, "_progress_total_hint", _progress_total_hint_for_operation)(deps, payload) or 0
    _start_operation_job(
        handler,
        context,
        registry=registry,
        initial_progress=_route(context, "_progress_payload", _progress_payload_for_operation)(
            "deleting_files",
            files_processed=0,
            files_total=total_hint,
        ),
        progress_payload=lambda processed, total, phase: _route(context, "_progress_payload", _progress_payload_for_operation)(
            phase,
            files_processed=processed,
            files_total=total,
        ),
        worker=lambda publish, cancel_check: _route(context, "_execute_delete_request", _execute_delete_request)(
            context,
            payload,
            progress_callback=publish,
            cancel_check=cancel_check,
        ),
        result_payload=lambda result: cast(dict[str, object], result),
        cancel_error=lambda: InterruptedError("Delete job was cancelled by user."),
    )


def start_export_job(handler: Any, context: WebRouteContext, payload: dict[str, object]) -> None:
    registry = _route(context, "_require_registry", _require_registry)(context.operation_registry, label="Operation")
    deps = cast(WebRouteDependencies, context.dependency_views.jobs)
    phase = "moving_files" if str(payload.get("mode") or "copy") == "move" else "exporting_files"
    total_hint = _route(context, "_progress_total_hint", _progress_total_hint_for_operation)(deps, payload) or 0
    _start_operation_job(
        handler,
        context,
        registry=registry,
        initial_progress=_route(context, "_progress_payload", _progress_payload_for_operation)(
            phase,
            files_processed=0,
            files_total=total_hint,
        ),
        progress_payload=lambda processed, total, phase_name: _route(context, "_progress_payload", _progress_payload_for_operation)(
            phase_name,
            files_processed=processed,
            files_total=total,
        ),
        worker=lambda publish, cancel_check: _route(context, "_execute_export_request", _execute_export_request)(
            context,
            payload,
            progress_callback=publish,
            cancel_check=cancel_check,
        ),
        result_payload=_route(context, "_export_result_payload", _export_result_payload),
        cancel_error=lambda: InterruptedError("Export job was cancelled by user."),
    )


def start_cache_clear_job(handler: Any, context: WebRouteContext, payload: dict[str, object]) -> None:
    registry = _route(context, "_require_registry", _require_registry)(context.operation_registry, label="Operation")
    deps = cast(WebRouteDependencies, context.dependency_views.jobs)
    deps.required_choice(payload.get("scope"), name="scope", choices=("scores", "review", "all"))
    _start_operation_job(
        handler,
        context,
        registry=registry,
        initial_progress=_route(context, "_progress_payload", _progress_payload_for_operation)(
            "clearing_cache",
            files_processed=0,
            files_total=1,
        ),
        progress_payload=lambda processed, total, phase_name: _route(context, "_progress_payload", _progress_payload_for_operation)(
            phase_name,
            files_processed=processed,
            files_total=total,
        ),
        worker=lambda publish, cancel_check: _route(context, "_execute_cache_clear_request", _execute_cache_clear_request)(
            context,
            payload,
            progress_callback=publish,
            cancel_check=cancel_check,
        ),
        result_payload=lambda result: cast(dict[str, object], result),
        cancel_error=lambda: InterruptedError("Cache clear job was cancelled by user."),
    )


def _send_rows_total_estimate(handler: Any, context: WebRouteContext) -> None:
    deps = cast(WebRouteDependencies, context.dependency_views.jobs)
    payload = deps.read_json_body(handler, max_body_size=context.max_request_body_size)
    with deps.database(context.db_path) as connection:
        rows_total = deps.count_score_rows(
            connection,
            raw_root=deps.optional_string(payload.get("root")),
        )
    _route(context, "send_json", send_json)(handler, {"rows_total": rows_total})


def start_scan_job(handler: Any, context: WebRouteContext, payload: dict[str, object]) -> None:
    deps = cast(WebRouteDependencies, context.dependency_views.jobs)
    scan_registry = _route(context, "_require_registry", _require_registry)(context.scan_registry, label="Scan")
    request = _ScanJobRequest.from_scan_request(deps.parse_scan_request(payload))

    from shotsieve.scanner import check_overlapping_roots
    overlaps = check_overlapping_roots(request.roots)
    if overlaps:
        parent, child = overlaps[0]
        handler.send_error(HTTPStatus.BAD_REQUEST, f"Overlapping folders detected: '{child}' is a subfolder of '{parent}'. Please remove the subfolder.")
        return

    if not _route(context, "try_acquire_operation_lock", try_acquire_operation_lock)(handler, context):
        return

    total_hint = request.files_total_hint
    job_id = scan_registry.create(initial_progress={
        "phase": "indexing",
        "files_processed": 0,
        "files_total": total_hint,
    })

    worker = partial(_run_scan_job, context, request, scan_registry, job_id)
    try:
        deps.thread_factory(target=worker, daemon=True).start()
    except Exception:
        context.operation_lock.release()
        raise
    _route(context, "send_json", send_json)(handler, {"job_id": job_id, "status": "running"})


def start_score_job(handler: Any, context: WebRouteContext, payload: dict[str, object]) -> None:
    deps = cast(WebRouteDependencies, context.dependency_views.jobs)
    score_registry = _route(context, "_require_registry", _require_registry)(context.score_registry, label="Score")
    learned_device = deps.optional_string(payload.get("device"))
    resource_profile = deps.optional_string(payload.get("resource_profile"))
    raw_preview_mode = normalize_raw_preview_mode(deps.optional_string(payload.get("preview_mode")))
    max_decode_pixels = parse_max_decode_pixels(payload.get("max_decode_megapixels"))
    requested_model = deps.optional_string(payload.get("learned_backend_name"))
    if requested_model:
        validate_model_name(requested_model)
    deps.require_learned_runtime(resource_profile=resource_profile, preferred_device=learned_device)

    if not _route(context, "try_acquire_operation_lock", try_acquire_operation_lock)(handler, context):
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
                score_registry.update_progress(job_id, _route(context, "progress_payload", progress_payload)(progress))

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
                    max_decode_pixels=max_decode_pixels,
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
            failure = _model_failure_summary(
                exc,
                model_name=requested_model or DEFAULT_MODEL_NAME,
                requested_runtime=learned_device,
                phase="score_job",
            )
            diagnostic = failure["diagnostic"]
            score_registry.fail(
                job_id,
                error=str(diagnostic.get("cause") or "Scoring failed."),
                summary=failure,
            )
        finally:
            context.operation_lock.release()

    try:
        deps.thread_factory(target=run_score_job, daemon=True).start()
    except Exception:
        context.operation_lock.release()
        raise
    _route(context, "send_json", send_json)(handler, {"job_id": job_id, "status": "running"})


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
    deps = cast(WebRouteDependencies, context.dependency_views.jobs)
    registry = _route(context, "_require_registry", _require_registry)(context.model_registry, label="Model preparation")
    raw_model = deps.optional_string(payload.get("model"))
    if not raw_model:
        raise ValueError("model is required")
    model_name = validate_model_name(raw_model)
    requested_device = deps.optional_string(payload.get("device"))
    prepare_fn = getattr(deps, "prepare_model", None)
    if not callable(prepare_fn):
        raise RuntimeError("Model preparation is unavailable")

    def failure_diagnostic(error: BaseException) -> dict[str, object]:
        diagnostic = read_preparation_record(context.db_path.parent)
        if diagnostic.get("state") in {"failed", "runtime_unavailable"}:
            return diagnostic
        attached = getattr(error, "model_diagnostic", None)
        if isinstance(attached, dict):
            return {
                **diagnostic,
                "state": "failed",
                "model": model_name,
                "error": attached.get("cause"),
                "error_report": attached,
                "recovery_action": attached.get("recovery_action"),
                "record_write_error": attached.get("record_write_error"),
                "cancelled": isinstance(error, InterruptedError),
            }
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

    def prepare_worker(publish: Callable[..., None], cancel_check: Callable[[], None]) -> object:
        return prepare_fn(
            model_name,
            data_dir=context.db_path.parent,
            device=requested_device,
            progress_callback=publish,
            cancel_check=cancel_check,
        )

    def model_exception_payload(error: BaseException) -> _OperationJobFailure:
        diagnostic = failure_diagnostic(error)
        error_report = diagnostic.get("error_report")
        report_cause = error_report.get("cause") if isinstance(error_report, dict) else None
        return _OperationJobFailure(
            error=str(
                diagnostic.get("error")
                or report_cause
                or (
                    "Model preparation was cancelled."
                    if isinstance(error, InterruptedError)
                    else "Model preparation failed."
                )
            ),
            progress=_model_prepare_progress(diagnostic),
            summary=diagnostic,
        )

    _start_operation_job(
        handler,
        context,
        registry=registry,
        initial_progress=_model_prepare_progress({
            "phase": "checking_storage",
            "model": model_name,
            "processed_counts": {"validation_images": 0},
        }),
        progress_payload=lambda record: _model_prepare_progress(cast(dict[str, object], record)),
        worker=prepare_worker,
        result_payload=lambda result: cast(dict[str, object], result),
        cancel_error=lambda: InterruptedError("Model preparation job was cancelled by user."),
        exception_payload=model_exception_payload,
    )


def start_compare_job(handler: Any, context: WebRouteContext, payload: dict[str, object]) -> None:
    deps = cast(WebRouteDependencies, context.dependency_views.jobs)
    compare_registry = _route(context, "_require_registry", _require_registry)(context.compare_registry, label="Compare")
    compare_request = deps.parse_compare_request(payload, default_batch_size=deps.default_batch_size())
    raw_preview_mode = normalize_raw_preview_mode(deps.optional_string(payload.get("preview_mode")))
    deps.require_learned_runtime(
        resource_profile=compare_request.get("resource_profile"),
        preferred_device=compare_request.get("device"),
    )

    if not _route(context, "try_acquire_operation_lock", try_acquire_operation_lock)(handler, context):
        return

    job_id = compare_registry.create(initial_progress={
        "model_name": None,
        "model_index": 0,
        "model_count": len(_route(context, "_compare_request_models", _compare_request_models)(compare_request)),
        "files_processed": 0,
        "files_total": 0,
    })

    def run_compare_job() -> None:
        try:
            def publish_progress(progress: AnalysisProgress) -> None:
                compare_registry.update_progress(job_id, _route(context, "progress_payload", progress_payload)(progress))

            with deps.database(context.db_path) as connection:
                preview_dir = deps.get_preview_cache_root(connection, db_path=context.db_path, persist=False)
                summary = deps.compare_learned_models(
                    connection,
                    model_names=_route(context, "_compare_request_models", _compare_request_models)(compare_request),
                    limit=compare_request["limit"],
                    offset=compare_request["offset"],
                    raw_root=compare_request["root"],
                    learned_device=compare_request["device"],
                    learned_batch_size=compare_request["batch_size"],
                    compare_chunk_size=compare_request["compare_chunk_size"],
                    progress_callback=publish_progress,
                    preview_dir=preview_dir,
                    raw_preview_mode=raw_preview_mode,
                    max_decode_pixels=compare_request["max_decode_pixels"],
                    resource_profile=compare_request.get("resource_profile"),
                )

            compare_registry.complete(job_id, summary=_route(context, "comparison_summary_payload", comparison_summary_payload)(summary))
        except Exception as exc:
            model_names = _route(context, "_compare_request_models", _compare_request_models)(compare_request)
            failure = _model_failure_summary(
                exc,
                model_name=",".join(model_names) if model_names else None,
                requested_runtime=compare_request.get("device"),
                phase="compare_job",
            )
            diagnostic = failure["diagnostic"]
            compare_registry.fail(
                job_id,
                error=str(diagnostic.get("cause") or "Model comparison failed."),
                summary=failure,
            )
        finally:
            context.operation_lock.release()

    try:
        deps.thread_factory(target=run_compare_job, daemon=True).start()
    except Exception:
        context.operation_lock.release()
        raise
    _route(context, "send_json", send_json)(handler, {"job_id": job_id, "status": "running"})


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
    if context.operation_lock.acquire(blocking=False):
        return True
    _route(context, "send_json_error", send_json_error)(
        handler,
        HTTPStatus.CONFLICT,
        "Another analysis operation is already running. Please wait for it to finish.",
    )
    return False


def handle_job_status(handler: Any, registry: JobRegistry, *, label: str) -> None:
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
    _handler_route(handler, "send_json", send_json)(handler, status_payload)


def handle_job_result(handler: Any, registry: JobRegistry, *, label: str) -> None:
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
            _handler_route(handler, "send_json", send_json)(handler, summary_payload)
            return
        _handler_route(handler, "send_json_error", send_json_error)(handler, HTTPStatus.INTERNAL_SERVER_ERROR, f"{label} job completed without a summary payload")
        return

    if status_value == "failed":
        if label in {"Operation", "Model preparation", "Score", "Compare"} and isinstance(status_payload.get("summary"), dict):
            _handler_route(handler, "send_json", send_json)(handler, status_payload["summary"])
            return
        _handler_route(handler, "send_json_error", send_json_error)(handler, HTTPStatus.BAD_REQUEST, str(status_payload.get("error") or f"{label} job failed"))
        return

    _handler_route(handler, "send_json_error", send_json_error)(handler, HTTPStatus.CONFLICT, f"{label} job is still running")


def handle_job_cancel(handler: Any, registry: JobRegistry, *, max_request_body_size: int) -> None:
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
    _handler_route(handler, "send_json", send_json)(handler, {"job_id": job_id, "cancelled": cancelled})
