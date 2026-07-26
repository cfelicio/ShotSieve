from __future__ import annotations

from http import HTTPStatus
from typing import Any
from urllib.parse import urlparse

from shotsieve.web_media import MediaDependencies, resolve_media_request, serve_media_response
from shotsieve.web_route_common import (
    DeleteResultPayload,
    ExportAggregate,
    ExportResultPayload,
    WebRouteContext,
    WebRouteDependencies,
    _begin_consistent_snapshot,
    _compare_request_models,
    _delete_result_payload,
    _export_result_payload,
    _finish_consistent_snapshot,
    _frozen_selection_batches,
    _is_ignorable_client_disconnect,
    _iter_selection_file_id_batches,
    _iter_sqlite_materialized_selection_batches,
    _materialize_selection_batches,
    _optional_payload_float,
    _optional_payload_int,
    _optional_payload_string_list,
    _parse_selection_payload,
    _require_registry,
    _require_root_for_destructive_selection,
    _scan_request_offset,
    _scan_request_roots,
    _scan_request_total_hint,
    _selection_excluded_ids,
    _validate_page_revision,
    _validate_selection_revision,
    log_request_message,
    send_json,
    send_json_error,
    serve_static,
)
from shotsieve.web_route_files import (
    _execute_cache_clear_request,
    _execute_delete_request,
    _execute_export_request,
    _handle_file_action_post_routes,
    _handle_filesystem_get_routes,
    _handle_media_get_routes,
    _operation_progress_callback,
    _progress_payload,
    _progress_total_hint,
)
from shotsieve.web_route_jobs import (
    _handle_analysis_post_routes,
    _handle_cache_post_routes,
    _handle_job_cancel_post_routes,
    _handle_job_get_routes,
    _raise_if_scan_cancelled,
    _scan_offset_consumed,
    _send_rows_total_estimate,
    comparison_summary_payload,
    handle_job_cancel,
    handle_job_result,
    handle_job_status,
    progress_payload,
    start_cache_clear_job,
    start_compare_job,
    start_delete_job,
    start_export_job,
    start_scan_job,
    start_score_job,
    try_acquire_operation_lock,
)
from shotsieve.web_route_review import (
    _handle_overview_get_routes,
    _handle_review_get_routes,
    _handle_review_post_routes,
)

_STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
    "/styles-layout.css": ("styles-layout.css", "text/css; charset=utf-8"),
    "/styles-workstation.css": ("styles-workstation.css", "text/css; charset=utf-8"),
    "/styles-polish.css": ("styles-polish.css", "text/css; charset=utf-8"),
    "/app-state.js": ("app-state.js", "application/javascript; charset=utf-8"),
    "/app.js": ("app.js", "application/javascript; charset=utf-8"),
    "/app-utils.js": ("app-utils.js", "application/javascript; charset=utf-8"),
    "/app-busy.js": ("app-busy.js", "application/javascript; charset=utf-8"),
    "/app-review.js": ("app-review.js", "application/javascript; charset=utf-8"),
    "/app-workflow-polling.js": ("app-workflow-polling.js", "application/javascript; charset=utf-8"),
    "/app-workflow-compare.js": ("app-workflow-compare.js", "application/javascript; charset=utf-8"),
    "/app-workflow-export.js": ("app-workflow-export.js", "application/javascript; charset=utf-8"),
    "/app-workflow-library.js": ("app-workflow-library.js", "application/javascript; charset=utf-8"),
    "/app-workflows.js": ("app-workflows.js", "application/javascript; charset=utf-8"),
    "/app-grid.js": ("app-grid.js", "application/javascript; charset=utf-8"),
    "/app-controller.js": ("app-controller.js", "application/javascript; charset=utf-8"),
    "/app-events.js": ("app-events.js", "application/javascript; charset=utf-8"),
}

_SELECTION_BATCH_SIZE = 500


def _handle_static_get_routes(handler: Any, context: WebRouteContext, parsed: Any) -> bool:
    static_entry = _STATIC_FILES.get(parsed.path)
    if static_entry is None:
        return False
    serve_static(handler, *static_entry, static_dir=context.static_dir)
    return True


def handle_get(handler: Any, context: WebRouteContext) -> None:
    parsed = urlparse(handler.path)
    route_families = (
        _handle_static_get_routes,
        _handle_overview_get_routes,
        _handle_job_get_routes,
        _handle_filesystem_get_routes,
        _handle_review_get_routes,
        _handle_media_get_routes,
    )
    for route_family in route_families:
        if route_family(handler, context, parsed):
            return

    handler.send_error(HTTPStatus.NOT_FOUND, "Route not found")


def handle_post(handler: Any, context: WebRouteContext) -> None:
    parsed = urlparse(handler.path)

    route_families = (
        _handle_review_post_routes,
        _handle_analysis_post_routes,
        _handle_job_cancel_post_routes,
        _handle_cache_post_routes,
        _handle_file_action_post_routes,
    )
    for route_family in route_families:
        if route_family(handler, context, parsed):
            return

    handler.send_error(HTTPStatus.NOT_FOUND, "Route not found")


__all__ = [
    "DeleteResultPayload",
    "ExportAggregate",
    "ExportResultPayload",
    "MediaDependencies",
    "WebRouteContext",
    "WebRouteDependencies",
    "_STATIC_FILES",
    "_SELECTION_BATCH_SIZE",
    "_begin_consistent_snapshot",
    "_compare_request_models",
    "_delete_result_payload",
    "_execute_cache_clear_request",
    "_execute_delete_request",
    "_execute_export_request",
    "_export_result_payload",
    "_finish_consistent_snapshot",
    "_frozen_selection_batches",
    "_handle_analysis_post_routes",
    "_handle_cache_post_routes",
    "_handle_file_action_post_routes",
    "_handle_filesystem_get_routes",
    "_handle_job_cancel_post_routes",
    "_handle_job_get_routes",
    "_handle_media_get_routes",
    "_handle_overview_get_routes",
    "_handle_review_get_routes",
    "_handle_review_post_routes",
    "_handle_static_get_routes",
    "_is_ignorable_client_disconnect",
    "_iter_selection_file_id_batches",
    "_iter_sqlite_materialized_selection_batches",
    "_materialize_selection_batches",
    "_operation_progress_callback",
    "_optional_payload_float",
    "_optional_payload_int",
    "_optional_payload_string_list",
    "_parse_selection_payload",
    "_progress_payload",
    "_progress_total_hint",
    "_raise_if_scan_cancelled",
    "_require_registry",
    "_require_root_for_destructive_selection",
    "_scan_offset_consumed",
    "_scan_request_offset",
    "_scan_request_roots",
    "_scan_request_total_hint",
    "_selection_excluded_ids",
    "_send_rows_total_estimate",
    "_validate_page_revision",
    "_validate_selection_revision",
    "comparison_summary_payload",
    "handle_get",
    "handle_job_cancel",
    "handle_job_result",
    "handle_job_status",
    "handle_post",
    "log_request_message",
    "progress_payload",
    "resolve_media_request",
    "send_json",
    "send_json_error",
    "serve_media_response",
    "serve_static",
    "start_cache_clear_job",
    "start_compare_job",
    "start_delete_job",
    "start_export_job",
    "start_scan_job",
    "start_score_job",
    "try_acquire_operation_lock",
]
