from __future__ import annotations

from http import HTTPStatus
from typing import Any, cast
from urllib.parse import parse_qs

from shotsieve.web_route_common import (
    WebRouteContext,
    WebRouteDependencies,
    _begin_consistent_snapshot,
    _finish_consistent_snapshot,
    _frozen_selection_batches,
    _parse_selection_payload,
    _validate_selection_revision,
    route_callback,
    send_bytes,
    send_json,
)


def _route(context: WebRouteContext, name: str, fallback: Any) -> Any:
    return route_callback(context, name, fallback)


def _review_filter_kwargs(deps: WebRouteDependencies, params: dict[str, list[str]]) -> dict[str, object]:
    """Parse one set of Review filters for page, count, and revision."""
    formats_raw = deps.first_value(params, "formats", None)
    formats = [format_name.strip() for format_name in formats_raw.split(",") if format_name.strip()] if formats_raw else None
    return {
        "root": deps.first_value(params, "root", None),
        "marked": deps.first_value(params, "marked", "all"),
        "issues": deps.first_value(params, "issues", "all"),
        "query": deps.first_value(params, "query", None),
        "min_score": deps.float_or_none(deps.first_value(params, "min_score", None)),
        "max_score": deps.float_or_none(deps.first_value(params, "max_score", None)),
        "formats": formats,
        "min_mp": deps.float_or_none(deps.first_value(params, "min_mp", None)),
        "max_mp": deps.float_or_none(deps.first_value(params, "max_mp", None)),
        "min_width": deps.optional_int(deps.first_value(params, "min_width", None)),
        "max_width": deps.optional_int(deps.first_value(params, "max_width", None)),
        "min_height": deps.optional_int(deps.first_value(params, "min_height", None)),
        "max_height": deps.optional_int(deps.first_value(params, "max_height", None)),
        "min_edge": deps.optional_int(deps.first_value(params, "min_edge", None)),
        "max_edge": deps.optional_int(deps.first_value(params, "max_edge", None)),
        "min_size": deps.optional_int(deps.first_value(params, "min_size", None)),
        "max_size": deps.optional_int(deps.first_value(params, "max_size", None)),
        "metadata": deps.first_value(params, "metadata", "all"),
    }


def _handle_overview_get_routes(handler: Any, context: WebRouteContext, parsed: Any) -> bool:
    deps = cast(WebRouteDependencies, context.dependency_views.review)
    emit_json = _route(context, "send_json", send_json)
    if parsed.path == "/api/overview":
        params = parse_qs(parsed.query)
        root = deps.first_value(params, "root", None)
        with deps.database(context.db_path) as connection:
            if root:
                emit_json(handler, deps.review_overview(connection, root=root))
            else:
                emit_json(handler, deps.review_overview(connection))
        return True

    if parsed.path == "/api/options":
        params = parse_qs(parsed.query)
        profile = deps.first_value(params, "resource_profile", None) or None
        emit_json(handler, deps.build_options_payload(context.db_path, resource_profile=profile))
        return True

    if parsed.path == "/api/analysis-diagnostics":
        params = parse_qs(parsed.query)
        with deps.database(context.db_path) as connection:
            emit_json(handler, deps.list_analysis_diagnostics(
                connection,
                root=deps.first_value(params, "root", None),
                limit=deps.int_or_default(deps.first_value(params, "limit", "100"), default=100, minimum=1, maximum=500),
            ))
        return True

    return False


def _handle_review_get_routes(handler: Any, context: WebRouteContext, parsed: Any) -> bool:
    deps = cast(WebRouteDependencies, context.dependency_views.review)
    emit_json = _route(context, "send_json", send_json)
    emit_bytes = _route(context, "send_bytes", send_bytes)
    if parsed.path == "/api/review/decisions.csv":
        params = parse_qs(parsed.query)
        root = deps.first_value(params, "root", None)
        decision = deps.required_choice(
            deps.first_value(params, "decision", "both"),
            name="decision",
            choices=("approved", "rejected", "both"),
        )
        if not root or not root.strip():
            raise ValueError("root is required")
        with deps.database(context.db_path) as connection:
            snapshot_active = _begin_consistent_snapshot(connection)
            try:
                body = deps.decision_csv(connection, root=root, decision=decision)
            except Exception:
                _finish_consistent_snapshot(connection, active=snapshot_active, success=False)
                raise
            else:
                _finish_consistent_snapshot(connection, active=snapshot_active, success=True)
        emit_bytes(
            handler,
            body.encode("utf-8"),
            content_type="text/csv; charset=utf-8",
            download_name="shotsieve-decisions.csv",
        )
        return True

    if parsed.path == "/api/files":
        params = parse_qs(parsed.query)
        filter_kwargs = _review_filter_kwargs(deps, params)

        with deps.database(context.db_path) as connection:
            snapshot_active = _begin_consistent_snapshot(connection)
            try:
                payload = deps.list_review_files(
                    connection,
                    sort=deps.first_value(params, "sort", "score_asc"),
                    **filter_kwargs,
                    limit=deps.int_or_default(deps.first_value(params, "limit", "60"), default=60, minimum=1, maximum=500),
                    offset=deps.int_or_default(deps.first_value(params, "offset", "0"), default=0, minimum=0),
                )
                total = deps.count_review_files(
                    connection,
                    **filter_kwargs,
                )
                selection_revision = deps.review_selection_revision(
                    connection,
                    scope="review-browser",
                    **filter_kwargs,
                )
            except Exception:
                _finish_consistent_snapshot(connection, active=snapshot_active, success=False)
                raise
            else:
                _finish_consistent_snapshot(connection, active=snapshot_active, success=True)
        emit_json(handler, {"items": payload, "total": total, "selection_revision": selection_revision})
        return True

    if parsed.path == "/api/files/count":
        params = parse_qs(parsed.query)
        filter_kwargs = _review_filter_kwargs(deps, params)

        with deps.database(context.db_path) as connection:
            snapshot_active = _begin_consistent_snapshot(connection)
            try:
                total = deps.count_review_files(
                    connection,
                    **filter_kwargs,
                )
                selection_revision = deps.review_selection_revision(
                    connection,
                    scope="review-browser",
                    **filter_kwargs,
                )
            except Exception:
                _finish_consistent_snapshot(connection, active=snapshot_active, success=False)
                raise
            else:
                _finish_consistent_snapshot(connection, active=snapshot_active, success=True)
        emit_json(handler, {"total": total, "selection_revision": selection_revision})
        return True

    if parsed.path == "/api/review/file-ids":
        params = parse_qs(parsed.query)
        marked = deps.required_choice(
            deps.first_value(params, "marked", None),
            name="marked",
            choices=("delete", "export", "none"),
        )
        state_filter_kwargs = {
            "marked": marked,
            "root": deps.first_value(params, "root", None),
            "query": deps.first_value(params, "query", None),
        }
        with deps.database(context.db_path) as connection:
            snapshot_active = _begin_consistent_snapshot(connection)
            try:
                ids = deps.list_review_state_file_ids(
                    connection,
                    **state_filter_kwargs,
                    limit=deps.int_or_default(deps.first_value(params, "limit", "500"), default=500, minimum=1, maximum=1000),
                    offset=deps.int_or_default(deps.first_value(params, "offset", "0"), default=0, minimum=0),
                )
                selection_revision = deps.review_selection_revision(
                    connection,
                    scope="review-state",
                    **state_filter_kwargs,
                )
            except Exception:
                _finish_consistent_snapshot(connection, active=snapshot_active, success=False)
                raise
            else:
                _finish_consistent_snapshot(connection, active=snapshot_active, success=True)
        emit_json(handler, {"ids": ids, "selection_revision": selection_revision})
        return True

    if parsed.path == "/api/file":
        params = parse_qs(parsed.query)
        file_id = deps.required_int(deps.first_value(params, "id", None), name="id", minimum=1)
        with deps.database(context.db_path) as connection:
            detail = deps.get_review_file_detail(connection, file_id)
        if detail is None:
            handler.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return True
        emit_json(handler, detail)
        return True

    return False


def _handle_review_post_routes(handler: Any, context: WebRouteContext, parsed: Any) -> bool:
    deps = cast(WebRouteDependencies, context.dependency_views.review)
    emit_json = _route(context, "send_json", send_json)
    if parsed.path == "/api/review":
        payload = deps.read_json_body(handler, max_body_size=context.max_request_body_size)
        file_id = deps.required_int(payload.get("file_id"), name="file_id", minimum=1)
        with deps.database(context.db_path) as connection:
            deps.update_review_state(
                connection,
                file_id=file_id,
                decision_state=deps.optional_string(payload.get("decision_state")),
                delete_marked=deps.optional_bool(payload.get("delete_marked"), name="delete_marked"),
                export_marked=deps.optional_bool(payload.get("export_marked"), name="export_marked"),
                updated_time=deps.utc_now(),
            )
            detail = deps.get_review_file_detail(connection, file_id)
        emit_json(handler, detail or {"ok": True})
        return True

    if parsed.path == "/api/review/batch":
        payload = deps.read_json_body(handler, max_body_size=context.max_request_body_size)
        selection = _route(context, "_parse_selection_payload", _parse_selection_payload)(deps, payload)
        with deps.database(context.db_path) as connection:
            batch_kwargs = {
                "decision_state": deps.optional_string(payload.get("decision_state")),
                "delete_marked": deps.optional_bool(payload.get("delete_marked"), name="delete_marked"),
                "export_marked": deps.optional_bool(payload.get("export_marked"), name="export_marked"),
                "updated_time": deps.utc_now(),
            }
            if selection is None:
                updated = deps.update_review_state_batch(
                    connection,
                    file_ids=deps.required_int_list(payload.get("file_ids"), name="file_ids"),
                    **batch_kwargs,
                )
            else:
                snapshot_active = _begin_consistent_snapshot(connection)
                try:
                    _route(context, "_validate_selection_revision", _validate_selection_revision)(connection, deps, selection)
                    updated = 0
                    for file_ids in _route(context, "_frozen_selection_batches", _frozen_selection_batches)(connection, deps, selection):
                        updated += deps.update_review_state_batch(
                            connection,
                            file_ids=file_ids,
                            **batch_kwargs,
                        )
                except Exception:
                    _finish_consistent_snapshot(connection, active=snapshot_active, success=False)
                    raise
                else:
                    _finish_consistent_snapshot(connection, active=snapshot_active, success=True)
        emit_json(handler, {"updated": updated})
        return True

    return False
