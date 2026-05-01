from __future__ import annotations

import json
import sqlite3
import socket
from dataclasses import dataclass
from http import HTTPStatus
from pathlib import Path
from typing import Any, Callable, TypedDict, cast
from urllib.parse import parse_qs, urlparse

from shotsieve.config import normalize_raw_preview_mode
from shotsieve.job_registry import JobRegistry
from shotsieve.scoring import AnalysisProgress
from shotsieve.web_media import MediaDependencies, resolve_media_request, serve_media_response
from shotsieve.web_request import CompareRequest, ScanRequest, try_parse_http_status


_STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
    "/styles-layout.css": ("styles-layout.css", "text/css; charset=utf-8"),
    "/styles-polish.css": ("styles-polish.css", "text/css; charset=utf-8"),
    "/app-state.js": ("app-state.js", "application/javascript; charset=utf-8"),
    "/app.js": ("app.js", "application/javascript; charset=utf-8"),
    "/app-utils.js": ("app-utils.js", "application/javascript; charset=utf-8"),
    "/app-busy.js": ("app-busy.js", "application/javascript; charset=utf-8"),
    "/app-review.js": ("app-review.js", "application/javascript; charset=utf-8"),
    "/app-workflow-polling.js": ("app-workflow-polling.js", "application/javascript; charset=utf-8"),
    "/app-workflows.js": ("app-workflows.js", "application/javascript; charset=utf-8"),
    "/app-events.js": ("app-events.js", "application/javascript; charset=utf-8"),
}


_SELECTION_BATCH_SIZE = 500


class DeleteResultPayload(TypedDict):
    deleted_ids: list[int]
    deleted_count: int
    failed: list[object]
    failed_count: int
    delete_from_disk: bool


class ExportResultPayload(TypedDict):
    copied: int
    moved: int
    failed: list[object]


@dataclass(slots=True)
class ExportAggregate:
    copied: int
    moved: int
    failed: list[object]


def _is_ignorable_client_disconnect(exc: BaseException) -> bool:
    if isinstance(exc, (BrokenPipeError, ConnectionAbortedError, ConnectionResetError, TimeoutError, socket.timeout)):
        return True
    return isinstance(exc, OSError) and "timed out" in str(exc).casefold()


@dataclass(frozen=True)
class WebRouteDependencies:
    coerce_bool: Callable[..., bool]
    first_value: Callable[[dict[str, list[str]], str, str | None], str | None]
    float_or_none: Callable[[str | None], float | None]
    int_or_default: Callable[..., int]
    optional_bool: Callable[..., bool | None]
    optional_int: Callable[..., int | None]
    optional_string: Callable[[object], str | None]
    required_choice: Callable[..., str]
    required_int: Callable[..., int]
    required_int_list: Callable[..., list[int]]
    required_path: Callable[..., Path]
    read_json_body: Callable[..., dict[str, object]]
    parse_scan_request: Callable[[dict[str, object]], ScanRequest]
    parse_compare_request: Callable[..., CompareRequest]
    database: Callable[[Path], Any]
    build_options_payload: Callable[..., dict[str, object]]
    filesystem_roots: Callable[[], list[dict[str, str]]]
    list_directory: Callable[[Path], dict[str, object]]
    review_overview: Callable[[Any], object]
    list_review_files: Callable[..., object]
    count_review_files: Callable[..., int]
    review_selection_revision: Callable[..., str]
    list_review_browser_file_ids: Callable[..., list[int]]
    list_review_state_file_ids: Callable[..., list[int]]
    get_review_file_detail: Callable[[Any, int], object | None]
    update_review_state: Callable[..., None]
    update_review_state_batch: Callable[..., int]
    media_path_for_file: Callable[..., Path | None]
    build_config: Callable[..., Any]
    is_within_any_root: Callable[[Path, list[Path]], bool]
    stable_preview_name: Callable[[Path], str]
    preview_name_candidates: Callable[[Path], list[str]]
    guess_media_type: Callable[[str], tuple[str | None, str | None]]
    utc_now: Callable[[], str]
    scan_root: Callable[..., Any]
    score_files: Callable[..., Any]
    compare_learned_models: Callable[..., Any]
    require_learned_runtime: Callable[..., None]
    get_preview_cache_root: Callable[..., Path]
    count_score_rows: Callable[..., int]
    clear_cache_scope: Callable[..., dict[str, int]]
    prune_missing_cache_entries: Callable[..., int]
    reveal_in_file_manager: Callable[[Path], str]
    delete_files: Callable[..., object]
    export_files: Callable[..., Any]
    default_batch_size: Callable[[], int]
    thread_factory: Callable[..., Any]


@dataclass(frozen=True)
class WebRouteContext:
    db_path: Path
    operation_lock: Any
    scan_registry: JobRegistry | None
    score_registry: JobRegistry | None
    compare_registry: JobRegistry | None
    max_request_body_size: int
    static_dir: Path
    media_mime_fallbacks: dict[str, str]
    dependencies: object
    operation_registry: JobRegistry | None = None


def _require_registry(registry: JobRegistry | None, *, label: str) -> JobRegistry:
    if registry is None:
        raise RuntimeError(f"{label} registry is unavailable")
    return registry


def _scan_request_roots(scan_request: ScanRequest) -> list[Path]:
    return scan_request["roots"]


def _scan_request_offset(scan_request: ScanRequest) -> int:
    return scan_request["offset"]


def _scan_request_total_hint(scan_request: ScanRequest) -> int:
    return scan_request["files_total_hint"]


def _compare_request_models(compare_request: CompareRequest) -> list[str]:
    return compare_request["models"]


def _selection_excluded_ids(selection: dict[str, object]) -> set[int]:
    raw_excluded_ids = selection.get("exclude_file_ids", [])
    if not isinstance(raw_excluded_ids, list):
        return set()
    return {int(file_id) for file_id in raw_excluded_ids}


def _delete_result_payload(result: object) -> DeleteResultPayload:
    if not isinstance(result, dict):
        raise TypeError("delete_files must return a mapping payload")

    raw_deleted_ids = result.get("deleted_ids", [])
    deleted_ids = [int(file_id) for file_id in raw_deleted_ids] if isinstance(raw_deleted_ids, list) else []
    raw_failed = result.get("failed", [])
    failed = list(raw_failed) if isinstance(raw_failed, list) else []
    return {
        "deleted_ids": deleted_ids,
        "deleted_count": int(result.get("deleted_count", 0) or 0),
        "failed": failed,
        "failed_count": int(result.get("failed_count", 0) or 0),
        "delete_from_disk": bool(result.get("delete_from_disk", False)),
    }


def _export_result_payload(result: object) -> ExportResultPayload:
    raw_failed = getattr(result, "failed", [])
    failed = list(raw_failed) if isinstance(raw_failed, list) else []
    return {
        "copied": int(getattr(result, "copied", 0) or 0),
        "moved": int(getattr(result, "moved", 0) or 0),
        "failed": failed,
    }


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


def _handle_static_get_routes(handler: Any, context: WebRouteContext, parsed: Any) -> bool:
    static_entry = _STATIC_FILES.get(parsed.path)
    if static_entry is None:
        return False
    serve_static(handler, *static_entry, static_dir=context.static_dir)
    return True


def _handle_overview_get_routes(handler: Any, context: WebRouteContext, parsed: Any) -> bool:
    deps = cast(WebRouteDependencies, context.dependencies)
    if parsed.path == "/api/overview":
        with deps.database(context.db_path) as connection:
            send_json(handler, deps.review_overview(connection))
        return True

    if parsed.path == "/api/options":
        params = parse_qs(parsed.query)
        profile = deps.first_value(params, "resource_profile", None) or None
        send_json(handler, deps.build_options_payload(context.db_path, resource_profile=profile))
        return True

    return False


def _handle_job_get_routes(handler: Any, context: WebRouteContext, parsed: Any) -> bool:
    status_routes = {
        "/api/compare-models/status": (context.compare_registry, "Compare"),
        "/api/operations/status": (context.operation_registry, "Operation"),
        "/api/score/status": (context.score_registry, "Score"),
        "/api/scan/status": (context.scan_registry, "Scan"),
    }
    status_route = status_routes.get(parsed.path)
    if status_route is not None:
        registry, label = status_route
        handle_job_status(handler, _require_registry(registry, label=label), label=label)
        return True

    result_routes = {
        "/api/compare-models/result": (context.compare_registry, "Compare"),
        "/api/operations/result": (context.operation_registry, "Operation"),
        "/api/score/result": (context.score_registry, "Score"),
        "/api/scan/result": (context.scan_registry, "Scan"),
    }
    result_route = result_routes.get(parsed.path)
    if result_route is None:
        return False

    registry, label = result_route
    handle_job_result(handler, _require_registry(registry, label=label), label=label)
    return True


def _handle_filesystem_get_routes(handler: Any, context: WebRouteContext, parsed: Any) -> bool:
    deps = cast(WebRouteDependencies, context.dependencies)
    if parsed.path == "/api/fs/roots":
        send_json(handler, {"items": deps.filesystem_roots()})
        return True

    if parsed.path == "/api/fs/list":
        params = parse_qs(parsed.query)
        directory = deps.required_path(deps.first_value(params, "path", None), name="path")
        send_json(handler, deps.list_directory(directory))
        return True

    return False


def _handle_review_get_routes(handler: Any, context: WebRouteContext, parsed: Any) -> bool:
    deps = cast(WebRouteDependencies, context.dependencies)
    if parsed.path == "/api/files":
        params = parse_qs(parsed.query)
        with deps.database(context.db_path) as connection:
            snapshot_active = _begin_consistent_snapshot(connection)
            try:
                payload = deps.list_review_files(
                    connection,
                    root=deps.first_value(params, "root", None),
                    sort=deps.first_value(params, "sort", "score_asc"),
                    marked=deps.first_value(params, "marked", "all"),
                    issues=deps.first_value(params, "issues", "all"),
                    query=deps.first_value(params, "query", None),
                    min_score=deps.float_or_none(deps.first_value(params, "min_score", None)),
                    max_score=deps.float_or_none(deps.first_value(params, "max_score", None)),
                    limit=deps.int_or_default(deps.first_value(params, "limit", "60"), default=60, minimum=1, maximum=500),
                    offset=deps.int_or_default(deps.first_value(params, "offset", "0"), default=0, minimum=0),
                )
                total = deps.count_review_files(
                    connection,
                    root=deps.first_value(params, "root", None),
                    marked=deps.first_value(params, "marked", "all"),
                    issues=deps.first_value(params, "issues", "all"),
                    query=deps.first_value(params, "query", None),
                    min_score=deps.float_or_none(deps.first_value(params, "min_score", None)),
                    max_score=deps.float_or_none(deps.first_value(params, "max_score", None)),
                )
                selection_revision = deps.review_selection_revision(
                    connection,
                    scope="review-browser",
                    root=deps.first_value(params, "root", None),
                    marked=deps.first_value(params, "marked", "all"),
                    issues=deps.first_value(params, "issues", "all"),
                    query=deps.first_value(params, "query", None),
                    min_score=deps.float_or_none(deps.first_value(params, "min_score", None)),
                    max_score=deps.float_or_none(deps.first_value(params, "max_score", None)),
                )
            except Exception:
                _finish_consistent_snapshot(connection, active=snapshot_active, success=False)
                raise
            else:
                _finish_consistent_snapshot(connection, active=snapshot_active, success=True)
        send_json(handler, {"items": payload, "total": total, "selection_revision": selection_revision})
        return True

    if parsed.path == "/api/files/count":
        params = parse_qs(parsed.query)
        with deps.database(context.db_path) as connection:
            snapshot_active = _begin_consistent_snapshot(connection)
            try:
                total = deps.count_review_files(
                    connection,
                    root=deps.first_value(params, "root", None),
                    marked=deps.first_value(params, "marked", "all"),
                    issues=deps.first_value(params, "issues", "all"),
                    query=deps.first_value(params, "query", None),
                    min_score=deps.float_or_none(deps.first_value(params, "min_score", None)),
                    max_score=deps.float_or_none(deps.first_value(params, "max_score", None)),
                )
                selection_revision = deps.review_selection_revision(
                    connection,
                    scope="review-browser",
                    root=deps.first_value(params, "root", None),
                    marked=deps.first_value(params, "marked", "all"),
                    issues=deps.first_value(params, "issues", "all"),
                    query=deps.first_value(params, "query", None),
                    min_score=deps.float_or_none(deps.first_value(params, "min_score", None)),
                    max_score=deps.float_or_none(deps.first_value(params, "max_score", None)),
                )
            except Exception:
                _finish_consistent_snapshot(connection, active=snapshot_active, success=False)
                raise
            else:
                _finish_consistent_snapshot(connection, active=snapshot_active, success=True)
        send_json(handler, {"total": total, "selection_revision": selection_revision})
        return True

    if parsed.path == "/api/review/file-ids":
        params = parse_qs(parsed.query)
        marked = deps.required_choice(
            deps.first_value(params, "marked", None),
            name="marked",
            choices=("delete", "export", "none"),
        )
        with deps.database(context.db_path) as connection:
            snapshot_active = _begin_consistent_snapshot(connection)
            try:
                ids = deps.list_review_state_file_ids(
                    connection,
                    marked=marked,
                    root=deps.first_value(params, "root", None),
                    query=deps.first_value(params, "query", None),
                    limit=deps.int_or_default(deps.first_value(params, "limit", "500"), default=500, minimum=1, maximum=1000),
                    offset=deps.int_or_default(deps.first_value(params, "offset", "0"), default=0, minimum=0),
                )
                selection_revision = deps.review_selection_revision(
                    connection,
                    scope="review-state",
                    marked=marked,
                    root=deps.first_value(params, "root", None),
                    query=deps.first_value(params, "query", None),
                )
            except Exception:
                _finish_consistent_snapshot(connection, active=snapshot_active, success=False)
                raise
            else:
                _finish_consistent_snapshot(connection, active=snapshot_active, success=True)
        send_json(handler, {"ids": ids, "selection_revision": selection_revision})
        return True

    if parsed.path == "/api/file":
        params = parse_qs(parsed.query)
        file_id = deps.required_int(deps.first_value(params, "id", None), name="id", minimum=1)
        with deps.database(context.db_path) as connection:
            detail = deps.get_review_file_detail(connection, file_id)
        if detail is None:
            handler.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return True
        send_json(handler, detail)
        return True

    return False


def _handle_media_get_routes(handler: Any, context: WebRouteContext, parsed: Any) -> bool:
    if parsed.path not in {"/api/media/preview", "/api/media/source"}:
        return False

    deps = cast(WebRouteDependencies, context.dependencies)
    params = parse_qs(parsed.query)
    file_id = deps.required_int(deps.first_value(params, "id", None), name="id", minimum=1)
    variant = "preview" if parsed.path.endswith("preview") else "source"
    media_result = resolve_media_request(
        db_path=context.db_path,
        file_id=file_id,
        variant=variant,
        dependencies=MediaDependencies(
            database=deps.database,
            build_config=deps.build_config,
            is_within_any_root=deps.is_within_any_root,
            media_path_for_file=deps.media_path_for_file,
            stable_preview_name=deps.stable_preview_name,
            preview_name_candidates=deps.preview_name_candidates,
            guess_media_type=deps.guess_media_type,
        ),
    )
    if media_result.error_status is not None:
        handler.send_error(media_result.error_status, media_result.error_message or "Media request failed")
        return True
    if media_result.path is None:
        handler.send_error(HTTPStatus.NOT_FOUND, "Image not found")
        return True
    serve_media_response(
        handler,
        media_result.path,
        guess_media_type=deps.guess_media_type,
        mime_fallbacks=context.media_mime_fallbacks,
    )
    return True


def _optional_payload_float(deps: WebRouteDependencies, value: object, *, name: str) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return deps.float_or_none(value)
    raise ValueError(f"{name} must be numeric")


def _begin_consistent_snapshot(connection: Any) -> bool:
    execute = getattr(connection, "execute", None)
    if not callable(execute):
        return False
    try:
        execute("BEGIN")
    except Exception:
        if isinstance(connection, sqlite3.Connection):
            raise
        return False
    return True


def _finish_consistent_snapshot(connection: Any, *, active: bool, success: bool) -> None:
    if not active:
        return
    finalize = getattr(connection, "commit" if success else "rollback", None)
    if callable(finalize):
        finalize()


def _parse_selection_payload(deps: WebRouteDependencies, payload: dict[str, object]) -> dict[str, object] | None:
    raw_selection = payload.get("selection")
    if raw_selection is None:
        return None
    if not isinstance(raw_selection, dict):
        raise ValueError("selection must be an object")

    scope = deps.required_choice(
        raw_selection.get("scope"),
        name="selection.scope",
        choices=("review-browser", "review-state"),
    )
    selection: dict[str, object] = {
        "scope": scope,
        "marked": deps.required_choice(
            raw_selection.get("marked") or "all",
            name="selection.marked",
            choices=("all", "delete", "export", "none"),
        ),
        "root": deps.optional_string(raw_selection.get("root")),
        "query": deps.optional_string(raw_selection.get("query")),
        "selection_revision": deps.optional_string(payload.get("selection_revision")),
    }
    if scope == "review-browser":
        selection["issues"] = deps.required_choice(
            raw_selection.get("issues") or "all",
            name="selection.issues",
            choices=("all", "issues"),
        )
        selection["min_score"] = _optional_payload_float(deps, raw_selection.get("min_score"), name="selection.min_score")
        selection["max_score"] = _optional_payload_float(deps, raw_selection.get("max_score"), name="selection.max_score")

    raw_exclude_ids = payload.get("exclude_file_ids")
    if raw_exclude_ids is None or raw_exclude_ids == []:
        selection["exclude_file_ids"] = []
    else:
        selection["exclude_file_ids"] = deps.required_int_list(raw_exclude_ids, name="exclude_file_ids")
    return selection


def _validate_selection_revision(connection: Any, deps: WebRouteDependencies, selection: dict[str, object]) -> None:
    selection_revision = selection.get("selection_revision")
    if not selection_revision:
        raise ValueError("selection_revision is required for filter-based bulk actions")
    current_revision = deps.review_selection_revision(
        connection,
        scope=selection["scope"],
        root=selection.get("root"),
        marked=selection["marked"],
        issues=selection.get("issues", "all"),
        query=selection.get("query"),
        min_score=selection.get("min_score"),
        max_score=selection.get("max_score"),
    )
    if selection_revision != current_revision:
        raise ValueError("Selected results changed. Refresh the queue and select again.")


def _materialize_selection_batches(connection: Any, deps: WebRouteDependencies, selection: dict[str, object]) -> list[list[int]]:
    batches: list[list[int]] = []
    for batch in _iter_selection_file_id_batches(connection, deps, selection):
        batches.append(list(batch))
    return batches


def _iter_sqlite_materialized_selection_batches(connection: sqlite3.Connection, deps: WebRouteDependencies, selection: dict[str, object]):
    table_name = "temp._shotsieve_selected_ids"
    connection.execute(f"DROP TABLE IF EXISTS {table_name}")
    connection.execute(
        f"""
        CREATE TEMP TABLE {table_name} (
            ordinal INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id INTEGER NOT NULL
        )
        """
    )
    try:
        for batch in _iter_selection_file_id_batches(connection, deps, selection):
            connection.executemany(
                f"INSERT INTO {table_name} (file_id) VALUES (?)",
                [(int(file_id),) for file_id in batch],
            )

        after_ordinal = 0
        while True:
            rows = connection.execute(
                f"""
                SELECT ordinal, file_id
                FROM {table_name}
                WHERE ordinal > ?
                ORDER BY ordinal ASC
                LIMIT ?
                """,
                (after_ordinal, _SELECTION_BATCH_SIZE),
            ).fetchall()
            if not rows:
                break
            after_ordinal = int(rows[-1]["ordinal"])
            yield [int(row["file_id"]) for row in rows]
    finally:
        connection.execute(f"DROP TABLE IF EXISTS {table_name}")


def _frozen_selection_batches(connection: Any, deps: WebRouteDependencies, selection: dict[str, object]):
    if isinstance(connection, sqlite3.Connection):
        yield from _iter_sqlite_materialized_selection_batches(connection, deps, selection)
        return
    yield from _materialize_selection_batches(connection, deps, selection)


def _iter_selection_file_id_batches(connection: Any, deps: WebRouteDependencies, selection: dict[str, object]):
    after_id = 0
    excluded_ids = _selection_excluded_ids(selection)
    while True:
        if selection["scope"] == "review-browser":
            raw_file_ids = deps.list_review_browser_file_ids(
                connection,
                root=selection.get("root"),
                marked=selection["marked"],
                issues=selection.get("issues", "all"),
                query=selection.get("query"),
                min_score=selection.get("min_score"),
                max_score=selection.get("max_score"),
                limit=_SELECTION_BATCH_SIZE,
                after_id=after_id,
            )
        else:
            raw_file_ids = deps.list_review_state_file_ids(
                connection,
                marked=selection["marked"],
                root=selection.get("root"),
                query=selection.get("query"),
                limit=_SELECTION_BATCH_SIZE,
                after_id=after_id,
            )

        if not raw_file_ids:
            break

        after_id = int(raw_file_ids[-1])
        file_ids = [file_id for file_id in raw_file_ids if file_id not in excluded_ids]

        if not file_ids:
            continue

        yield file_ids


def _handle_review_post_routes(handler: Any, context: WebRouteContext, parsed: Any) -> bool:
    deps = cast(WebRouteDependencies, context.dependencies)
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
        send_json(handler, detail or {"ok": True})
        return True

    if parsed.path == "/api/review/batch":
        payload = deps.read_json_body(handler, max_body_size=context.max_request_body_size)
        selection = _parse_selection_payload(deps, payload)
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
                    _validate_selection_revision(connection, deps, selection)
                    updated = 0
                    for file_ids in _frozen_selection_batches(connection, deps, selection):
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
        send_json(handler, {"updated": updated})
        return True

    return False


def _handle_analysis_post_routes(handler: Any, context: WebRouteContext, parsed: Any) -> bool:
    deps = cast(WebRouteDependencies, context.dependencies)
    if parsed.path == "/api/scan/start":
        payload = deps.read_json_body(handler, max_body_size=context.max_request_body_size)
        start_scan_job(handler, context, payload)
        return True

    if parsed.path == "/api/score/start":
        payload = deps.read_json_body(handler, max_body_size=context.max_request_body_size)
        start_score_job(handler, context, payload)
        return True

    if parsed.path == "/api/compare-models/start":
        payload = deps.read_json_body(handler, max_body_size=context.max_request_body_size)
        start_compare_job(handler, context, payload)
        return True

    if parsed.path in {"/api/score-estimate", "/api/compare-estimate"}:
        _send_rows_total_estimate(handler, context)
        return True

    return False


def _handle_job_cancel_post_routes(handler: Any, context: WebRouteContext, parsed: Any) -> bool:
    cancel_routes = {
        "/api/compare-models/cancel": context.compare_registry,
        "/api/operations/cancel": context.operation_registry,
        "/api/score/cancel": context.score_registry,
        "/api/scan/cancel": context.scan_registry,
    }
    registry = cancel_routes.get(parsed.path)
    if registry is None:
        return False

    handle_job_cancel(handler, _require_registry(registry, label="Job"), max_request_body_size=context.max_request_body_size)
    return True


def _handle_cache_post_routes(handler: Any, context: WebRouteContext, parsed: Any) -> bool:
    if parsed.path == "/api/cache/clear/start":
        deps = cast(WebRouteDependencies, context.dependencies)
        payload = deps.read_json_body(handler, max_body_size=context.max_request_body_size)
        start_cache_clear_job(handler, context, payload)
        return True

    if parsed.path != "/api/cache/clear":
        return False

    deps = cast(WebRouteDependencies, context.dependencies)
    payload = deps.read_json_body(handler, max_body_size=context.max_request_body_size)
    result = _execute_cache_clear_request(context, payload, progress_callback=None, cancel_check=None)
    send_json(handler, result)
    return True


def _handle_file_action_post_routes(handler: Any, context: WebRouteContext, parsed: Any) -> bool:
    deps = cast(WebRouteDependencies, context.dependencies)
    if parsed.path == "/api/files/open":
        payload = deps.read_json_body(handler, max_body_size=context.max_request_body_size)
        file_id = deps.required_int(payload.get("file_id"), name="file_id", minimum=1)
        media_result = resolve_media_request(
            db_path=context.db_path,
            file_id=file_id,
            variant="source",
            dependencies=MediaDependencies(
                database=deps.database,
                build_config=deps.build_config,
                is_within_any_root=deps.is_within_any_root,
                media_path_for_file=deps.media_path_for_file,
                stable_preview_name=deps.stable_preview_name,
                preview_name_candidates=deps.preview_name_candidates,
                guess_media_type=deps.guess_media_type,
            ),
        )
        if media_result.error_status is not None:
            raise ValueError(media_result.error_message or "File not found")
        if media_result.path is None:
            raise ValueError("File not found")
        method = deps.reveal_in_file_manager(media_result.path)
        send_json(handler, {"opened": True, "path": str(media_result.path), "method": method})
        return True

    if parsed.path == "/api/files/delete":
        payload = deps.read_json_body(handler, max_body_size=context.max_request_body_size)
        delete_result = _execute_delete_request(context, payload, progress_callback=None, cancel_check=None)
        send_json(handler, delete_result)
        return True

    if parsed.path == "/api/files/delete/start":
        payload = deps.read_json_body(handler, max_body_size=context.max_request_body_size)
        start_delete_job(handler, context, payload)
        return True

    if parsed.path == "/api/files/export":
        payload = deps.read_json_body(handler, max_body_size=context.max_request_body_size)
        export_result = _execute_export_request(context, payload, progress_callback=None, cancel_check=None)
        send_json(
            handler,
            _export_result_payload(export_result),
        )
        return True

    if parsed.path == "/api/files/export/start":
        payload = deps.read_json_body(handler, max_body_size=context.max_request_body_size)
        start_export_job(handler, context, payload)
        return True

    return False


def _progress_payload(phase: str, *, files_processed: int = 0, files_total: int = 0) -> dict[str, object]:
    return {
        "phase": phase,
        "files_processed": max(0, int(files_processed)),
        "files_total": max(0, int(files_total)),
    }


def _progress_total_hint(deps: WebRouteDependencies, payload: dict[str, object]) -> int | None:
    optional_int = getattr(deps, "optional_int", None)
    if callable(optional_int):
        return optional_int(payload.get("count"), minimum=0)

    raw_value = payload.get("count")
    if raw_value is None or raw_value == "":
        return None
    if isinstance(raw_value, bool):
        raise ValueError("count must be an integer")
    if isinstance(raw_value, int):
        return max(0, raw_value)
    if isinstance(raw_value, float):
        return max(0, int(raw_value))
    if isinstance(raw_value, str):
        return max(0, int(raw_value))
    raise ValueError("count must be an integer")


def _operation_progress_callback(
    progress_callback: Callable[[int, int, str], None] | None,
    *,
    phase: str,
    offset: int,
    total_hint: int | None,
):
    if progress_callback is None:
        return None

    def update(local_processed: int, local_total: int) -> None:
        total = total_hint if total_hint is not None else (offset + max(0, int(local_total)))
        processed = offset + max(0, int(local_processed))
        progress_callback(processed, total, phase)

    return update


def _execute_delete_request(
    context: WebRouteContext,
    payload: dict[str, object],
    *,
    progress_callback: Callable[[int, int, str], None] | None,
    cancel_check: Callable[[], None] | None,
) -> DeleteResultPayload:
    deps = cast(WebRouteDependencies, context.dependencies)
    selection = _parse_selection_payload(deps, payload)
    delete_from_disk = deps.coerce_bool(payload.get("delete_from_disk"), default=False)
    total_hint = _progress_total_hint(deps, payload)

    with deps.database(context.db_path) as connection:
        preview_cache_root = deps.get_preview_cache_root(connection, db_path=context.db_path, persist=False)
        if selection is None:
            file_ids = deps.required_int_list(payload.get("file_ids"), name="file_ids")
            total = total_hint if total_hint is not None else len(file_ids)
            if progress_callback is not None:
                progress_callback(0, total, "deleting_files")
            return _delete_result_payload(deps.delete_files(
                connection,
                file_ids=file_ids,
                delete_from_disk=delete_from_disk,
                preview_cache_root=preview_cache_root,
                progress_callback=_operation_progress_callback(progress_callback, phase="deleting_files", offset=0, total_hint=total),
                cancel_check=cancel_check,
            ))

        snapshot_active = _begin_consistent_snapshot(connection)
        try:
            _validate_selection_revision(connection, deps, selection)
            delete_result: DeleteResultPayload = {
                "deleted_ids": [],
                "deleted_count": 0,
                "failed": [],
                "failed_count": 0,
                "delete_from_disk": delete_from_disk,
            }
            processed_so_far = 0
            if progress_callback is not None:
                progress_callback(0, total_hint or 0, "deleting_files")
            for file_ids in _frozen_selection_batches(connection, deps, selection):
                if cancel_check is not None:
                    cancel_check()
                batch_total = total_hint if total_hint is not None else processed_so_far + len(file_ids)
                batch_result = _delete_result_payload(deps.delete_files(
                    connection,
                    file_ids=file_ids,
                    delete_from_disk=delete_from_disk,
                    preview_cache_root=preview_cache_root,
                    progress_callback=_operation_progress_callback(progress_callback, phase="deleting_files", offset=processed_so_far, total_hint=batch_total),
                    cancel_check=cancel_check,
                ))
                delete_result["deleted_ids"].extend(batch_result["deleted_ids"])
                delete_result["deleted_count"] += int(batch_result["deleted_count"])
                delete_result["failed"].extend(batch_result["failed"])
                delete_result["failed_count"] += int(batch_result["failed_count"])
                processed_so_far += len(file_ids)
                if progress_callback is not None:
                    progress_callback(processed_so_far, total_hint or processed_so_far, "deleting_files")
        except Exception:
            _finish_consistent_snapshot(connection, active=snapshot_active, success=False)
            raise
        else:
            _finish_consistent_snapshot(connection, active=snapshot_active, success=True)

    return delete_result


def _execute_export_request(
    context: WebRouteContext,
    payload: dict[str, object],
    *,
    progress_callback: Callable[[int, int, str], None] | None,
    cancel_check: Callable[[], None] | None,
) -> object:
    deps = cast(WebRouteDependencies, context.dependencies)
    selection = _parse_selection_payload(deps, payload)
    destination = deps.optional_string(payload.get("destination"))
    mode_raw = payload.get("mode")
    mode = (
        "copy"
        if mode_raw is None
        else deps.required_choice(mode_raw, name="mode", choices=("copy", "move"))
    )
    phase = "moving_files" if mode == "move" else "exporting_files"
    total_hint = _progress_total_hint(deps, payload)
    if not destination:
        raise ValueError("destination is required")

    with deps.database(context.db_path) as connection:
        preview_cache_root = deps.get_preview_cache_root(connection, db_path=context.db_path, persist=False)
        if selection is None:
            file_ids = deps.required_int_list(payload.get("file_ids"), name="file_ids")
            total = total_hint if total_hint is not None else len(file_ids)
            if progress_callback is not None:
                progress_callback(0, total, phase)
            return deps.export_files(
                connection,
                file_ids=file_ids,
                destination=destination,
                mode=mode,
                preview_cache_root=preview_cache_root,
                progress_callback=_operation_progress_callback(progress_callback, phase=phase, offset=0, total_hint=total),
                cancel_check=cancel_check,
            )

        snapshot_active = _begin_consistent_snapshot(connection)
        try:
            _validate_selection_revision(connection, deps, selection)
            copied = 0
            moved = 0
            failed: list[object] = []
            processed_so_far = 0
            if progress_callback is not None:
                progress_callback(0, total_hint or 0, phase)
            for file_ids in _frozen_selection_batches(connection, deps, selection):
                if cancel_check is not None:
                    cancel_check()
                batch_total = total_hint if total_hint is not None else processed_so_far + len(file_ids)
                batch_result = deps.export_files(
                    connection,
                    file_ids=file_ids,
                    destination=destination,
                    mode=mode,
                    preview_cache_root=preview_cache_root,
                    progress_callback=_operation_progress_callback(progress_callback, phase=phase, offset=processed_so_far, total_hint=batch_total),
                    cancel_check=cancel_check,
                )
                copied += int(getattr(batch_result, "copied", 0) or 0)
                moved += int(getattr(batch_result, "moved", 0) or 0)
                failed.extend(list(getattr(batch_result, "failed", []) or []))
                processed_so_far += len(file_ids)
                if progress_callback is not None:
                    progress_callback(processed_so_far, total_hint or processed_so_far, phase)
            export_result = ExportAggregate(copied=copied, moved=moved, failed=failed)
        except Exception:
            _finish_consistent_snapshot(connection, active=snapshot_active, success=False)
            raise
        else:
            _finish_consistent_snapshot(connection, active=snapshot_active, success=True)

    return export_result


def _execute_cache_clear_request(
    context: WebRouteContext,
    payload: dict[str, object],
    *,
    progress_callback: Callable[[int, int, str], None] | None,
    cancel_check: Callable[[], None] | None,
) -> dict[str, int]:
    deps = cast(WebRouteDependencies, context.dependencies)
    scope = deps.required_choice(payload.get("scope"), name="scope", choices=("scores", "review", "all", "missing"))
    with deps.database(context.db_path) as connection:
        preview_cache_root = deps.get_preview_cache_root(connection, db_path=context.db_path, persist=False)
        if scope == "missing":
            if progress_callback is not None:
                progress_callback(0, 1, "clearing_cache")
            if cancel_check is not None:
                cancel_check()
            removed = deps.prune_missing_cache_entries(connection, preview_cache_root=preview_cache_root)
            if progress_callback is not None:
                progress_callback(1, 1, "clearing_cache")
            return {"files": removed, "scores": 0, "review": 0, "scan_runs": 0}
        return deps.clear_cache_scope(
            connection,
            scope=scope,
            preview_cache_root=preview_cache_root,
            progress_callback=progress_callback,
            cancel_check=cancel_check,
        )


def start_delete_job(handler: Any, context: WebRouteContext, payload: dict[str, object]) -> None:
    registry = _require_registry(context.operation_registry, label="Operation")
    deps = cast(WebRouteDependencies, context.dependencies)
    if not try_acquire_operation_lock(handler, context):
        return

    total_hint = _progress_total_hint(deps, payload) or 0
    job_id = registry.create(initial_progress=_progress_payload("deleting_files", files_processed=0, files_total=total_hint))

    def run_job() -> None:
        try:
            def publish(processed: int, total: int, phase: str) -> None:
                registry.update_progress(job_id, _progress_payload(phase, files_processed=processed, files_total=total))

            def cancel_check() -> None:
                if registry.is_cancelled(job_id):
                    raise InterruptedError("Delete job was cancelled by user.")

            result = _execute_delete_request(context, payload, progress_callback=publish, cancel_check=cancel_check)
            registry.complete(job_id, summary=result)
        except Exception as exc:
            registry.fail(job_id, error=str(exc))
        finally:
            context.operation_lock.release()

    deps.thread_factory(target=run_job, daemon=True).start()
    send_json(handler, {"job_id": job_id, "status": "running"})


def start_export_job(handler: Any, context: WebRouteContext, payload: dict[str, object]) -> None:
    registry = _require_registry(context.operation_registry, label="Operation")
    deps = cast(WebRouteDependencies, context.dependencies)
    if not try_acquire_operation_lock(handler, context):
        return

    phase = "moving_files" if str(payload.get("mode") or "copy") == "move" else "exporting_files"
    total_hint = _progress_total_hint(deps, payload) or 0
    job_id = registry.create(initial_progress=_progress_payload(phase, files_processed=0, files_total=total_hint))

    def run_job() -> None:
        try:
            def publish(processed: int, total: int, phase_name: str) -> None:
                registry.update_progress(job_id, _progress_payload(phase_name, files_processed=processed, files_total=total))

            def cancel_check() -> None:
                if registry.is_cancelled(job_id):
                    raise InterruptedError("Export job was cancelled by user.")

            result = _execute_export_request(context, payload, progress_callback=publish, cancel_check=cancel_check)
            registry.complete(job_id, summary=_export_result_payload(result))
        except Exception as exc:
            registry.fail(job_id, error=str(exc))
        finally:
            context.operation_lock.release()

    deps.thread_factory(target=run_job, daemon=True).start()
    send_json(handler, {"job_id": job_id, "status": "running"})


def start_cache_clear_job(handler: Any, context: WebRouteContext, payload: dict[str, object]) -> None:
    registry = _require_registry(context.operation_registry, label="Operation")
    deps = cast(WebRouteDependencies, context.dependencies)
    if not try_acquire_operation_lock(handler, context):
        return

    job_id = registry.create(initial_progress=_progress_payload("clearing_cache", files_processed=0, files_total=1))

    def run_job() -> None:
        try:
            def publish(processed: int, total: int, phase_name: str) -> None:
                registry.update_progress(job_id, _progress_payload(phase_name, files_processed=processed, files_total=total))

            def cancel_check() -> None:
                if registry.is_cancelled(job_id):
                    raise InterruptedError("Cache clear job was cancelled by user.")

            result = _execute_cache_clear_request(context, payload, progress_callback=publish, cancel_check=cancel_check)
            registry.complete(job_id, summary=result)
        except Exception as exc:
            registry.fail(job_id, error=str(exc))
        finally:
            context.operation_lock.release()

    deps.thread_factory(target=run_job, daemon=True).start()
    send_json(handler, {"job_id": job_id, "status": "running"})


def _send_rows_total_estimate(handler: Any, context: WebRouteContext) -> None:
    deps = cast(WebRouteDependencies, context.dependencies)
    payload = deps.read_json_body(handler, max_body_size=context.max_request_body_size)
    with deps.database(context.db_path) as connection:
        rows_total = deps.count_score_rows(
            connection,
            raw_root=deps.optional_string(payload.get("root")),
        )
    send_json(handler, {"rows_total": rows_total})


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


def start_scan_job(handler: Any, context: WebRouteContext, payload: dict[str, object]) -> None:
    deps = cast(WebRouteDependencies, context.dependencies)
    scan_registry = _require_registry(context.scan_registry, label="Scan")
    scan_request = deps.parse_scan_request(payload)
    if not try_acquire_operation_lock(handler, context):
        return

    total_hint = max(0, _scan_request_total_hint(scan_request))
    job_id = scan_registry.create(initial_progress={
        "phase": "indexing",
        "files_processed": 0,
        "files_total": total_hint,
    })

    def run_scan_job() -> None:
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

            aggregated = {
                "files_seen": 0,
                "files_added": 0,
                "files_updated": 0,
                "files_unchanged": 0,
                "files_removed": 0,
                "files_failed": 0,
            }
            processed_before_root = 0
            remaining_offset = max(0, _scan_request_offset(scan_request))
            remaining_limit = scan_request["limit"]
            cancel_progress: dict[str, object] | None = None

            def publish_progress(processed_in_root: int, _root_total: int, phase: str) -> None:
                files_total = total_hint if total_hint > 0 else 0
                scan_registry.update_progress(job_id, {
                    "phase": phase,
                    "files_processed": max(0, processed_before_root + processed_in_root),
                    "files_total": files_total,
                })

            cancel_error: str | None = None
            with deps.database(config.db_path) as connection:
                for root in _scan_request_roots(scan_request):
                    try:
                        _raise_if_scan_cancelled(scan_registry, job_id)
                        if remaining_limit is not None and remaining_limit <= 0:
                            break

                        root_total_hint = None
                        if total_hint > 0:
                            root_total_hint = max(0, total_hint - processed_before_root)

                        root_offset = remaining_offset
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
                            cancel_check=lambda: _raise_if_scan_cancelled(scan_registry, job_id),
                        )
                        aggregated["files_seen"] += summary.files_seen
                        aggregated["files_added"] += summary.files_added
                        aggregated["files_updated"] += summary.files_updated
                        aggregated["files_unchanged"] += summary.files_unchanged
                        aggregated["files_removed"] += summary.files_removed
                        aggregated["files_failed"] += summary.files_failed
                        processed_before_root += summary.files_seen
                        remaining_offset = max(0, remaining_offset - _scan_offset_consumed(summary, requested_offset=root_offset))
                        if remaining_limit is not None:
                            remaining_limit = max(0, remaining_limit - summary.files_seen)
                    except InterruptedError as exc:
                        cancel_error = str(exc)
                        cancel_files_processed = max(
                            0,
                            processed_before_root + int(getattr(exc, "processed_count", 0) or 0),
                        )
                        cancel_progress = {
                            "phase": "failed",
                            "files_processed": cancel_files_processed,
                            "files_total": total_hint if total_hint > 0 else cancel_files_processed,
                        }
                        break

            if cancel_error is not None:
                scan_registry.fail(job_id, error=cancel_error, progress=cancel_progress)
                return

            scan_registry.update_progress(job_id, {
                "phase": "scanning",
                "files_processed": aggregated["files_seen"],
                "files_total": total_hint if total_hint > 0 else aggregated["files_seen"],
            })
            scan_registry.complete(job_id, summary=aggregated)
        except Exception as exc:
            scan_registry.fail(job_id, error=str(exc))
        finally:
            context.operation_lock.release()

    deps.thread_factory(target=run_scan_job, daemon=True).start()
    send_json(handler, {"job_id": job_id, "status": "running"})


def start_score_job(handler: Any, context: WebRouteContext, payload: dict[str, object]) -> None:
    deps = cast(WebRouteDependencies, context.dependencies)
    score_registry = _require_registry(context.score_registry, label="Score")
    learned_device = deps.optional_string(payload.get("device"))
    resource_profile = deps.optional_string(payload.get("resource_profile"))
    raw_preview_mode = normalize_raw_preview_mode(deps.optional_string(payload.get("preview_mode")))
    deps.require_learned_runtime(resource_profile=resource_profile, preferred_device=learned_device)

    if not try_acquire_operation_lock(handler, context):
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
                score_registry.update_progress(job_id, progress_payload(progress))

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
    send_json(handler, {"job_id": job_id, "status": "running"})


def start_compare_job(handler: Any, context: WebRouteContext, payload: dict[str, object]) -> None:
    deps = cast(WebRouteDependencies, context.dependencies)
    compare_registry = _require_registry(context.compare_registry, label="Compare")
    compare_request = deps.parse_compare_request(payload, default_batch_size=deps.default_batch_size())
    raw_preview_mode = normalize_raw_preview_mode(deps.optional_string(payload.get("preview_mode")))
    deps.require_learned_runtime(
        resource_profile=compare_request.get("resource_profile"),
        preferred_device=compare_request.get("device"),
    )

    if not try_acquire_operation_lock(handler, context):
        return

    job_id = compare_registry.create(initial_progress={
        "model_name": None,
        "model_index": 0,
        "model_count": len(_compare_request_models(compare_request)),
        "files_processed": 0,
        "files_total": 0,
    })

    def run_compare_job() -> None:
        try:
            def publish_progress(progress: AnalysisProgress) -> None:
                compare_registry.update_progress(job_id, progress_payload(progress))

            with deps.database(context.db_path) as connection:
                preview_dir = deps.get_preview_cache_root(connection, db_path=context.db_path, persist=False)
                summary = deps.compare_learned_models(
                    connection,
                    model_names=_compare_request_models(compare_request),
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

            compare_registry.complete(job_id, summary=comparison_summary_payload(summary))
        except Exception as exc:
            compare_registry.fail(job_id, error=str(exc))
        finally:
            context.operation_lock.release()

    deps.thread_factory(target=run_compare_job, daemon=True).start()
    send_json(handler, {"job_id": job_id, "status": "running"})


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
    send_json_error(
        handler,
        HTTPStatus.CONFLICT,
        "Another analysis operation is already running. Please wait for it to finish.",
    )
    return False


def handle_job_status(handler: Any, registry: JobRegistry, *, label: str) -> None:
    deps = cast(WebRouteDependencies, getattr(handler, "_shotsieve_route_dependencies"))
    parsed = urlparse(handler.path)
    params = parse_qs(parsed.query)
    job_id = deps.first_value(params, "job_id", None)
    if not job_id:
        raise ValueError("job_id is required")
    status_payload = registry.status(job_id)
    if status_payload is None:
        handler.send_error(HTTPStatus.NOT_FOUND, f"{label} job not found")
        return
    send_json(handler, status_payload)


def handle_job_result(handler: Any, registry: JobRegistry, *, label: str) -> None:
    deps = cast(WebRouteDependencies, getattr(handler, "_shotsieve_route_dependencies"))
    parsed = urlparse(handler.path)
    params = parse_qs(parsed.query)
    job_id = deps.first_value(params, "job_id", None)
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
            send_json(handler, summary_payload)
            return
        send_json_error(handler, HTTPStatus.INTERNAL_SERVER_ERROR, f"{label} job completed without a summary payload")
        return

    if status_value == "failed":
        send_json_error(handler, HTTPStatus.BAD_REQUEST, str(status_payload.get("error") or f"{label} job failed"))
        return

    send_json_error(handler, HTTPStatus.CONFLICT, f"{label} job is still running")


def handle_job_cancel(handler: Any, registry: JobRegistry, *, max_request_body_size: int) -> None:
    deps = cast(WebRouteDependencies, getattr(handler, "_shotsieve_route_dependencies"))
    parsed = urlparse(handler.path)
    params = parse_qs(parsed.query)
    job_id = deps.first_value(params, "job_id", None) or ""
    if not job_id:
        content_length = int(handler.headers.get("Content-Length", "0"))
        payload = deps.read_json_body(handler, max_body_size=max_request_body_size) if content_length > 0 else {}
        job_id = deps.optional_string(payload.get("job_id")) or ""
    if not job_id:
        raise ValueError("job_id is required")
    cancelled = registry.cancel(job_id)
    send_json(handler, {"job_id": job_id, "cancelled": cancelled})


def log_request_message(handler: Any, log: Any, format_string: str, *args: object) -> None:
    message = (format_string % args).translate(handler._control_char_table)
    status_code = next(
        (
            parsed
            for candidate in args[:2]
            for parsed in [try_parse_http_status(candidate)]
            if parsed is not None
        ),
        None,
    )
    log_method = log.warning if status_code is not None and status_code >= 400 else log.debug
    log_method(
        "%s - - [%s] %s",
        handler.address_string(),
        handler.log_date_time_string(),
        message,
    )


def serve_static(handler: Any, name: str, content_type: str, *, static_dir: Path) -> None:
    path = static_dir / name
    if not path.exists():
        handler.send_error(HTTPStatus.NOT_FOUND, "Static asset not found")
        return
    data = path.read_bytes()
    handler.send_response(HTTPStatus.OK)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Cache-Control", "no-cache, must-revalidate")
    handler.end_headers()
    try:
        handler.wfile.write(data)
    except Exception as exc:
        if _is_ignorable_client_disconnect(exc):
            return
        raise


def send_json(handler: Any, payload: object) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(HTTPStatus.OK)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    try:
        handler.wfile.write(body)
    except Exception as exc:
        if _is_ignorable_client_disconnect(exc):
            return
        raise


def send_json_error(handler: Any, status: HTTPStatus, message: str) -> None:
    body = json.dumps({"error": message}).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    try:
        handler.wfile.write(body)
    except Exception as exc:
        if _is_ignorable_client_disconnect(exc):
            return
        raise
