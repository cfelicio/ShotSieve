from __future__ import annotations

import sys
from http import HTTPStatus
from typing import Any, Callable, cast
from urllib.parse import parse_qs

from shotsieve.web_media import MediaDependencies, serve_media_response
from shotsieve.web_route_common import (
    DeleteResultPayload,
    ExportAggregate,
    WebRouteContext,
    WebRouteDependencies,
    _begin_consistent_snapshot,
    _finish_consistent_snapshot,
)


def _get_web_routes() -> Any:
    return sys.modules["shotsieve.web_routes"]


def _handle_filesystem_get_routes(handler: Any, context: WebRouteContext, parsed: Any) -> bool:
    deps = cast(WebRouteDependencies, context.dependencies)
    routes = _get_web_routes()
    if parsed.path == "/api/fs/roots":
        routes.send_json(handler, {"items": deps.filesystem_roots()})
        return True

    if parsed.path == "/api/fs/list":
        params = parse_qs(parsed.query)
        directory = deps.required_path(deps.first_value(params, "path", None), name="path")
        routes.send_json(handler, deps.list_directory(directory))
        return True

    return False


def _handle_media_get_routes(handler: Any, context: WebRouteContext, parsed: Any) -> bool:
    if parsed.path not in {"/api/media/preview", "/api/media/source"}:
        return False

    deps = cast(WebRouteDependencies, context.dependencies)
    routes = _get_web_routes()
    params = parse_qs(parsed.query)
    file_id = deps.required_int(deps.first_value(params, "id", None), name="id", minimum=1)
    variant = "preview" if parsed.path.endswith("preview") else "source"
    media_result = routes.resolve_media_request(
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


def _handle_file_action_post_routes(handler: Any, context: WebRouteContext, parsed: Any) -> bool:
    deps = cast(WebRouteDependencies, context.dependencies)
    routes = _get_web_routes()
    if parsed.path == "/api/files/open":
        payload = deps.read_json_body(handler, max_body_size=context.max_request_body_size)
        file_id = deps.required_int(payload.get("file_id"), name="file_id", minimum=1)
        media_result = routes.resolve_media_request(
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
        routes.send_json(handler, {"opened": True, "path": str(media_result.path), "method": method})
        return True

    if parsed.path == "/api/files/delete":
        payload = deps.read_json_body(handler, max_body_size=context.max_request_body_size)
        delete_result = routes._execute_delete_request(context, payload, progress_callback=None, cancel_check=None)
        routes.send_json(handler, delete_result)
        return True

    if parsed.path == "/api/files/delete/start":
        payload = deps.read_json_body(handler, max_body_size=context.max_request_body_size)
        routes.start_delete_job(handler, context, payload)
        return True

    if parsed.path == "/api/files/export":
        payload = deps.read_json_body(handler, max_body_size=context.max_request_body_size)
        export_result = routes._execute_export_request(context, payload, progress_callback=None, cancel_check=None)
        routes.send_json(
            handler,
            routes._export_result_payload(export_result),
        )
        return True

    if parsed.path == "/api/files/export/start":
        payload = deps.read_json_body(handler, max_body_size=context.max_request_body_size)
        routes.start_export_job(handler, context, payload)
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
    routes = _get_web_routes()
    selection = routes._parse_selection_payload(deps, payload)
    if selection is not None:
        routes._require_root_for_destructive_selection(selection)
    delete_from_disk = deps.coerce_bool(payload.get("delete_from_disk"), default=False)
    total_hint = routes._progress_total_hint(deps, payload)

    with deps.database(context.db_path) as connection:
        preview_cache_root = deps.get_preview_cache_root(connection, db_path=context.db_path, persist=False)
        if selection is None:
            file_ids = deps.required_int_list(payload.get("file_ids"), name="file_ids")
            routes._validate_page_revision(connection, deps, payload)
            total = total_hint if total_hint is not None else len(file_ids)
            if progress_callback is not None:
                progress_callback(0, total, "deleting_files")
            return routes._delete_result_payload(deps.delete_files(
                connection,
                file_ids=file_ids,
                delete_from_disk=delete_from_disk,
                preview_cache_root=preview_cache_root,
                progress_callback=routes._operation_progress_callback(progress_callback, phase="deleting_files", offset=0, total_hint=total),
                cancel_check=cancel_check,
            ))

        snapshot_active = _begin_consistent_snapshot(connection)
        try:
            routes._validate_selection_revision(connection, deps, selection)
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
            for file_ids in routes._frozen_selection_batches(connection, deps, selection):
                if cancel_check is not None:
                    cancel_check()
                batch_total = total_hint if total_hint is not None else processed_so_far + len(file_ids)
                batch_result = routes._delete_result_payload(deps.delete_files(
                    connection,
                    file_ids=file_ids,
                    delete_from_disk=delete_from_disk,
                    preview_cache_root=preview_cache_root,
                    progress_callback=routes._operation_progress_callback(progress_callback, phase="deleting_files", offset=processed_so_far, total_hint=batch_total),
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
    routes = _get_web_routes()
    selection = routes._parse_selection_payload(deps, payload)
    if selection is not None:
        routes._require_root_for_destructive_selection(selection)
    destination = deps.optional_string(payload.get("destination"))
    mode_raw = payload.get("mode")
    mode = (
        "copy"
        if mode_raw is None
        else deps.required_choice(mode_raw, name="mode", choices=("copy", "move"))
    )
    phase = "moving_files" if mode == "move" else "exporting_files"
    total_hint = routes._progress_total_hint(deps, payload)
    if not destination:
        raise ValueError("destination is required")

    with deps.database(context.db_path) as connection:
        preview_cache_root = deps.get_preview_cache_root(connection, db_path=context.db_path, persist=False)
        if selection is None:
            file_ids = deps.required_int_list(payload.get("file_ids"), name="file_ids")
            routes._validate_page_revision(connection, deps, payload)
            total = total_hint if total_hint is not None else len(file_ids)
            if progress_callback is not None:
                progress_callback(0, total, phase)
            return deps.export_files(
                connection,
                file_ids=file_ids,
                destination=destination,
                mode=mode,
                preview_cache_root=preview_cache_root,
                progress_callback=routes._operation_progress_callback(progress_callback, phase=phase, offset=0, total_hint=total),
                cancel_check=cancel_check,
            )

        snapshot_active = _begin_consistent_snapshot(connection)
        try:
            routes._validate_selection_revision(connection, deps, selection)
            copied = 0
            moved = 0
            failed: list[object] = []
            processed_so_far = 0
            if progress_callback is not None:
                progress_callback(0, total_hint or 0, phase)
            for file_ids in routes._frozen_selection_batches(connection, deps, selection):
                if cancel_check is not None:
                    cancel_check()
                batch_total = total_hint if total_hint is not None else processed_so_far + len(file_ids)
                batch_result = deps.export_files(
                    connection,
                    file_ids=file_ids,
                    destination=destination,
                    mode=mode,
                    preview_cache_root=preview_cache_root,
                    progress_callback=routes._operation_progress_callback(progress_callback, phase=phase, offset=processed_so_far, total_hint=batch_total),
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
