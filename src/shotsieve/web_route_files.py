from __future__ import annotations

import sys
from http import HTTPStatus
from pathlib import Path
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
from shotsieve.models import (
    FileOperationResult,
    FileOperationSummary,
    attach_file_operation_summary,
    operation_summary_from_exception,
)


def _get_web_routes() -> Any:
    return sys.modules["shotsieve.web_routes"]


def _handle_filesystem_get_routes(handler: Any, context: WebRouteContext, parsed: Any) -> bool:
    deps = cast(WebRouteDependencies, context.dependencies)
    routes = _get_web_routes()
    if parsed.path == "/api/cache/missing/preview":
        params = parse_qs(parsed.query)
        root = _required_missing_cleanup_root(deps, deps.first_value(params, "root", None))
        with deps.database(context.db_path) as connection:
            routes.send_json(
                handler,
                deps.preview_missing_cache_entries(connection, root=root),
            )
        return True

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
                try:
                    progress_callback(0, total, "deleting_files")
                except InterruptedError as exc:
                    stopped = FileOperationSummary(action="delete", delete_from_disk=delete_from_disk)
                    _append_unprocessed_operation_rows(connection, stopped, file_ids, action="delete", error=exc)
                    attach_file_operation_summary(exc, stopped, cancelled=True)
                    raise
            return routes._delete_result_payload(deps.delete_files(
                connection,
                file_ids=file_ids,
                delete_from_disk=delete_from_disk,
                preview_cache_root=preview_cache_root,
                progress_callback=routes._operation_progress_callback(progress_callback, phase="deleting_files", offset=0, total_hint=total),
                cancel_check=cancel_check,
            ))

        snapshot_active = _begin_consistent_snapshot(connection)
        operation_summary = FileOperationSummary(action="delete", delete_from_disk=delete_from_disk)
        delete_result: DeleteResultPayload = {
            "deleted_ids": [],
            "deleted_count": 0,
            "failed": [],
            "failed_count": 0,
            "delete_from_disk": delete_from_disk,
        }
        contract_enabled = False
        batches = []
        current_batch_index = 0
        batch_completed = False
        try:
            routes._validate_selection_revision(connection, deps, selection)
            # Materializing the already-filtered IDs before the first mutation
            # freezes the selection and lets a stopped job report later rows as
            # not attempted.
            batches = list(routes._frozen_selection_batches(connection, deps, selection))
            processed_so_far = 0
            if progress_callback is not None:
                progress_callback(0, total_hint or 0, "deleting_files")
            for batch_index, file_ids in enumerate(batches):
                current_batch_index = batch_index
                batch_completed = False
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
                contract_enabled = contract_enabled or "items" in batch_result
                delete_result["deleted_ids"].extend(batch_result["deleted_ids"])
                delete_result["deleted_count"] += int(batch_result["deleted_count"])
                delete_result["failed"].extend(batch_result["failed"])
                delete_result["failed_count"] += int(batch_result["failed_count"])
                if contract_enabled:
                    operation_summary.merge_payload(batch_result)
                processed_so_far += len(file_ids)
                batch_completed = True
                if progress_callback is not None:
                    progress_callback(processed_so_far, total_hint or processed_so_far, "deleting_files")
        except Exception as exc:
            partial_payload = operation_summary_from_exception(exc)
            if batches:
                if partial_payload is not None:
                    operation_summary.merge_payload(partial_payload)
                contract_enabled = True
                first_remaining_batch = current_batch_index + 1 if batch_completed else current_batch_index
                _append_unprocessed_operation_rows(
                    connection,
                    operation_summary,
                    [
                        file_id
                        for batch in batches[first_remaining_batch:]
                        for file_id in batch
                    ],
                    action="delete",
                    error=exc,
                )
                operation_summary.fatal_error = str(exc)
                needs_rollback = bool(getattr(exc, "file_operation_needs_rollback", False))
                attach_file_operation_summary(
                    exc,
                    operation_summary,
                    cancelled=bool(getattr(exc, "file_operation_cancelled", False)),
                    needs_rollback=needs_rollback,
                )
                _finish_consistent_snapshot(
                    connection,
                    active=snapshot_active,
                    success=not needs_rollback,
                )
            else:
                _finish_consistent_snapshot(connection, active=snapshot_active, success=False)
            raise
        else:
            _finish_consistent_snapshot(connection, active=snapshot_active, success=True)

    if contract_enabled:
        operation_summary.delete_from_disk = delete_from_disk
        return cast(DeleteResultPayload, operation_summary.to_dict())
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
                try:
                    progress_callback(0, total, phase)
                except InterruptedError as exc:
                    stopped = FileOperationSummary(action=mode)
                    _append_unprocessed_operation_rows(connection, stopped, file_ids, action=mode, error=exc)
                    attach_file_operation_summary(exc, stopped, cancelled=True)
                    raise
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
        operation_summary = FileOperationSummary(action="export")
        contract_enabled = False
        batches = []
        processed_so_far = 0
        current_batch_index = 0
        batch_completed = False
        try:
            routes._validate_selection_revision(connection, deps, selection)
            batches = list(routes._frozen_selection_batches(connection, deps, selection))
            if progress_callback is not None:
                progress_callback(0, total_hint or 0, phase)
            for batch_index, file_ids in enumerate(batches):
                current_batch_index = batch_index
                batch_completed = False
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
                if isinstance(batch_result, FileOperationSummary):
                    contract_enabled = True
                    operation_summary.merge(batch_result)
                else:
                    raw_items = getattr(batch_result, "items", None)
                    if isinstance(raw_items, list):
                        contract_enabled = True
                        operation_summary.merge_payload(routes._export_result_payload(batch_result))
                    else:
                        operation_summary.copied += int(getattr(batch_result, "copied", 0) or 0)
                        operation_summary.moved += int(getattr(batch_result, "moved", 0) or 0)
                processed_so_far += len(file_ids)
                batch_completed = True
                if progress_callback is not None:
                    progress_callback(processed_so_far, total_hint or processed_so_far, phase)
        except Exception as exc:
            partial_payload = operation_summary_from_exception(exc)
            if batches:
                if partial_payload is not None:
                    operation_summary.merge_payload(partial_payload)
                contract_enabled = True
                first_remaining_batch = current_batch_index + 1 if batch_completed else current_batch_index
                _append_unprocessed_operation_rows(
                    connection,
                    operation_summary,
                    [
                        file_id
                        for batch in batches[first_remaining_batch:]
                        for file_id in batch
                    ],
                    action=mode,
                    error=exc,
                )
                operation_summary.fatal_error = str(exc)
                needs_rollback = bool(getattr(exc, "file_operation_needs_rollback", False))
                attach_file_operation_summary(
                    exc,
                    operation_summary,
                    cancelled=bool(getattr(exc, "file_operation_cancelled", False)),
                    needs_rollback=needs_rollback,
                )
                _finish_consistent_snapshot(
                    connection,
                    active=snapshot_active,
                    success=not needs_rollback,
                )
            else:
                _finish_consistent_snapshot(connection, active=snapshot_active, success=False)
            raise
        else:
            _finish_consistent_snapshot(connection, active=snapshot_active, success=True)

    if contract_enabled:
        operation_summary.contract_enabled = True
        return operation_summary
    return ExportAggregate(copied=operation_summary.copied, moved=operation_summary.moved)


def _append_unprocessed_operation_rows(
    connection: Any,
    summary: FileOperationSummary,
    file_ids: list[int],
    *,
    action: str,
    error: BaseException,
) -> None:
    if not file_ids:
        return
    rows_by_id: dict[int, object] = {}
    execute = getattr(connection, "execute", None)
    if callable(execute):
        try:
            placeholders = ",".join("?" for _ in file_ids)
            rows = execute(
                f"SELECT id, path FROM files WHERE id IN ({placeholders})",
                tuple(file_ids),
            ).fetchall()
            rows_by_id = {int(row["id"]): row for row in rows}
        except Exception:
            rows_by_id = {}
    accounted_ids = {item.file_id for item in summary.items}
    for file_id in file_ids:
        if int(file_id) in accounted_ids:
            continue
        row = rows_by_id.get(int(file_id))
        source = str(row["path"]) if row is not None else ""
        summary.add(
            FileOperationResult(
                file_id=int(file_id),
                source=source,
                destination=None,
                action=action,
                outcome="unprocessed",
                stage="not_started",
                error_text=str(error),
                retry_safe=True,
            )
        )
        accounted_ids.add(int(file_id))


def _execute_cache_clear_request(
    context: WebRouteContext,
    payload: dict[str, object],
    *,
    progress_callback: Callable[[int, int, str], None] | None,
    cancel_check: Callable[[], None] | None,
) -> dict[str, int]:
    deps = cast(WebRouteDependencies, context.dependencies)
    scope = deps.required_choice(payload.get("scope"), name="scope", choices=("scores", "review", "all"))
    with deps.database(context.db_path) as connection:
        preview_cache_root = deps.get_preview_cache_root(connection, db_path=context.db_path, persist=False)
        return deps.clear_cache_scope(
            connection,
            scope=scope,
            preview_cache_root=preview_cache_root,
            progress_callback=progress_callback,
            cancel_check=cancel_check,
        )


def _required_missing_cleanup_root(deps: WebRouteDependencies, raw_root: object) -> Path:
    root_text = deps.optional_string(raw_root)
    if not root_text or not root_text.strip():
        raise ValueError("root is required for missing-entry cleanup")
    if "|" in root_text:
        raise ValueError("missing-entry cleanup accepts one root at a time")
    try:
        return Path(root_text.strip()).expanduser().resolve()
    except OSError as exc:
        raise ValueError(f"Invalid cleanup root: {exc}") from exc


def _execute_missing_cache_apply_request(
    context: WebRouteContext,
    payload: dict[str, object],
) -> dict[str, object]:
    deps = cast(WebRouteDependencies, context.dependencies)
    root = _required_missing_cleanup_root(deps, payload.get("root"))
    token = deps.optional_string(payload.get("token"))
    if not token:
        raise ValueError("token is required for missing-entry cleanup")
    candidate_ids = deps.required_int_list(payload.get("candidate_ids"), name="candidate_ids")

    with deps.database(context.db_path) as connection:
        preview_cache_root = deps.get_preview_cache_root(
            connection,
            db_path=context.db_path,
            persist=False,
        )
        return deps.apply_missing_cache_entries(
            connection,
            root=root,
            token=token,
            candidate_ids=candidate_ids,
            preview_cache_root=preview_cache_root,
        )
