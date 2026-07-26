from __future__ import annotations

import json
import socket
import sqlite3
import sys
from dataclasses import dataclass
from http import HTTPStatus
from pathlib import Path
from typing import Any, Callable, TypedDict

from shotsieve.job_registry import JobRegistry
from shotsieve.web_request import CompareRequest, ScanRequest, try_parse_http_status

_SELECTION_BATCH_SIZE = 500


def _get_web_routes() -> Any:
    return sys.modules["shotsieve.web_routes"]


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
    required_string_list: Callable[..., list[str]]
    required_path: Callable[..., Path]
    required_path_list: Callable[..., list[Path]]
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
    list_analysis_diagnostics: Callable[..., dict[str, object]]
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


def _optional_payload_float(deps: WebRouteDependencies, value: object, *, name: str) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return deps.float_or_none(value)
    raise ValueError(f"{name} must be numeric")


def _optional_payload_int(deps: WebRouteDependencies, value: object, *, name: str) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    raise ValueError(f"{name} must be an integer")


def _optional_payload_string_list(deps: WebRouteDependencies, value: object, *, name: str) -> list[str] | None:
    if value is None or value == "":
        return None
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    raise ValueError(f"{name} must be a list of strings")


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
    if scope == "review-state" and not selection["root"]:
        raise ValueError("selection.root is required for review-state bulk actions")
    if scope == "review-browser":
        selection["issues"] = deps.required_choice(
            raw_selection.get("issues") or "all",
            name="selection.issues",
            choices=("all", "issues"),
        )
        selection["min_score"] = _optional_payload_float(deps, raw_selection.get("min_score"), name="selection.min_score")
        selection["max_score"] = _optional_payload_float(deps, raw_selection.get("max_score"), name="selection.max_score")
        selection["formats"] = _optional_payload_string_list(deps, raw_selection.get("formats"), name="selection.formats")
        selection["min_mp"] = _optional_payload_float(deps, raw_selection.get("min_mp"), name="selection.min_mp")
        selection["max_mp"] = _optional_payload_float(deps, raw_selection.get("max_mp"), name="selection.max_mp")
        selection["min_width"] = _optional_payload_int(deps, raw_selection.get("min_width"), name="selection.min_width")
        selection["max_width"] = _optional_payload_int(deps, raw_selection.get("max_width"), name="selection.max_width")
        selection["min_height"] = _optional_payload_int(deps, raw_selection.get("min_height"), name="selection.min_height")
        selection["max_height"] = _optional_payload_int(deps, raw_selection.get("max_height"), name="selection.max_height")
        selection["min_size"] = _optional_payload_int(deps, raw_selection.get("min_size"), name="selection.min_size")
        selection["max_size"] = _optional_payload_int(deps, raw_selection.get("max_size"), name="selection.max_size")
        selection["metadata"] = deps.required_choice(
            raw_selection.get("metadata") or "all",
            name="selection.metadata",
            choices=("all", "valid", "unknown"),
        )

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
        formats=selection.get("formats"),
        min_mp=selection.get("min_mp"),
        max_mp=selection.get("max_mp"),
        min_width=selection.get("min_width"),
        max_width=selection.get("max_width"),
        min_height=selection.get("min_height"),
        max_height=selection.get("max_height"),
        min_size=selection.get("min_size"),
        max_size=selection.get("max_size"),
        metadata=selection.get("metadata", "all"),
    )
    if selection_revision != current_revision:
        raise ValueError("Selected results changed. Refresh the queue and select again.")


def _validate_page_revision(connection: Any, deps: WebRouteDependencies, payload: dict[str, object]) -> None:
    page_revision = deps.optional_string(payload.get("selection_revision"))
    page_selection = payload.get("page_selection")
    if not page_revision:
        raise ValueError("selection_revision is required for file_ids operations")
    if not page_selection or not isinstance(page_selection, dict):
        raise ValueError("page_selection is required for file_ids operations")
    current = deps.review_selection_revision(
        connection,
        scope=page_selection.get("scope", "review-browser"),
        root=deps.optional_string(page_selection.get("root")),
        marked=page_selection.get("marked", "all"),
        issues=page_selection.get("issues", "all"),
        query=deps.optional_string(page_selection.get("query")),
        min_score=_optional_payload_float(deps, page_selection.get("min_score"), name="page_selection.min_score"),
        max_score=_optional_payload_float(deps, page_selection.get("max_score"), name="page_selection.max_score"),
        formats=_optional_payload_string_list(deps, page_selection.get("formats"), name="page_selection.formats"),
        min_mp=_optional_payload_float(deps, page_selection.get("min_mp"), name="page_selection.min_mp"),
        max_mp=_optional_payload_float(deps, page_selection.get("max_mp"), name="page_selection.max_mp"),
        min_width=_optional_payload_int(deps, page_selection.get("min_width"), name="page_selection.min_width"),
        max_width=_optional_payload_int(deps, page_selection.get("max_width"), name="page_selection.max_width"),
        min_height=_optional_payload_int(deps, page_selection.get("min_height"), name="page_selection.min_height"),
        max_height=_optional_payload_int(deps, page_selection.get("max_height"), name="page_selection.max_height"),
        min_size=_optional_payload_int(deps, page_selection.get("min_size"), name="page_selection.min_size"),
        max_size=_optional_payload_int(deps, page_selection.get("max_size"), name="page_selection.max_size"),
        metadata=page_selection.get("metadata", "all"),
    )
    if page_revision != current:
        raise ValueError("Selected results changed. Refresh the queue and select again.")


def _require_root_for_destructive_selection(selection: dict[str, object]) -> None:
    if not selection.get("root"):
        raise ValueError("selection.root is required for destructive bulk operations")


def _materialize_selection_batches(connection: Any, deps: WebRouteDependencies, selection: dict[str, object]) -> list[list[int]]:
    batches: list[list[int]] = []
    for batch in _get_web_routes()._iter_selection_file_id_batches(connection, deps, selection):
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
        for batch in _get_web_routes()._iter_selection_file_id_batches(connection, deps, selection):
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
        yield from _get_web_routes()._iter_sqlite_materialized_selection_batches(connection, deps, selection)
        return
    yield from _get_web_routes()._materialize_selection_batches(connection, deps, selection)


def _iter_selection_file_id_batches(connection: Any, deps: WebRouteDependencies, selection: dict[str, object]):
    after_id = 0
    excluded_ids = _get_web_routes()._selection_excluded_ids(selection)
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
                formats=selection.get("formats"),
                min_mp=selection.get("min_mp"),
                max_mp=selection.get("max_mp"),
                min_width=selection.get("min_width"),
                max_width=selection.get("max_width"),
                min_height=selection.get("min_height"),
                max_height=selection.get("max_height"),
                min_size=selection.get("min_size"),
                max_size=selection.get("max_size"),
                metadata=selection.get("metadata", "all"),
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
