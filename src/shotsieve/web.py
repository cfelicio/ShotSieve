from __future__ import annotations

import logging
import mimetypes
import os
import shutil
import socket
import sqlite3
import subprocess
import sys
import threading
import webbrowser
from string import ascii_uppercase
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import cast

from shotsieve.config import DEFAULT_RAW_PREVIEW_MODE, DEFAULT_SUPPORTED_EXTENSIONS, RAW_PREVIEW_MODES, build_config
from shotsieve.db import database, get_preview_cache_root, initialize_database
from shotsieve.export import export_files
from shotsieve.job_registry import JobRegistry
from shotsieve.learned_iqa import (
    DEFAULT_BATCH_SIZE,
    LearnedBackendUnavailableError,
    available_learned_backends,
    runtime_curated_learned_models,
)
from shotsieve.preview import MIN_RAW_THUMBNAIL_LONG_EDGE, preview_capabilities, preview_name_candidates, stable_preview_name
from shotsieve.review import (
    clear_cache_scope,
    count_review_files,
    delete_files,
    get_review_file_detail,
    list_review_browser_file_ids,
    list_review_files,
    list_review_state_file_ids,
    media_path_for_file,
    prune_missing_cache_entries,
    review_selection_revision,
    review_overview,
    update_review_state,
    update_review_state_batch,
)
from shotsieve.scanner import scan_root, utc_now
from shotsieve.scoring import AnalysisProgress  # noqa: F401
from shotsieve.scoring import compare_learned_models, count_score_rows, score_files
from shotsieve import web_request as _request_helpers
from shotsieve.web_routes import (
    WebRouteContext,
    WebRouteDependencies,
    handle_get,
    handle_post,
    log_request_message,
    send_json_error,
)
from shotsieve import web_security as _security_helpers

_coerce_bool = _request_helpers.coerce_bool
_first = _request_helpers.first_value
_float_or_none = _request_helpers.float_or_none
_int_or_default = _request_helpers.int_or_default
_optional_bool = _request_helpers.optional_bool
_optional_int = _request_helpers.optional_int
_optional_string = _request_helpers.optional_string
_required_choice = _request_helpers.required_choice
_required_int = _request_helpers.required_int
_required_int_list = _request_helpers.required_int_list
_required_path = _request_helpers.required_path
_required_path_list = _request_helpers.required_path_list

_is_allowed_post_origin = _security_helpers.is_allowed_post_origin
_is_loopback_host = _security_helpers.is_loopback_host
_is_within_any_root = _security_helpers.is_within_any_root

log = logging.getLogger(__name__)

_DEFAULT_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
}
_DEFAULT_REQUEST_HEADER_READ_TIMEOUT_SECONDS = 2.0
_DEFAULT_REQUEST_BODY_READ_TIMEOUT_SECONDS = 5.0
_DEFAULT_REQUEST_IO_POLL_TIMEOUT_SECONDS = 0.25
_DEFAULT_RESPONSE_WRITE_TIMEOUT_SECONDS = 5.0
_DEFAULT_MAX_CONCURRENT_REQUESTS = 8


def _server_busy_response() -> bytes:
    body = b'{"error":"Server is busy. Please retry in a moment."}'
    headers = [
        b"HTTP/1.1 503 Service Unavailable",
        b"Connection: close",
        b"Content-Type: application/json; charset=utf-8",
        f"Content-Length: {len(body)}".encode("ascii"),
        b"Retry-After: 1",
        b"X-Content-Type-Options: nosniff",
        b"X-Frame-Options: DENY",
        b"",
        b"",
    ]
    return b"\r\n".join(headers) + body


class BoundedReviewHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 16

    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[BaseHTTPRequestHandler],
        *,
        max_concurrent_requests: int = _DEFAULT_MAX_CONCURRENT_REQUESTS,
        request_header_read_timeout_seconds: float = _DEFAULT_REQUEST_HEADER_READ_TIMEOUT_SECONDS,
        request_read_timeout_seconds: float = _DEFAULT_REQUEST_BODY_READ_TIMEOUT_SECONDS,
        request_io_poll_timeout_seconds: float = _DEFAULT_REQUEST_IO_POLL_TIMEOUT_SECONDS,
        response_write_timeout_seconds: float = _DEFAULT_RESPONSE_WRITE_TIMEOUT_SECONDS,
    ) -> None:
        super().__init__(server_address, handler_class)
        self.request_header_read_timeout_seconds = max(0.1, float(request_header_read_timeout_seconds))
        self.request_read_timeout_seconds = max(0.1, float(request_read_timeout_seconds))
        self.request_body_read_timeout_seconds = self.request_read_timeout_seconds
        self.request_io_poll_timeout_seconds = max(0.05, float(request_io_poll_timeout_seconds))
        self.response_write_timeout_seconds = max(0.1, float(response_write_timeout_seconds))
        self._request_slots = threading.BoundedSemaphore(max(1, int(max_concurrent_requests)))

    def process_request(self, request, client_address) -> None:
        request_socket = cast(socket.socket, request)
        request_socket.settimeout(self.request_io_poll_timeout_seconds)
        if not self._request_slots.acquire(blocking=False):
            try:
                request_socket.sendall(_server_busy_response())
                request_socket.shutdown(socket.SHUT_WR)
            except OSError:
                pass
            self.close_request(request_socket)
            return

        try:
            super().process_request(request_socket, client_address)
        except Exception:
            self._request_slots.release()
            raise

    def process_request_thread(self, request, client_address) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._request_slots.release()


def _learned_install_guidance(*, preferred_device: str | None) -> str:
    normalized_device = (preferred_device or "").strip().casefold()
    if normalized_device == "directml":
        return (
            "Install shotsieve[learned-iqa-directml] for DirectML runtimes "
            "(or shotsieve[learned-iqa] for CPU/CUDA runtimes)."
        )
    return "Install shotsieve[learned-iqa]."


def _require_learned_runtime(*, resource_profile: str | None, preferred_device: str | None) -> None:
    guidance = _learned_install_guidance(preferred_device=preferred_device)
    learned = available_learned_backends(resource_profile=resource_profile)
    status = str(learned.get("pyiqa") or "").strip().casefold()
    if status == "installed":
        return

    detail = learned.get("pyiqa_error")
    detail_text = detail.strip() if isinstance(detail, str) else ""
    message = f"Learned IQA dependencies are unavailable. {guidance}"
    if detail_text:
        message = f"{message} Runtime detail: {detail_text}"
    raise LearnedBackendUnavailableError(message)


def resolve_static_dir() -> Path:
    module_static = Path(__file__).with_name("static")
    if module_static.exists():
        return module_static

    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            for candidate in (
                Path(meipass) / "shotsieve" / "static",
                Path(meipass) / "static",
            ):
                if candidate.exists():
                    return candidate

    return module_static


STATIC_DIR = resolve_static_dir()
_MEDIA_MIME_FALLBACKS = {
    ".webp": "image/webp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".heic": "image/heic",
    ".heif": "image/heif",
}


def serve_review_ui(
    *,
    db_path: Path,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
) -> None:
    server = build_review_server(db_path, host=host, port=port)
    url = f"http://{host}:{port}"

    if open_browser:
        webbrowser.open(url)

    print(f"Review UI available at {url}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def build_review_server(
    db_path: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    max_concurrent_requests: int = _DEFAULT_MAX_CONCURRENT_REQUESTS,
    request_header_read_timeout_seconds: float = _DEFAULT_REQUEST_HEADER_READ_TIMEOUT_SECONDS,
    request_read_timeout_seconds: float = _DEFAULT_REQUEST_BODY_READ_TIMEOUT_SECONDS,
    request_io_poll_timeout_seconds: float = _DEFAULT_REQUEST_IO_POLL_TIMEOUT_SECONDS,
    response_write_timeout_seconds: float = _DEFAULT_RESPONSE_WRITE_TIMEOUT_SECONDS,
) -> BoundedReviewHTTPServer:
    initialize_database(db_path)
    handler = build_handler(db_path)
    return BoundedReviewHTTPServer(
        (host, port),
        handler,
        max_concurrent_requests=max_concurrent_requests,
        request_header_read_timeout_seconds=request_header_read_timeout_seconds,
        request_read_timeout_seconds=request_read_timeout_seconds,
        request_io_poll_timeout_seconds=request_io_poll_timeout_seconds,
        response_write_timeout_seconds=response_write_timeout_seconds,
    )


def build_handler(db_path: Path):
    operation_lock = threading.Lock()
    scan_registry = JobRegistry(max_jobs=10)
    score_registry = JobRegistry(max_jobs=10)
    compare_registry = JobRegistry(max_jobs=10)

    def route_scan_root(*args, **kwargs):
        return scan_root(*args, **kwargs)

    def route_score_files(*args, **kwargs):
        return score_files(*args, **kwargs)

    def route_compare_learned_models(*args, **kwargs):
        return compare_learned_models(*args, **kwargs)

    def route_require_learned_runtime(*args, **kwargs):
        return _require_learned_runtime(*args, **kwargs)

    def route_guess_media_type(name: str):
        return mimetypes.guess_type(name)

    def route_is_loopback_host(host: str | None) -> bool:
        return _is_loopback_host(host)

    def route_is_allowed_post_origin(origin: str | None, host_header: str | None) -> bool:
        return _is_allowed_post_origin(origin, host_header)

    def route_is_within_any_root(candidate: Path, roots: list[Path]) -> bool:
        return _is_within_any_root(candidate, roots)

    def route_read_json_body(handler: BaseHTTPRequestHandler, *, max_body_size: int) -> dict[str, object]:
        return _request_helpers.read_json_body(handler, max_body_size=max_body_size)

    def route_parse_scan_request(payload: dict[str, object]):
        return _request_helpers.parse_scan_request(payload)

    def route_parse_compare_request(
        payload: dict[str, object],
        *,
        default_batch_size: int,
    ):
        return _request_helpers.parse_compare_request(
            payload,
            default_batch_size=default_batch_size,
        )

    route_context = WebRouteContext(
        db_path=db_path,
        operation_lock=operation_lock,
        scan_registry=scan_registry,
        score_registry=score_registry,
        compare_registry=compare_registry,
        max_request_body_size=10 * 1024 * 1024,
        static_dir=STATIC_DIR,
        media_mime_fallbacks=_MEDIA_MIME_FALLBACKS,
        dependencies=WebRouteDependencies(
            coerce_bool=lambda value, *, default: _coerce_bool(value, default=default),
            first_value=lambda params, key, default=None: _first(params, key, default),
            float_or_none=lambda value: _float_or_none(value),
            int_or_default=lambda value, *, default, minimum=0, maximum=None: _int_or_default(
                value,
                default=default,
                minimum=minimum,
                maximum=maximum,
            ),
            optional_bool=lambda value, *, name: _optional_bool(value, name=name),
            optional_int=lambda value, minimum=0: _optional_int(value, minimum=minimum),
            optional_string=lambda value: _optional_string(value),
            required_choice=lambda value, *, name, choices: _required_choice(value, name=name, choices=choices),
            required_int=lambda value, *, name, minimum=0: _required_int(value, name=name, minimum=minimum),
            required_int_list=lambda value, *, name: _required_int_list(value, name=name),
            required_path=lambda value, *, name: _required_path(value, name=name),
            read_json_body=lambda handler, *, max_body_size: route_read_json_body(handler, max_body_size=max_body_size),
            parse_scan_request=lambda payload: route_parse_scan_request(payload),
            parse_compare_request=lambda payload, *, default_batch_size: route_parse_compare_request(
                payload,
                default_batch_size=default_batch_size,
            ),
            database=lambda path: database(path),
            build_options_payload=lambda path, *, resource_profile=None: build_options_payload(
                path,
                resource_profile=resource_profile,
            ),
            filesystem_roots=lambda: filesystem_roots(),
            list_directory=lambda path: list_directory(path),
            review_overview=lambda connection: review_overview(connection),
            list_review_files=lambda *args, **kwargs: list_review_files(*args, **kwargs),
            count_review_files=lambda *args, **kwargs: count_review_files(*args, **kwargs),
            review_selection_revision=lambda *args, **kwargs: review_selection_revision(*args, **kwargs),
            list_review_browser_file_ids=lambda *args, **kwargs: list_review_browser_file_ids(*args, **kwargs),
            list_review_state_file_ids=lambda *args, **kwargs: list_review_state_file_ids(*args, **kwargs),
            get_review_file_detail=lambda *args, **kwargs: get_review_file_detail(*args, **kwargs),
            update_review_state=lambda *args, **kwargs: update_review_state(*args, **kwargs),
            update_review_state_batch=lambda *args, **kwargs: update_review_state_batch(*args, **kwargs),
            media_path_for_file=lambda *args, **kwargs: media_path_for_file(*args, **kwargs),
            build_config=lambda *args, **kwargs: build_config(*args, **kwargs),
            is_within_any_root=route_is_within_any_root,
            stable_preview_name=lambda path: stable_preview_name(path),
            preview_name_candidates=lambda path: list(preview_name_candidates(path)),
            guess_media_type=route_guess_media_type,
            utc_now=lambda: utc_now(),
            scan_root=route_scan_root,
            score_files=route_score_files,
            compare_learned_models=route_compare_learned_models,
            require_learned_runtime=route_require_learned_runtime,
            get_preview_cache_root=lambda *args, **kwargs: get_preview_cache_root(*args, **kwargs),
            count_score_rows=lambda *args, **kwargs: count_score_rows(*args, **kwargs),
            clear_cache_scope=lambda *args, **kwargs: clear_cache_scope(*args, **kwargs),
            prune_missing_cache_entries=lambda *args, **kwargs: prune_missing_cache_entries(*args, **kwargs),
            reveal_in_file_manager=lambda path: reveal_in_file_manager(path),
            delete_files=lambda *args, **kwargs: delete_files(*args, **kwargs),
            export_files=lambda *args, **kwargs: export_files(*args, **kwargs),
            default_batch_size=lambda: DEFAULT_BATCH_SIZE,
            thread_factory=lambda *args, **kwargs: threading.Thread(*args, **kwargs),
        ),
    )

    class ReviewHandler(BaseHTTPRequestHandler):
        _shotsieve_route_dependencies = route_context.dependencies
        protocol_version = "HTTP/1.1"

        def setup(self) -> None:
            super().setup()
            _request_helpers.install_deadline_aware_reader(
                self,
                initial_timeout_message="Request headers read timed out",
            )
            _request_helpers.set_handler_read_deadline(
                self,
                seconds=getattr(self.server, "request_header_read_timeout_seconds", _DEFAULT_REQUEST_HEADER_READ_TIMEOUT_SECONDS),
                message="Request headers read timed out",
            )

        def _clear_read_deadline_for_response(self) -> None:
            _request_helpers.clear_handler_read_deadline(self)
            try:
                self.connection.settimeout(getattr(self.server, "response_write_timeout_seconds", _DEFAULT_RESPONSE_WRITE_TIMEOUT_SECONDS))
            except OSError:
                pass

        def handle_one_request(self) -> None:
            try:
                self.requestline = ""
                self.request_version = self.protocol_version
                self.command = ""
                try:
                    self.connection.settimeout(getattr(self.server, "request_io_poll_timeout_seconds", _DEFAULT_REQUEST_IO_POLL_TIMEOUT_SECONDS))
                except OSError:
                    pass
                _request_helpers.set_handler_read_deadline(
                    self,
                    seconds=getattr(self.server, "request_header_read_timeout_seconds", _DEFAULT_REQUEST_HEADER_READ_TIMEOUT_SECONDS),
                    message="Request headers read timed out",
                )
                self.raw_requestline = self.rfile.readline(65537)
                if len(self.raw_requestline) > 65536:
                    self.requestline = ""
                    self.request_version = ""
                    self.command = ""
                    self.send_error(HTTPStatus.REQUEST_URI_TOO_LONG)
                    return
                if not self.raw_requestline:
                    self.close_connection = True
                    return
                if not self.parse_request():
                    return
                method_name = f"do_{self.command}"
                if not hasattr(self, method_name):
                    self.send_error(HTTPStatus.NOT_IMPLEMENTED, f"Unsupported method ({self.command!r})")
                    return
                method = getattr(self, method_name)
                method()
                self.wfile.flush()
            except _request_helpers.RequestBodyTimeoutError as exc:
                self.log_error("Request timed out: %r", exc)
                self.close_connection = True
                self._clear_read_deadline_for_response()
                send_json_error(self, HTTPStatus.REQUEST_TIMEOUT, str(exc))
            except TimeoutError as exc:
                self.log_error("Request timed out: %r", exc)
                self.close_connection = True
                self._clear_read_deadline_for_response()

        def do_GET(self) -> None:
            self._clear_read_deadline_for_response()
            if _security_helpers.reject_non_local_client(
                self,
                is_loopback_host_func=route_is_loopback_host,
            ):
                return
            try:
                handle_get(self, route_context)
            except (ValueError, LearnedBackendUnavailableError) as exc:
                send_json_error(self, HTTPStatus.BAD_REQUEST, str(exc))
            except Exception as exc:
                send_json_error(self, HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

        def do_POST(self) -> None:
            self._clear_read_deadline_for_response()
            if _security_helpers.reject_non_local_client(
                self,
                is_loopback_host_func=route_is_loopback_host,
            ):
                return
            if _security_helpers.reject_disallowed_origin(
                self,
                send_json_error=lambda status, message: send_json_error(self, status, message),
                is_allowed_post_origin_func=route_is_allowed_post_origin,
            ):
                return
            try:
                handle_post(self, route_context)
            except _request_helpers.RequestBodyTimeoutError as exc:
                self.close_connection = True
                send_json_error(self, HTTPStatus.REQUEST_TIMEOUT, str(exc))
            except (ValueError, LearnedBackendUnavailableError) as exc:
                send_json_error(self, HTTPStatus.BAD_REQUEST, str(exc))
            except sqlite3.OperationalError as exc:
                message = str(exc)
                if "database is locked" in message.casefold():
                    send_json_error(
                        self,
                        HTTPStatus.CONFLICT,
                        "Database is busy with another operation. Please wait for it to finish and try again.",
                    )
                else:
                    send_json_error(self, HTTPStatus.INTERNAL_SERVER_ERROR, message)
            except Exception as exc:
                send_json_error(self, HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

        def log_message(self, format: str, *args) -> None:
            log_request_message(self, log, format, *args)

        def end_headers(self) -> None:
            self.close_connection = True
            self.send_header("Connection", "close")
            for name, value in _DEFAULT_SECURITY_HEADERS.items():
                self.send_header(name, value)
            super().end_headers()

    return ReviewHandler


def build_options_payload(db_path: Path, *, resource_profile: str | None = None) -> dict[str, object]:
    learned = available_learned_backends(resource_profile=resource_profile)
    capabilities = preview_capabilities()
    with database(db_path) as connection:
        preview_dir = get_preview_cache_root(connection, db_path=db_path, persist=False)
    runtime_targets_ui = ["auto", "cpu", "cuda", "xpu", "directml", "mps"]
    return {
        "database": str(db_path.resolve()),
        "preview_dir": str(preview_dir),
        "default_extensions": list(DEFAULT_SUPPORTED_EXTENSIONS),
        "default_preview_mode": DEFAULT_RAW_PREVIEW_MODE,
        "preview_modes": list(RAW_PREVIEW_MODES),
        "raw_preview_auto_min_long_edge": MIN_RAW_THUMBNAIL_LONG_EDGE,
        "learned": learned,
        "learned_models": list(runtime_curated_learned_models()),
        "default_scoring_mode": learned["default_model"],
        "runtime_targets": runtime_targets_ui,
        "default_batch_size": DEFAULT_BATCH_SIZE,
        "preview_capabilities": capabilities,
    }


def filesystem_roots() -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    if os.name == "nt":
        for letter in ascii_uppercase:
            root = Path(f"{letter}:\\")
            if root.exists():
                items.append({"name": str(root), "path": str(root)})
    else:
        items.append({"name": "/", "path": "/"})

    home = Path.home().resolve()
    home_value = str(home)
    if all(item["path"] != home_value for item in items):
        items.insert(0, {"name": f"Home ({home.name or home_value})", "path": home_value})
    return items


def list_directory(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    if not resolved.exists() or not resolved.is_dir():
        raise ValueError("path must point to an existing directory")

    try:
        children = [child for child in resolved.iterdir() if child.is_dir()]
    except OSError as exc:
        raise ValueError(str(exc)) from exc

    children.sort(key=lambda child: child.name.casefold())
    parent = resolved.parent if resolved.parent != resolved else None
    return {
        "path": str(resolved),
        "parent": str(parent) if parent is not None else None,
        "items": [{"name": child.name, "path": str(child)} for child in children],
    }


def reveal_in_file_manager(path: Path) -> str:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise ValueError(f"File does not exist: {resolved}")

    if sys.platform.startswith("win"):
        explorer_executable = shutil.which("explorer.exe") or shutil.which("explorer")
        if explorer_executable is None:
            windows_dir = Path(os.environ.get("WINDIR") or os.environ.get("SystemRoot") or r"C:\Windows")
            candidate = windows_dir / "explorer.exe"
            if candidate.exists():
                explorer_executable = str(candidate)
        if explorer_executable:
            subprocess.Popen([explorer_executable, f"/select,{resolved}"])
            return "windows-explorer"
        if hasattr(os, "startfile"):
            os.startfile(str(resolved.parent))
            return "windows-startfile"
        raise OSError("Windows Explorer is unavailable")

    if sys.platform == "darwin":
        subprocess.Popen(["open", "-R", str(resolved)])
        return "finder"

    opener = shutil.which("xdg-open")
    if opener:
        subprocess.Popen([opener, str(resolved.parent)])
        return "xdg-open"

    raise OSError("No supported file manager opener is available")
