"""Tests for web route handling: review batch, file actions, and request validation."""
from __future__ import annotations

import io
import json
import socket
import threading
import time
from types import SimpleNamespace
from http import HTTPStatus
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import pytest

from shotsieve.web import build_review_server

from conftest import read_socket_response


def _captured_list(store: dict[str, object], key: str) -> list[object]:
    existing = store.get(key)
    if isinstance(existing, list):
        return existing
    bucket: list[object] = []
    store[key] = bucket
    return bucket


def _captured_events(store: dict[str, object]) -> list[tuple[str, object]]:
    existing = store.get("events")
    if isinstance(existing, list):
        return existing  # type: ignore[return-value]
    bucket: list[tuple[str, object]] = []
    store["events"] = bucket
    return bucket


def test_route_shape_helpers_normalize_scan_compare_and_selection_payloads(tmp_path: Path) -> None:
    from shotsieve import web_request as request_module
    from shotsieve import web_routes as route_module

    photo_root = tmp_path / "photos"
    photo_root.mkdir()

    scan_request = request_module.parse_scan_request({
        "roots": [str(photo_root)],
        "offset": 2,
        "files_total_hint": 7,
        "preview_mode": "high-quality",
    })
    compare_request = request_module.parse_compare_request(
        {"models": [" topiq_nr ", "clipiqa"], "offset": 1},
        default_batch_size=8,
    )

    assert route_module._scan_request_roots(scan_request) == [photo_root.resolve()]
    assert route_module._scan_request_offset(scan_request) == 2
    assert route_module._scan_request_total_hint(scan_request) == 7
    assert scan_request["preview_mode"] == "high-quality"
    assert route_module._compare_request_models(compare_request) == ["topiq_nr", "clipiqa"]
    assert route_module._selection_excluded_ids({"exclude_file_ids": [1, 2]}) == {1, 2}


def test_route_result_helpers_normalize_delete_and_export_payloads() -> None:
    from shotsieve import web_routes as route_module

    delete_result = route_module._delete_result_payload({
        "deleted_ids": [11, 12],
        "deleted_count": 2,
        "failed": [],
        "failed_count": 0,
        "delete_from_disk": True,
    })
    export_result = route_module._export_result_payload(
        SimpleNamespace(copied=3, moved=1, failed=["oops"])
    )

    assert delete_result == {
        "deleted_ids": [11, 12],
        "deleted_count": 2,
        "failed": [],
        "failed_count": 0,
        "delete_from_disk": True,
    }
    assert export_result == {"copied": 3, "moved": 1, "failed": ["oops"]}


def test_reveal_in_file_manager_uses_windows_explorer_select(monkeypatch, tmp_path: Path) -> None:
    from shotsieve import web as web_module

    target_path = tmp_path / "photos" / "sample.nef"
    target_path.parent.mkdir(parents=True)
    target_path.write_bytes(b"raw")

    captured: dict[str, object] = {}

    monkeypatch.setattr(web_module.sys, "platform", "win32")
    monkeypatch.setattr(web_module.shutil, "which", lambda name: "C:/Windows/explorer.exe" if name.startswith("explorer") else None)

    def fake_popen(args, **_kwargs):
        captured["args"] = list(args)
        return SimpleNamespace()

    monkeypatch.setattr(web_module.subprocess, "Popen", fake_popen)

    method = web_module.reveal_in_file_manager(target_path)

    assert method == "windows-explorer"
    assert captured["args"] == ["C:/Windows/explorer.exe", f"/select,{target_path.resolve()}"]


class TestRouteHandling:
    def test_files_open_route_reveals_source_path_in_file_manager(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        from shotsieve import web_routes as route_module

        captured: dict[str, object] = {}
        source_path = (tmp_path / "photos" / "sample.nef").resolve()
        source_path.parent.mkdir(parents=True)
        source_path.write_bytes(b"raw")

        def fake_send_json(_handler, payload: object) -> None:
            captured["payload"] = payload

        monkeypatch.setattr(route_module, "send_json", fake_send_json)
        monkeypatch.setattr(
            route_module,
            "resolve_media_request",
            lambda **_kwargs: SimpleNamespace(path=source_path, error_status=None, error_message=None),
        )

        def fake_reveal_in_file_manager(path: Path) -> str:
            captured["revealed"] = path
            return "windows-explorer"

        deps = SimpleNamespace(
            read_json_body=lambda _handler, *, max_body_size: {"file_id": 7},
            required_int=lambda value, *, name, minimum=0: int(value),
            database=lambda _path: None,
            build_config=lambda *_args, **_kwargs: None,
            is_within_any_root=lambda *_args, **_kwargs: True,
            media_path_for_file=lambda *_args, **_kwargs: source_path,
            stable_preview_name=lambda _path: "preview-name",
            preview_name_candidates=lambda _path: ["preview-name"],
            guess_media_type=lambda _name: (None, None),
            reveal_in_file_manager=fake_reveal_in_file_manager,
        )
        context = route_module.WebRouteContext(
            db_path=tmp_path / "shotsieve.db",
            operation_lock=threading.Lock(),
            scan_registry=None,
            score_registry=None,
            compare_registry=None,
            max_request_body_size=1024,
            static_dir=tmp_path,
            media_mime_fallbacks={},
            dependencies=deps,
        )
        handler = SimpleNamespace(path="/api/files/open", headers={"Content-Length": "20"})

        handled = route_module._handle_file_action_post_routes(handler, context, urlparse(handler.path))

        assert handled is True
        assert captured["revealed"] == source_path
        assert captured["payload"] == {
            "opened": True,
            "path": str(source_path),
            "method": "windows-explorer",
        }

    def test_handle_job_result_keeps_completed_summary_after_socket_write_failure(self):
        from shotsieve.job_registry import JobRegistry
        from shotsieve import web_routes as route_module

        registry = JobRegistry(max_jobs=10)
        job_id = registry.create(initial_progress={"files_processed": 0, "files_total": 1})
        summary = {"files_scored": 1, "files_failed": 0}
        registry.complete(job_id, summary=summary)

        deps = SimpleNamespace(
            first_value=lambda params, key, default=None: (params.get(key) or [default])[0],
        )

        class BrokenPipeWriter:
            def write(self, _body: bytes) -> int:
                raise BrokenPipeError()

        class FakeHandler:
            def __init__(self, *, path: str, wfile):
                self.path = path
                self.wfile = wfile
                self.headers = {}
                self._shotsieve_route_dependencies = deps
                self.status_codes: list[int] = []
                self.sent_headers: list[tuple[str, str]] = []
                self.error_calls: list[tuple[int, str]] = []

            def send_response(self, status_code: int) -> None:
                self.status_codes.append(status_code)

            def send_header(self, name: str, value: str) -> None:
                self.sent_headers.append((name, value))

            def end_headers(self) -> None:
                return None

            def send_error(self, status_code: int, message: str) -> None:
                self.error_calls.append((status_code, message))

        broken_handler = FakeHandler(
            path=f"/api/score/result?job_id={job_id}",
            wfile=BrokenPipeWriter(),
        )

        route_module.handle_job_result(broken_handler, registry, label="Score")

        status_payload = registry.status(job_id)
        assert broken_handler.status_codes == [HTTPStatus.OK]
        assert broken_handler.error_calls == []
        assert status_payload is not None
        assert status_payload["status"] == "completed"
        assert status_payload["summary"] == summary

        retry_body = io.BytesIO()
        retry_handler = FakeHandler(
            path=f"/api/score/result?job_id={job_id}",
            wfile=retry_body,
        )

        route_module.handle_job_result(retry_handler, registry, label="Score")

        assert retry_handler.error_calls == []
        assert json.loads(retry_body.getvalue().decode("utf-8")) == summary

    def test_post_route_times_out_partial_request_body(self, tmp_path: Path):
        db_path = tmp_path / "data" / "shotsieve.db"
        server = build_review_server(
            db_path,
            host="127.0.0.1",
            port=0,
            request_read_timeout_seconds=0.2,
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        try:
            with socket.create_connection(("127.0.0.1", server.server_port), timeout=2) as client:
                partial_request = (
                    f"POST /api/cache/clear HTTP/1.1\r\n"
                    f"Host: 127.0.0.1:{server.server_port}\r\n"
                    f"Origin: http://127.0.0.1:{server.server_port}\r\n"
                    f"Content-Type: application/json\r\n"
                    f"Content-Length: 18\r\n"
                    f"Connection: close\r\n\r\n"
                    "{\"scope\":"
                ).encode("utf-8")
                client.sendall(partial_request)

                response = read_socket_response(client)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

        assert b"408 Request Timeout" in response
        assert b"Request body read timed out" in response

    def test_post_route_allows_brief_body_gaps_below_deadline(self, tmp_path: Path):
        db_path = tmp_path / "data" / "shotsieve.db"
        server = build_review_server(
            db_path,
            host="127.0.0.1",
            port=0,
            request_read_timeout_seconds=1.0,
            request_io_poll_timeout_seconds=0.1,
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        try:
            with socket.create_connection(("127.0.0.1", server.server_port), timeout=2) as client:
                client.sendall(
                    (
                        f"POST /api/cache/clear HTTP/1.1\r\n"
                        f"Host: 127.0.0.1:{server.server_port}\r\n"
                        f"Origin: http://127.0.0.1:{server.server_port}\r\n"
                        f"Content-Type: application/json\r\n"
                        f"Content-Length: 19\r\n"
                        f"Connection: close\r\n\r\n"
                        "{\"scope\":"
                    ).encode("utf-8")
                )
                time.sleep(0.35)
                client.sendall(b'"missing"}')

                response = read_socket_response(client)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

        assert b"200 OK" in response
        assert b'"files":' in response

    def test_server_returns_408_for_partial_request_headers(self, tmp_path: Path):
        db_path = tmp_path / "data" / "shotsieve.db"
        server = build_review_server(
            db_path,
            host="127.0.0.1",
            port=0,
            request_header_read_timeout_seconds=0.2,
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        try:
            with socket.create_connection(("127.0.0.1", server.server_port), timeout=2) as client:
                client.sendall(b"POST /api/cache/clear HTTP/1.1\r\nHost: 127.0.0.1")
                response = read_socket_response(client)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

        assert b"408 Request Timeout" in response
        assert b"Request headers read timed out" in response

    def test_review_server_rejects_new_requests_when_worker_slots_are_full(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        from shotsieve import web_request as request_module

        entered_read = threading.Event()
        original_read_json_body = request_module.read_json_body

        def instrumented_read_json_body(handler, *, max_body_size: int):
            entered_read.set()
            return original_read_json_body(handler, max_body_size=max_body_size)

        monkeypatch.setattr(request_module, "read_json_body", instrumented_read_json_body)

        db_path = tmp_path / "data" / "shotsieve.db"
        server = build_review_server(
            db_path,
            host="127.0.0.1",
            port=0,
            max_concurrent_requests=1,
            request_read_timeout_seconds=5.0,
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        blocking_client: socket.socket | None = None

        try:
            blocking_client = socket.create_connection(("127.0.0.1", server.server_port), timeout=2)
            blocking_client.sendall(
                (
                    f"POST /api/cache/clear HTTP/1.1\r\n"
                    f"Host: 127.0.0.1:{server.server_port}\r\n"
                    f"Origin: http://127.0.0.1:{server.server_port}\r\n"
                    f"Content-Type: application/json\r\n"
                    f"Content-Length: 18\r\n"
                    f"Connection: close\r\n\r\n"
                    "{\"scope\":"
                ).encode("utf-8")
            )

            assert entered_read.wait(timeout=2)

            req = Request(
                f"http://127.0.0.1:{server.server_port}/api/overview",
                headers={"Origin": f"http://127.0.0.1:{server.server_port}"},
            )
            # On Windows the server may RST the TCP connection before the
            # client reads the HTTP 503 response, producing a
            # ConnectionAbortedError rather than an HTTPError.  Both signals
            # mean the request was rejected, which is what we test.
            with pytest.raises((HTTPError, ConnectionError)) as exc_info:
                urlopen(req, timeout=2)
        finally:
            if blocking_client is not None:
                blocking_client.close()
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

        if isinstance(exc_info.value, HTTPError):
            assert exc_info.value.code == HTTPStatus.SERVICE_UNAVAILABLE

    def test_review_batch_route_accepts_review_browser_selection_payload(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        from shotsieve import web_routes as route_module

        captured: dict[str, object] = {}
        connection = object()

        class _DatabaseContext:
            def __enter__(self):
                return connection

            def __exit__(self, exc_type, exc, tb):
                return False

        def fake_send_json(_handler, payload: object) -> None:
            captured["payload"] = payload

        batches = [[101, 102], [103], []]

        def fake_list_review_browser_file_ids(_connection, **kwargs):
            _captured_list(captured, "selection_calls").append(dict(kwargs))
            return batches.pop(0)

        def fake_update_review_state_batch(_connection, **kwargs):
            _captured_list(captured, "updated_batches").append(list(kwargs["file_ids"]))
            return len(kwargs["file_ids"])

        monkeypatch.setattr(route_module, "send_json", fake_send_json)

        deps = SimpleNamespace(
            read_json_body=lambda _handler, *, max_body_size: {
                "selection": {
                    "scope": "review-browser",
                    "root": "C:/photos",
                    "marked": "delete",
                    "issues": "issues",
                    "query": "cats",
                    "min_score": 10,
                    "max_score": 90,
                },
                "selection_revision": "rev-1",
                "decision_state": "delete",
                "delete_marked": True,
                "export_marked": False,
            },
            required_int_list=lambda value, *, name: (_ for _ in ()).throw(AssertionError("file_ids should not be required for filter selections")),
            required_choice=lambda value, *, name, choices: value if value in choices else (_ for _ in ()).throw(ValueError(name)),
            optional_string=lambda value: value if isinstance(value, str) else None,
            optional_bool=lambda value, *, name: value if isinstance(value, bool) else None,
            float_or_none=lambda value: None if value is None else float(value),
            database=lambda _path: _DatabaseContext(),
            review_selection_revision=lambda _connection, **kwargs: "rev-1",
            list_review_browser_file_ids=fake_list_review_browser_file_ids,
            update_review_state_batch=fake_update_review_state_batch,
            utc_now=lambda: "2026-04-22T00:00:00+00:00",
        )
        context = route_module.WebRouteContext(
            db_path=tmp_path / "shotsieve.db",
            operation_lock=threading.Lock(),
            scan_registry=None,
            score_registry=None,
            compare_registry=None,
            max_request_body_size=1024,
            static_dir=tmp_path,
            media_mime_fallbacks={},
            dependencies=deps,
        )
        handler = SimpleNamespace(path="/api/review/batch", headers={"Content-Length": "20"})

        handled = route_module._handle_review_post_routes(handler, context, urlparse(handler.path))

        assert handled is True
        assert captured["updated_batches"] == [[101, 102], [103]]
        assert captured["selection_calls"] == [
            {
                "root": "C:/photos",
                "marked": "delete",
                "issues": "issues",
                "query": "cats",
                "min_score": 10.0,
                "max_score": 90.0,
                "limit": 500,
                "after_id": 0,
            },
            {
                "root": "C:/photos",
                "marked": "delete",
                "issues": "issues",
                "query": "cats",
                "min_score": 10.0,
                "max_score": 90.0,
                "limit": 500,
                "after_id": 102,
            },
            {
                "root": "C:/photos",
                "marked": "delete",
                "issues": "issues",
                "query": "cats",
                "min_score": 10.0,
                "max_score": 90.0,
                "limit": 500,
                "after_id": 103,
            },
        ]
        assert captured["payload"] == {"updated": 3}

    def test_review_batch_route_excludes_unchecked_ids_from_selection_payload(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        from shotsieve import web_routes as route_module

        captured: dict[str, object] = {}
        connection = object()

        class _DatabaseContext:
            def __enter__(self):
                return connection

            def __exit__(self, exc_type, exc, tb):
                return False

        def fake_send_json(_handler, payload: object) -> None:
            captured["payload"] = payload

        batches = [[101, 102], [103], []]

        def fake_list_review_browser_file_ids(_connection, **kwargs):
            return batches.pop(0)

        def fake_update_review_state_batch(_connection, **kwargs):
            _captured_list(captured, "updated_batches").append(list(kwargs["file_ids"]))
            return len(kwargs["file_ids"])

        monkeypatch.setattr(route_module, "send_json", fake_send_json)

        deps = SimpleNamespace(
            read_json_body=lambda _handler, *, max_body_size: {
                "selection": {
                    "scope": "review-browser",
                    "marked": "all",
                },
                "selection_revision": "rev-1",
                "exclude_file_ids": [102],
                "decision_state": "export",
                "delete_marked": False,
                "export_marked": True,
            },
            required_int_list=lambda value, *, name: [int(item) for item in value],
            required_choice=lambda value, *, name, choices: value if value in choices else (_ for _ in ()).throw(ValueError(name)),
            optional_string=lambda value: value if isinstance(value, str) else None,
            optional_bool=lambda value, *, name: value if isinstance(value, bool) else None,
            float_or_none=lambda value: None if value is None else float(value),
            database=lambda _path: _DatabaseContext(),
            review_selection_revision=lambda _connection, **kwargs: "rev-1",
            list_review_browser_file_ids=fake_list_review_browser_file_ids,
            update_review_state_batch=fake_update_review_state_batch,
            utc_now=lambda: "2026-04-22T00:00:00+00:00",
        )
        context = route_module.WebRouteContext(
            db_path=tmp_path / "shotsieve.db",
            operation_lock=threading.Lock(),
            scan_registry=None,
            score_registry=None,
            compare_registry=None,
            max_request_body_size=1024,
            static_dir=tmp_path,
            media_mime_fallbacks={},
            dependencies=deps,
        )
        handler = SimpleNamespace(path="/api/review/batch", headers={"Content-Length": "20"})

        handled = route_module._handle_review_post_routes(handler, context, urlparse(handler.path))

        assert handled is True
        assert captured["updated_batches"] == [[101], [103]]
        assert captured["payload"] == {"updated": 2}

    def test_review_batch_route_materializes_selection_before_mutation(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        from shotsieve import web_routes as route_module

        captured: dict[str, object] = {"events": []}

        class _Connection:
            def execute(self, sql: str):
                _captured_events(captured).append(("execute", sql))
                return None

            def commit(self):
                _captured_events(captured).append(("commit", None))

            def rollback(self):
                _captured_events(captured).append(("rollback", None))

        connection = _Connection()

        class _DatabaseContext:
            def __enter__(self):
                return connection

            def __exit__(self, exc_type, exc, tb):
                return False

        batches = [[101, 102], [103], []]

        def fake_send_json(_handler, payload: object) -> None:
            captured["payload"] = payload

        def fake_list_review_browser_file_ids(_connection, **kwargs):
            _captured_events(captured).append(("list", dict(kwargs)))
            return batches.pop(0)

        def fake_update_review_state_batch(_connection, **kwargs):
            _captured_events(captured).append(("update", list(kwargs["file_ids"])))
            return len(kwargs["file_ids"])

        monkeypatch.setattr(route_module, "send_json", fake_send_json)

        deps = SimpleNamespace(
            read_json_body=lambda _handler, *, max_body_size: {
                "selection": {
                    "scope": "review-browser",
                    "marked": "all",
                },
                "selection_revision": "rev-1",
                "decision_state": "export",
                "delete_marked": False,
                "export_marked": True,
            },
            required_int_list=lambda value, *, name: [int(item) for item in value],
            required_choice=lambda value, *, name, choices: value if value in choices else (_ for _ in ()).throw(ValueError(name)),
            optional_string=lambda value: value if isinstance(value, str) else None,
            optional_bool=lambda value, *, name: value if isinstance(value, bool) else None,
            float_or_none=lambda value: None if value is None else float(value),
            database=lambda _path: _DatabaseContext(),
            review_selection_revision=lambda _connection, **kwargs: "rev-1",
            list_review_browser_file_ids=fake_list_review_browser_file_ids,
            update_review_state_batch=fake_update_review_state_batch,
            utc_now=lambda: "2026-04-22T00:00:00+00:00",
        )
        context = route_module.WebRouteContext(
            db_path=tmp_path / "shotsieve.db",
            operation_lock=threading.Lock(),
            scan_registry=None,
            score_registry=None,
            compare_registry=None,
            max_request_body_size=1024,
            static_dir=tmp_path,
            media_mime_fallbacks={},
            dependencies=deps,
        )
        handler = SimpleNamespace(path="/api/review/batch", headers={"Content-Length": "20"})

        handled = route_module._handle_review_post_routes(handler, context, urlparse(handler.path))

        assert handled is True
        assert captured["payload"] == {"updated": 3}
        events = _captured_events(captured)
        list_indices = [index for index, event in enumerate(events) if event[0] == "list"]
        update_indices = [index for index, event in enumerate(events) if event[0] == "update"]
        assert events[0] == ("execute", "BEGIN")
        assert events[-1] == ("commit", None)
        assert list_indices
        assert update_indices
        assert max(list_indices) < min(update_indices)

    def test_files_delete_route_accepts_review_state_selection_payload(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        from shotsieve import web_routes as route_module

        captured: dict[str, object] = {}
        connection = object()
        preview_root = (tmp_path / "previews").resolve()

        class _DatabaseContext:
            def __enter__(self):
                return connection

            def __exit__(self, exc_type, exc, tb):
                return False

        def fake_send_json(_handler, payload: object) -> None:
            captured["payload"] = payload

        batches = [[11, 12], [13], []]

        def fake_list_review_state_file_ids(_connection, **kwargs):
            _captured_list(captured, "selection_calls").append(dict(kwargs))
            return batches.pop(0)

        def fake_delete_files(_connection, **kwargs):
            _captured_list(captured, "deleted_batches").append(list(kwargs["file_ids"]))
            ids = list(kwargs["file_ids"])
            return {
                "deleted_ids": ids,
                "deleted_count": len(ids),
                "failed": [],
                "failed_count": 0,
                "delete_from_disk": kwargs["delete_from_disk"],
            }

        monkeypatch.setattr(route_module, "send_json", fake_send_json)

        deps = SimpleNamespace(
            read_json_body=lambda _handler, *, max_body_size: {
                "selection": {
                    "scope": "review-state",
                    "marked": "delete",
                },
                "selection_revision": "rev-1",
                "delete_from_disk": True,
            },
            required_int_list=lambda value, *, name: (_ for _ in ()).throw(AssertionError("file_ids should not be required for filter selections")),
            required_choice=lambda value, *, name, choices: value if value in choices else (_ for _ in ()).throw(ValueError(name)),
            optional_string=lambda value: value if isinstance(value, str) else None,
            coerce_bool=lambda value, *, default: default if value is None else bool(value),
            database=lambda _path: _DatabaseContext(),
            review_selection_revision=lambda _connection, **kwargs: "rev-1",
            list_review_state_file_ids=fake_list_review_state_file_ids,
            get_preview_cache_root=lambda _connection, *, db_path, persist: preview_root,
            delete_files=fake_delete_files,
        )
        context = route_module.WebRouteContext(
            db_path=tmp_path / "shotsieve.db",
            operation_lock=threading.Lock(),
            scan_registry=None,
            score_registry=None,
            compare_registry=None,
            max_request_body_size=1024,
            static_dir=tmp_path,
            media_mime_fallbacks={},
            dependencies=deps,
        )
        handler = SimpleNamespace(path="/api/files/delete", headers={"Content-Length": "20"})

        handled = route_module._handle_file_action_post_routes(handler, context, urlparse(handler.path))

        assert handled is True
        assert captured["deleted_batches"] == [[11, 12], [13]]
        assert captured["payload"] == {
            "deleted_ids": [11, 12, 13],
            "deleted_count": 3,
            "failed": [],
            "failed_count": 0,
            "delete_from_disk": True,
        }

    def test_files_export_route_accepts_review_browser_selection_payload(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        from shotsieve import web_routes as route_module

        captured: dict[str, object] = {}
        connection = object()
        preview_root = (tmp_path / "previews").resolve()

        class _DatabaseContext:
            def __enter__(self):
                return connection

            def __exit__(self, exc_type, exc, tb):
                return False

        def fake_send_json(_handler, payload: object) -> None:
            captured["payload"] = payload

        batches = [[21, 22], [23], []]

        def fake_list_review_browser_file_ids(_connection, **kwargs):
            _captured_list(captured, "selection_calls").append(dict(kwargs))
            return batches.pop(0)

        def fake_export_files(_connection, **kwargs):
            _captured_list(captured, "export_batches").append(list(kwargs["file_ids"]))
            ids = list(kwargs["file_ids"])
            return SimpleNamespace(copied=len(ids), moved=0, failed=[])

        monkeypatch.setattr(route_module, "send_json", fake_send_json)

        deps = SimpleNamespace(
            read_json_body=lambda _handler, *, max_body_size: {
                "selection": {
                    "scope": "review-browser",
                    "marked": "all",
                    "root": "C:/photos",
                    "query": "keepers",
                },
                "selection_revision": "rev-1",
                "destination": str(tmp_path / "export"),
                "mode": "copy",
            },
            required_int_list=lambda value, *, name: (_ for _ in ()).throw(AssertionError("file_ids should not be required for filter selections")),
            required_choice=lambda value, *, name, choices: value if value in choices else (_ for _ in ()).throw(ValueError(name)),
            optional_string=lambda value: value if isinstance(value, str) else None,
            float_or_none=lambda value: None if value is None else float(value),
            database=lambda _path: _DatabaseContext(),
            review_selection_revision=lambda _connection, **kwargs: "rev-1",
            list_review_browser_file_ids=fake_list_review_browser_file_ids,
            get_preview_cache_root=lambda _connection, *, db_path, persist: preview_root,
            export_files=fake_export_files,
        )
        context = route_module.WebRouteContext(
            db_path=tmp_path / "shotsieve.db",
            operation_lock=threading.Lock(),
            scan_registry=None,
            score_registry=None,
            compare_registry=None,
            max_request_body_size=1024,
            static_dir=tmp_path,
            media_mime_fallbacks={},
            dependencies=deps,
        )
        handler = SimpleNamespace(path="/api/files/export", headers={"Content-Length": "20"})

        handled = route_module._handle_file_action_post_routes(handler, context, urlparse(handler.path))

        assert handled is True
        assert captured["export_batches"] == [[21, 22], [23]]
        assert captured["payload"] == {"copied": 3, "moved": 0, "failed": []}
