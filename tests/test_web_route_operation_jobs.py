"""Tests for asynchronous score, comparison, and file-operation routes."""
from __future__ import annotations

import json
import threading
import time
from http import HTTPStatus
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from shotsieve.db import database, initialize_database
from shotsieve.job_registry import JobRegistry
from shotsieve.web import build_handler

from conftest import create_image, find_free_port


def test_operation_job_launcher_owns_shared_lifecycle(monkeypatch, tmp_path: Path):
    from shotsieve import web_routes as route_module
    from shotsieve.web_route_jobs import _start_operation_job

    registry = JobRegistry()
    operation_lock = threading.Lock()
    captured: dict[str, object] = {}
    events: list[object] = []

    class SynchronousThread:
        def __init__(self, target):
            self.target = target

        def start(self):
            self.target()

    monkeypatch.setattr(
        route_module,
        "try_acquire_operation_lock",
        lambda _handler, context: context.operation_lock.acquire(blocking=False),
    )
    monkeypatch.setattr(
        route_module,
        "send_json",
        lambda _handler, payload: captured.setdefault("start", payload),
    )

    deps = SimpleNamespace(
        thread_factory=lambda *, target, daemon: SynchronousThread(target),
    )
    context = route_module.WebRouteContext(
        db_path=tmp_path / "shotsieve.db",
        operation_lock=operation_lock,
        scan_registry=None,
        score_registry=None,
        compare_registry=None,
        max_request_body_size=1024,
        static_dir=tmp_path,
        media_mime_fallbacks={},
        dependencies=deps,
        operation_registry=registry,
    )

    _start_operation_job(
        SimpleNamespace(),
        context,
        registry=registry,
        initial_progress={"phase": "starting"},
        progress_payload=lambda phase: {"phase": phase},
        worker=lambda publish, cancel_check: (
            events.append("worker"),
            publish("running"),
            cancel_check(),
            {"done": True},
        )[-1],
        result_payload=lambda result: {"result": result},
        cancel_error=lambda: InterruptedError("cancelled"),
    )

    job_id = captured["start"]["job_id"]
    status = registry.status(job_id)
    assert events == ["worker"]
    assert captured["start"]["status"] == "running"
    assert not operation_lock.locked()
    assert status is not None
    assert status["status"] == "completed"
    assert status["progress"] == {"phase": "running"}
    assert status["summary"] == {"result": {"done": True}}


class TestRouteHandlingAsync:
    def test_score_start_fails_fast_when_learned_iqa_runtime_missing(self, tmp_path: Path, monkeypatch):
        from http.server import ThreadingHTTPServer
        from shotsieve import web as web_module

        def fake_available_learned_backends(*, resource_profile=None):
            return {
                "pyiqa": "not-installed",
                "pyiqa_error": "No module named 'pyiqa'",
            }

        monkeypatch.setattr(web_module, "available_learned_backends", fake_available_learned_backends)

        db_path = tmp_path / "data" / "shotsieve.db"
        initialize_database(db_path)
        port = find_free_port()
        server = ThreadingHTTPServer(("127.0.0.1", port), build_handler(db_path))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            start_req = Request(
                f"http://127.0.0.1:{port}/api/score/start",
                data=json.dumps({"root": None, "learned_backend_name": "topiq_nr", "device": "cuda"}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with pytest.raises(HTTPError) as exc_info:
                urlopen(start_req)

            assert exc_info.value.code == HTTPStatus.BAD_REQUEST
            body = exc_info.value.read().decode("utf-8")
            assert "learned-iqa" in body
        finally:
            server.shutdown()

    def test_compare_start_fails_fast_when_learned_iqa_runtime_missing(self, tmp_path: Path, monkeypatch):
        from http.server import ThreadingHTTPServer
        from shotsieve import web as web_module

        def fake_available_learned_backends(*, resource_profile=None):
            return {
                "pyiqa": "not-installed",
                "pyiqa_error": "No module named 'pyiqa'",
            }

        monkeypatch.setattr(web_module, "available_learned_backends", fake_available_learned_backends)

        db_path = tmp_path / "data" / "shotsieve.db"
        initialize_database(db_path)
        port = find_free_port()
        server = ThreadingHTTPServer(("127.0.0.1", port), build_handler(db_path))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            start_req = Request(
                f"http://127.0.0.1:{port}/api/compare-models/start",
                data=json.dumps({"models": ["topiq_nr", "clipiqa"], "root": None, "device": "cuda"}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with pytest.raises(HTTPError) as exc_info:
                urlopen(start_req)

            assert exc_info.value.code == HTTPStatus.BAD_REQUEST
            body = exc_info.value.read().decode("utf-8")
            assert "learned-iqa" in body
        finally:
            server.shutdown()

    def test_delete_async_status_and_result_routes(self, tmp_path: Path, monkeypatch):
        from http.server import ThreadingHTTPServer
        from shotsieve import web as web_module

        started_event = threading.Event()
        release_event = threading.Event()

        def fake_delete_files(_connection, **kwargs):
            progress_callback = kwargs.get("progress_callback")
            cancel_check = kwargs.get("cancel_check")
            file_ids = list(kwargs.get("file_ids") or [])
            total = len(file_ids)
            if progress_callback is not None:
                progress_callback(0, total)
                progress_callback(1, total)
            started_event.set()
            release_event.wait(timeout=2)
            if callable(cancel_check):
                cancel_check()
            if progress_callback is not None:
                progress_callback(total, total)
            return {
                "deleted_ids": file_ids,
                "deleted_count": total,
                "failed": [],
                "failed_count": 0,
                "delete_from_disk": kwargs.get("delete_from_disk", False),
            }

        monkeypatch.setattr(web_module, "delete_files", fake_delete_files)

        db_path = tmp_path / "data" / "shotsieve.db"
        photo_dir = tmp_path / "photos"
        photo_dir.mkdir()
        create_image(photo_dir / "sample-0.jpg")
        create_image(photo_dir / "sample-1.jpg")
        initialize_database(db_path)

        with database(db_path) as connection:
            from shotsieve.scanner import scan_root

            scan_root(
                connection,
                root=photo_dir,
                recursive=True,
                extensions=(".jpg",),
                preview_dir=tmp_path / "previews",
            )
            file_ids = [row["id"] for row in connection.execute("SELECT id FROM files ORDER BY id ASC").fetchall()]
            current_rev = web_module.review_selection_revision(connection, scope="review-browser", marked="all")

        port = find_free_port()
        server = ThreadingHTTPServer(("127.0.0.1", port), build_handler(db_path))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            start_req = Request(
                f"http://127.0.0.1:{port}/api/files/delete/start",
                data=json.dumps({
                    "file_ids": file_ids,
                    "delete_from_disk": True,
                    "count": len(file_ids),
                    "selection_revision": current_rev,
                    "page_selection": {"scope": "review-browser", "marked": "all"},
                }).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            start_response = urlopen(start_req)
            start_payload = json.loads(start_response.read().decode("utf-8"))
            job_id = start_payload["job_id"]

            assert started_event.wait(timeout=2)

            status_response = urlopen(f"http://127.0.0.1:{port}/api/operations/status?job_id={job_id}")
            status_payload = json.loads(status_response.read().decode("utf-8"))
            assert status_payload["status"] in {"running", "completed"}
            assert status_payload["progress"]["phase"] == "deleting_files"
            assert status_payload["progress"]["files_processed"] >= 1
            assert status_payload["progress"]["files_total"] == 2

            release_event.set()

            completed_payload = None
            deadline = time.time() + 2
            while time.time() < deadline:
                status_response = urlopen(f"http://127.0.0.1:{port}/api/operations/status?job_id={job_id}")
                polled = json.loads(status_response.read().decode("utf-8"))
                if polled["status"] == "completed":
                    completed_payload = polled
                    break
                time.sleep(0.05)

            assert completed_payload is not None
            assert completed_payload["summary"]["deleted_count"] == 2

            result_response = urlopen(f"http://127.0.0.1:{port}/api/operations/result?job_id={job_id}")
            result_payload = json.loads(result_response.read().decode("utf-8"))
            assert result_payload["deleted_count"] == 2
            assert result_payload["deleted_ids"] == file_ids
        finally:
            release_event.set()
            server.shutdown()

    def test_cache_clear_async_status_and_result_routes(self, tmp_path: Path, monkeypatch):
        from http.server import ThreadingHTTPServer
        from shotsieve import web as web_module

        started_event = threading.Event()
        release_event = threading.Event()

        def fake_clear_cache_scope(_connection, **kwargs):
            progress_callback = kwargs.get("progress_callback")
            if progress_callback is not None:
                progress_callback(0, 1, "clearing_cache")
            started_event.set()
            release_event.wait(timeout=2)
            if progress_callback is not None:
                progress_callback(1, 1, "clearing_cache")
            return {"files": 3, "scores": 4, "review": 2, "scan_runs": 1}

        monkeypatch.setattr(web_module, "clear_cache_scope", fake_clear_cache_scope)

        db_path = tmp_path / "data" / "shotsieve.db"
        initialize_database(db_path)

        port = find_free_port()
        server = ThreadingHTTPServer(("127.0.0.1", port), build_handler(db_path))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            start_req = Request(
                f"http://127.0.0.1:{port}/api/cache/clear/start",
                data=json.dumps({"scope": "all"}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            start_response = urlopen(start_req)
            start_payload = json.loads(start_response.read().decode("utf-8"))
            job_id = start_payload["job_id"]

            assert started_event.wait(timeout=2)

            status_response = urlopen(f"http://127.0.0.1:{port}/api/operations/status?job_id={job_id}")
            status_payload = json.loads(status_response.read().decode("utf-8"))
            assert status_payload["status"] in {"running", "completed"}
            assert status_payload["progress"]["phase"] == "clearing_cache"

            release_event.set()

            completed_payload = None
            deadline = time.time() + 2
            while time.time() < deadline:
                status_response = urlopen(f"http://127.0.0.1:{port}/api/operations/status?job_id={job_id}")
                polled = json.loads(status_response.read().decode("utf-8"))
                if polled["status"] == "completed":
                    completed_payload = polled
                    break
                time.sleep(0.05)

            assert completed_payload is not None
            assert completed_payload["summary"]["files"] == 3

            result_response = urlopen(f"http://127.0.0.1:{port}/api/operations/result?job_id={job_id}")
            result_payload = json.loads(result_response.read().decode("utf-8"))
            assert result_payload == {"files": 3, "scores": 4, "review": 2, "scan_runs": 1}
        finally:
            release_event.set()
            server.shutdown()

    def test_cache_clear_async_start_rejects_missing_scope(self, tmp_path: Path):
        from http.server import ThreadingHTTPServer

        db_path = tmp_path / "data" / "shotsieve.db"
        initialize_database(db_path)

        port = find_free_port()
        server = ThreadingHTTPServer(("127.0.0.1", port), build_handler(db_path))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            request = Request(
                f"http://127.0.0.1:{port}/api/cache/clear/start",
                data=json.dumps({"scope": "missing"}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with pytest.raises(HTTPError) as exc_info:
                urlopen(request)
        finally:
            server.shutdown()

        assert exc_info.value.code == HTTPStatus.BAD_REQUEST
