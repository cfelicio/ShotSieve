"""Tests for asynchronous scan lifecycle, pagination, and cancellation routes."""
from __future__ import annotations

import json
import threading
import time
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from urllib.request import Request, urlopen

import pytest

from shotsieve.db import database, initialize_database
from shotsieve.web import build_handler

from conftest import create_image, find_free_port


def test_scan_job_request_snapshots_mutable_request_values(tmp_path: Path):
    from shotsieve.web_request import parse_scan_request
    from shotsieve.web_route_scan import _ScanJobRequest

    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    parsed = parse_scan_request({
        "roots": [str(first_root)],
        "ignore_rules": ["ignored"],
        "files_total_hint": 7,
    })
    request = _ScanJobRequest.from_scan_request(parsed)

    parsed["roots"].append(second_root)
    parsed["ignore_rules"].append("changed")

    assert request.roots == (first_root.resolve(),)
    assert request.ignore_rules == ("ignored",)
    assert request.files_total_hint == 7
    with pytest.raises(FrozenInstanceError):
        request.offset = 4


class TestRouteHandlingAsync:
    def test_scan_async_status_and_result_routes(self, tmp_path: Path, monkeypatch):
        from dataclasses import dataclass
        from http.server import ThreadingHTTPServer
        from shotsieve import web as web_module

        started_event = threading.Event()
        release_event = threading.Event()

        @dataclass
        class FakeSummary:
            files_seen: int = 3
            files_added: int = 2
            files_updated: int = 1
            files_unchanged: int = 0
            files_removed: int = 0
            files_failed: int = 0

        def fake_scan_root(*args, **kwargs):
            progress_callback = kwargs.get("progress_callback")
            total = int(kwargs.get("files_total_hint") or 3)
            if progress_callback:
                progress_callback(1, total, "scanning")
            started_event.set()
            release_event.wait(timeout=2)
            if progress_callback:
                progress_callback(total, total, "scanning")
            return FakeSummary()

        monkeypatch.setattr(web_module, "scan_root", fake_scan_root)

        db_path = tmp_path / "data" / "shotsieve.db"
        photo_dir = tmp_path / "photos"
        photo_dir.mkdir()
        create_image(photo_dir / "sample.jpg")
        initialize_database(db_path)

        port = find_free_port()
        server = ThreadingHTTPServer(("127.0.0.1", port), build_handler(db_path))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            start_req = Request(
                f"http://127.0.0.1:{port}/api/scan/start",
                data=json.dumps({"roots": [str(photo_dir)], "recursive": True, "generate_previews": False}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            start_response = urlopen(start_req)
            start_payload = json.loads(start_response.read().decode("utf-8"))
            job_id = start_payload["job_id"]

            assert started_event.wait(timeout=2)

            status_response = urlopen(f"http://127.0.0.1:{port}/api/scan/status?job_id={job_id}")
            status_payload = json.loads(status_response.read().decode("utf-8"))
            assert status_payload["status"] in {"running", "completed"}
            assert status_payload["progress"]["phase"] == "scanning"
            assert status_payload["progress"]["files_processed"] >= 1

            release_event.set()

            completed_payload = None
            deadline = time.time() + 2
            while time.time() < deadline:
                status_response = urlopen(f"http://127.0.0.1:{port}/api/scan/status?job_id={job_id}")
                polled = json.loads(status_response.read().decode("utf-8"))
                if polled["status"] == "completed":
                    completed_payload = polled
                    break
                time.sleep(0.05)

            assert completed_payload is not None
            assert completed_payload["summary"]["files_seen"] == 3
            assert completed_payload["summary"]["overall_status"] == "completed"
            assert completed_payload["summary"]["root_results"][0]["status"] == "completed"

            result_response = urlopen(f"http://127.0.0.1:{port}/api/scan/result?job_id={job_id}")
            result_payload = json.loads(result_response.read().decode("utf-8"))
            assert result_payload["files_seen"] == 3
            assert result_payload["files_added"] == 2
            assert result_payload["root_results"][0]["root_path"] == str(photo_dir.resolve())
        finally:
            release_event.set()
            server.shutdown()

    def test_scan_async_multi_root_commits_success_before_failed_root(self, tmp_path: Path, monkeypatch):
        from http.server import ThreadingHTTPServer
        from shotsieve import scanner as scanner_module
        from shotsieve.scanner import FileDiscoveryError

        db_path = tmp_path / "data" / "shotsieve.db"
        first_root = tmp_path / "first"
        failed_root = tmp_path / "unavailable"
        first_root.mkdir()
        failed_root.mkdir()
        create_image(first_root / "sample.jpg")
        initialize_database(db_path)

        original_discover_files = scanner_module.discover_files

        def fail_second_root(root, *args, **kwargs):
            if root == failed_root:
                raise FileDiscoveryError(root, PermissionError("second root unavailable"))
            return original_discover_files(root, *args, **kwargs)

        monkeypatch.setattr(scanner_module, "discover_files", fail_second_root)

        port = find_free_port()
        server = ThreadingHTTPServer(("127.0.0.1", port), build_handler(db_path))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            start_request = Request(
                f"http://127.0.0.1:{port}/api/scan/start",
                data=json.dumps({
                    "roots": [str(first_root), str(failed_root)],
                    "recursive": True,
                    "generate_previews": False,
                }).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            job_id = json.loads(urlopen(start_request).read().decode("utf-8"))["job_id"]

            failed_payload = None
            deadline = time.time() + 3
            while time.time() < deadline:
                status = json.loads(
                    urlopen(f"http://127.0.0.1:{port}/api/scan/status?job_id={job_id}").read().decode("utf-8")
                )
                if status["status"] == "failed":
                    failed_payload = status
                    break
                time.sleep(0.05)

            assert failed_payload is not None
            assert failed_payload["summary"]["overall_status"] == "failed"
            assert [item["status"] for item in failed_payload["summary"]["root_results"]] == [
                "completed",
                "failed",
            ]
            assert failed_payload["summary"]["root_results"][1]["root_path"] == str(failed_root.resolve())
            assert "Unable to enumerate" in failed_payload["summary"]["root_results"][1]["error_text"]

            with database(db_path) as connection:
                file_count = connection.execute("SELECT COUNT(*) AS count FROM files").fetchone()["count"]
                runs = connection.execute(
                    "SELECT root_path, status FROM scan_runs ORDER BY id ASC"
                ).fetchall()

            assert file_count == 1
            assert [(row["root_path"], row["status"]) for row in runs] == [
                (str(first_root.resolve()), "completed"),
                (str(failed_root.resolve()), "failed"),
            ]
        finally:
            server.shutdown()

    def test_scan_async_failure_marks_later_root_not_processed(self, tmp_path: Path, monkeypatch):
        from http.server import ThreadingHTTPServer
        from shotsieve import scanner as scanner_module
        from shotsieve.scanner import FileDiscoveryError

        db_path = tmp_path / "data" / "shotsieve.db"
        failed_root = tmp_path / "unavailable"
        later_root = tmp_path / "later"
        failed_root.mkdir()
        later_root.mkdir()
        create_image(later_root / "sample.jpg")
        initialize_database(db_path)

        original_discover_files = scanner_module.discover_files

        def fail_first_root(root, *args, **kwargs):
            if root == failed_root:
                raise FileDiscoveryError(root, PermissionError("first root unavailable"))
            return original_discover_files(root, *args, **kwargs)

        monkeypatch.setattr(scanner_module, "discover_files", fail_first_root)

        port = find_free_port()
        server = ThreadingHTTPServer(("127.0.0.1", port), build_handler(db_path))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            start_request = Request(
                f"http://127.0.0.1:{port}/api/scan/start",
                data=json.dumps({
                    "roots": [str(failed_root), str(later_root)],
                    "recursive": True,
                    "generate_previews": False,
                }).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            job_id = json.loads(urlopen(start_request).read().decode("utf-8"))["job_id"]

            failed_payload = None
            deadline = time.time() + 3
            while time.time() < deadline:
                status = json.loads(
                    urlopen(f"http://127.0.0.1:{port}/api/scan/status?job_id={job_id}").read().decode("utf-8")
                )
                if status["status"] == "failed":
                    failed_payload = status
                    break
                time.sleep(0.05)

            assert failed_payload is not None
            assert [item["status"] for item in failed_payload["summary"]["root_results"]] == [
                "failed",
                "not_processed",
            ]
            with database(db_path) as connection:
                assert connection.execute("SELECT COUNT(*) AS count FROM files").fetchone()["count"] == 0
                assert connection.execute("SELECT COUNT(*) AS count FROM scan_runs").fetchone()["count"] == 1
        finally:
            server.shutdown()

    def test_scan_async_forwards_ignore_rules_to_scanner(self, tmp_path: Path, monkeypatch):
        from dataclasses import dataclass
        from http.server import ThreadingHTTPServer
        from shotsieve import web as web_module

        @dataclass
        class FakeSummary:
            files_seen: int = 0
            files_added: int = 0
            files_updated: int = 0
            files_unchanged: int = 0
            files_removed: int = 0
            files_failed: int = 0

        captured: dict[str, object] = {}

        def fake_scan_root(*_args, **kwargs):
            captured["ignore_rules"] = kwargs["ignore_rules"]
            return FakeSummary()

        monkeypatch.setattr(web_module, "scan_root", fake_scan_root)

        db_path = tmp_path / "data" / "shotsieve.db"
        photo_dir = tmp_path / "photos"
        excluded_dir = photo_dir / "Others" / "Beatrice Low Res"
        excluded_dir.mkdir(parents=True)
        initialize_database(db_path)

        port = find_free_port()
        server = ThreadingHTTPServer(("127.0.0.1", port), build_handler(db_path))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            start_request = Request(
                f"http://127.0.0.1:{port}/api/scan/start",
                data=json.dumps({
                    "roots": [str(photo_dir)],
                    "generate_previews": False,
                    "ignore_rules": [str(excluded_dir)],
                }).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            job_id = json.loads(urlopen(start_request).read().decode("utf-8"))["job_id"]

            deadline = time.time() + 2
            while time.time() < deadline:
                status = json.loads(urlopen(f"http://127.0.0.1:{port}/api/scan/status?job_id={job_id}").read().decode("utf-8"))
                if status["status"] == "completed":
                    break
                time.sleep(0.05)
        finally:
            server.shutdown()

        assert captured["ignore_rules"] == [str(excluded_dir)]

    def test_scan_async_cancel_route_stops_running_job(self, tmp_path: Path, monkeypatch):
        from dataclasses import dataclass
        from http.server import ThreadingHTTPServer
        from shotsieve import web as web_module

        started_event = threading.Event()
        hold_event = threading.Event()

        @dataclass
        class FakeSummary:
            files_seen: int = 4
            files_added: int = 2
            files_updated: int = 1
            files_unchanged: int = 1
            files_removed: int = 0
            files_failed: int = 0

        def fake_scan_root(*args, **kwargs):
            progress_callback = kwargs.get("progress_callback")
            total = int(kwargs.get("files_total_hint") or 4)
            if progress_callback:
                progress_callback(1, total, "scanning")
            started_event.set()
            hold_event.wait(timeout=2)
            if progress_callback:
                progress_callback(2, total, "scanning")
            return FakeSummary()

        monkeypatch.setattr(web_module, "scan_root", fake_scan_root)

        db_path = tmp_path / "data" / "shotsieve.db"
        photo_dir = tmp_path / "photos"
        photo_dir.mkdir()
        create_image(photo_dir / "sample.jpg")
        initialize_database(db_path)

        port = find_free_port()
        server = ThreadingHTTPServer(("127.0.0.1", port), build_handler(db_path))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            start_req = Request(
                f"http://127.0.0.1:{port}/api/scan/start",
                data=json.dumps({"roots": [str(photo_dir)], "recursive": True, "generate_previews": False}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            start_response = urlopen(start_req)
            start_payload = json.loads(start_response.read().decode("utf-8"))
            job_id = start_payload["job_id"]

            assert started_event.wait(timeout=2)

            cancel_req = Request(
                f"http://127.0.0.1:{port}/api/scan/cancel",
                data=json.dumps({"job_id": job_id}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            cancel_response = urlopen(cancel_req)
            cancel_payload = json.loads(cancel_response.read().decode("utf-8"))
            assert cancel_payload == {"job_id": job_id, "cancelled": True}

            hold_event.set()

            failed_payload = None
            deadline = time.time() + 2
            while time.time() < deadline:
                status_response = urlopen(f"http://127.0.0.1:{port}/api/scan/status?job_id={job_id}")
                polled = json.loads(status_response.read().decode("utf-8"))
                if polled["status"] == "failed":
                    failed_payload = polled
                    break
                time.sleep(0.05)

            assert failed_payload is not None
            assert "cancelled" in str(failed_payload.get("error", "")).lower()
        finally:
            hold_event.set()
            server.shutdown()

    def test_scan_async_cancel_stops_real_scanner_mid_root(self, tmp_path: Path, monkeypatch):
        from http.server import ThreadingHTTPServer
        from shotsieve import learned_iqa as learned_iqa_module
        from shotsieve import scanner as scanner_module

        started_event = threading.Event()
        release_event = threading.Event()
        preview_calls: list[str] = []

        def fake_generate_preview(path: Path, preview_dir: Path, *, raw_preview_mode: str = "auto"):
            preview_calls.append(path.name)
            if len(preview_calls) == 1:
                started_event.set()
                release_event.wait(timeout=2)
            return SimpleNamespace(
                path=str(preview_dir / f"{path.stem}.jpg"),
                status="ready",
                width=120,
                height=80,
                capture_time=None,
                error_text=None,
            )

        monkeypatch.setattr(learned_iqa_module, "recommended_cpu_workers", lambda _profile=None: 1)
        monkeypatch.setattr(scanner_module, "generate_preview", fake_generate_preview)

        db_path = tmp_path / "data" / "shotsieve.db"
        photo_dir = tmp_path / "photos"
        photo_dir.mkdir()
        for index in range(3):
            create_image(photo_dir / f"sample-{index}.jpg")
        initialize_database(db_path)

        port = find_free_port()
        server = ThreadingHTTPServer(("127.0.0.1", port), build_handler(db_path))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            start_req = Request(
                f"http://127.0.0.1:{port}/api/scan/start",
                data=json.dumps({
                    "roots": [str(photo_dir)],
                    "recursive": True,
                    "generate_previews": True,
                }).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            start_response = urlopen(start_req)
            start_payload = json.loads(start_response.read().decode("utf-8"))
            job_id = start_payload["job_id"]

            assert started_event.wait(timeout=2)

            cancel_req = Request(
                f"http://127.0.0.1:{port}/api/scan/cancel",
                data=json.dumps({"job_id": job_id}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            cancel_response = urlopen(cancel_req)
            cancel_payload = json.loads(cancel_response.read().decode("utf-8"))
            assert cancel_payload == {"job_id": job_id, "cancelled": True}

            release_event.set()

            failed_payload = None
            deadline = time.time() + 2
            while time.time() < deadline:
                status_response = urlopen(f"http://127.0.0.1:{port}/api/scan/status?job_id={job_id}")
                polled = json.loads(status_response.read().decode("utf-8"))
                if polled["status"] == "failed":
                    failed_payload = polled
                    break
                time.sleep(0.05)

            assert failed_payload is not None
            assert "cancelled" in str(failed_payload.get("error", "")).lower()
            assert len(preview_calls) == 1

            with database(db_path) as connection:
                stored_file_count = connection.execute("SELECT COUNT(*) AS count FROM files").fetchone()["count"]
                latest_scan_run = connection.execute(
                    "SELECT files_seen, status, error_text FROM scan_runs ORDER BY id DESC LIMIT 1"
                ).fetchone()

            assert stored_file_count == 1
            assert latest_scan_run is not None
            assert latest_scan_run["files_seen"] == stored_file_count
            assert latest_scan_run["status"] == "failed"
            assert "cancelled" in str(latest_scan_run["error_text"] or "").lower()
            assert failed_payload["progress"]["files_processed"] == stored_file_count
        finally:
            release_event.set()
            server.shutdown()

    def test_scan_async_cancel_limits_in_flight_preview_work_in_pooled_path(self, tmp_path: Path, monkeypatch):
        from concurrent.futures import ThreadPoolExecutor
        from http.server import ThreadingHTTPServer
        from shotsieve import learned_iqa as learned_iqa_module
        from shotsieve import scanner as scanner_module

        started_event = threading.Event()
        release_event = threading.Event()
        preview_calls: list[str] = []
        preview_lock = threading.Lock()

        class ThreadBackedPool:
            def __init__(self, max_workers=None):
                self._delegate = ThreadPoolExecutor(max_workers=max_workers)

            def submit(self, fn, *args, **kwargs):
                return self._delegate.submit(fn, *args, **kwargs)

            def shutdown(self, wait=True):
                self._delegate.shutdown(wait=wait)

        def fake_generate_preview(path: Path, preview_dir: Path, *, raw_preview_mode: str = "auto"):
            with preview_lock:
                preview_calls.append(path.name)
                started_event.set()
            release_event.wait(timeout=2)
            return SimpleNamespace(
                path=str(preview_dir / f"{path.stem}.jpg"),
                status="ready",
                width=120,
                height=80,
                capture_time=None,
                error_text=None,
            )

        monkeypatch.setattr(learned_iqa_module, "recommended_cpu_workers", lambda _profile=None: 2)
        monkeypatch.setattr(scanner_module.concurrent.futures, "ProcessPoolExecutor", ThreadBackedPool)
        monkeypatch.setattr(scanner_module, "generate_preview", fake_generate_preview)

        db_path = tmp_path / "data" / "shotsieve.db"
        photo_dir = tmp_path / "photos"
        photo_dir.mkdir()
        for index in range(5):
            create_image(photo_dir / f"sample-{index}.jpg")
        initialize_database(db_path)

        port = find_free_port()
        server = ThreadingHTTPServer(("127.0.0.1", port), build_handler(db_path))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            start_req = Request(
                f"http://127.0.0.1:{port}/api/scan/start",
                data=json.dumps({
                    "roots": [str(photo_dir)],
                    "recursive": True,
                    "generate_previews": True,
                }).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            start_response = urlopen(start_req)
            start_payload = json.loads(start_response.read().decode("utf-8"))
            job_id = start_payload["job_id"]

            assert started_event.wait(timeout=2)

            cancel_req = Request(
                f"http://127.0.0.1:{port}/api/scan/cancel",
                data=json.dumps({"job_id": job_id}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            cancel_response = urlopen(cancel_req)
            cancel_payload = json.loads(cancel_response.read().decode("utf-8"))
            assert cancel_payload == {"job_id": job_id, "cancelled": True}

            release_event.set()

            failed_payload = None
            deadline = time.time() + 2
            while time.time() < deadline:
                status_response = urlopen(f"http://127.0.0.1:{port}/api/scan/status?job_id={job_id}")
                polled = json.loads(status_response.read().decode("utf-8"))
                if polled["status"] == "failed":
                    failed_payload = polled
                    break
                time.sleep(0.05)

            assert failed_payload is not None
            assert "cancelled" in str(failed_payload.get("error", "")).lower()
            assert len(preview_calls) <= 2

            with database(db_path) as connection:
                stored_file_count = connection.execute("SELECT COUNT(*) AS count FROM files").fetchone()["count"]
                latest_scan_run = connection.execute(
                    "SELECT files_seen, status FROM scan_runs ORDER BY id DESC LIMIT 1"
                ).fetchone()

            assert stored_file_count == len(preview_calls)
            assert latest_scan_run is not None
            assert latest_scan_run["files_seen"] == stored_file_count
            assert latest_scan_run["status"] == "failed"
            assert failed_payload["progress"]["files_processed"] == stored_file_count
        finally:
            release_event.set()
            server.shutdown()

    def test_scan_async_cancel_via_progress_callback_keeps_files_seen_in_sync(self, tmp_path: Path, monkeypatch):
        from http.server import ThreadingHTTPServer
        from shotsieve import scanner as scanner_module

        allow_second_path = threading.Event()
        first_file_queued = threading.Event()

        db_path = tmp_path / "data" / "shotsieve.db"
        photo_dir = tmp_path / "photos"
        photo_dir.mkdir()
        first_path = photo_dir / "sample-0.jpg"
        second_path = photo_dir / "sample-1.jpg"
        create_image(first_path)
        create_image(second_path)
        initialize_database(db_path)

        def fake_discover_files(*_args, **_kwargs):
            yield first_path
            first_file_queued.set()
            allow_second_path.wait(timeout=2)
            yield second_path

        monkeypatch.setattr(scanner_module, "discover_files", fake_discover_files)

        port = find_free_port()
        server = ThreadingHTTPServer(("127.0.0.1", port), build_handler(db_path))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            start_req = Request(
                f"http://127.0.0.1:{port}/api/scan/start",
                data=json.dumps({
                    "roots": [str(photo_dir)],
                    "recursive": True,
                    "generate_previews": False,
                }).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            start_response = urlopen(start_req)
            start_payload = json.loads(start_response.read().decode("utf-8"))
            job_id = start_payload["job_id"]

            assert first_file_queued.wait(timeout=2)

            cancel_req = Request(
                f"http://127.0.0.1:{port}/api/scan/cancel",
                data=json.dumps({"job_id": job_id}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            cancel_response = urlopen(cancel_req)
            cancel_payload = json.loads(cancel_response.read().decode("utf-8"))
            assert cancel_payload == {"job_id": job_id, "cancelled": True}

            allow_second_path.set()

            failed_payload = None
            deadline = time.time() + 2
            while time.time() < deadline:
                status_response = urlopen(f"http://127.0.0.1:{port}/api/scan/status?job_id={job_id}")
                polled = json.loads(status_response.read().decode("utf-8"))
                if polled["status"] == "failed":
                    failed_payload = polled
                    break
                time.sleep(0.05)

            assert failed_payload is not None
            assert "cancelled" in str(failed_payload.get("error", "")).lower()

            with database(db_path) as connection:
                stored_file_count = connection.execute("SELECT COUNT(*) AS count FROM files").fetchone()["count"]
                latest_scan_run = connection.execute(
                    "SELECT files_seen, status FROM scan_runs ORDER BY id DESC LIMIT 1"
                ).fetchone()

            assert stored_file_count == 0
            assert latest_scan_run is not None
            assert latest_scan_run["files_seen"] == stored_file_count
            assert latest_scan_run["status"] == "failed"
            assert failed_payload["progress"]["files_processed"] == stored_file_count
        finally:
            allow_second_path.set()
            server.shutdown()

    def test_scan_async_cancel_accepts_job_id_from_query_without_request_body(self, tmp_path: Path):
        from http.server import ThreadingHTTPServer

        db_path = tmp_path / "data" / "shotsieve.db"
        initialize_database(db_path)

        port = find_free_port()
        server = ThreadingHTTPServer(("127.0.0.1", port), build_handler(db_path))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            request = Request(
                f"http://127.0.0.1:{port}/api/scan/cancel?job_id=missing-job",
                method="POST",
                headers={"Origin": f"http://127.0.0.1:{port}"},
            )
            response = urlopen(request)
            payload = json.loads(response.read().decode("utf-8"))
        finally:
            server.shutdown()

        assert payload == {"job_id": "missing-job", "cancelled": False}

    def test_scan_async_start_avoids_redundant_prewalk_before_scan_root(self, tmp_path: Path, monkeypatch):
        from dataclasses import dataclass
        from http.server import ThreadingHTTPServer
        from shotsieve import web as web_module

        @dataclass
        class FakeSummary:
            files_seen: int = 1
            files_added: int = 1
            files_updated: int = 0
            files_unchanged: int = 0
            files_removed: int = 0
            files_failed: int = 0

        def fake_scan_root(*args, **kwargs):
            progress_callback = kwargs.get("progress_callback")
            total = int(kwargs.get("files_total_hint") or 0)
            if progress_callback:
                progress_callback(1, total, "scanning")
            return FakeSummary()

        monkeypatch.setattr(web_module, "scan_root", fake_scan_root)

        db_path = tmp_path / "data" / "shotsieve.db"
        photo_dir = tmp_path / "photos"
        photo_dir.mkdir()
        create_image(photo_dir / "sample.jpg")
        initialize_database(db_path)

        port = find_free_port()
        server = ThreadingHTTPServer(("127.0.0.1", port), build_handler(db_path))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            start_req = Request(
                f"http://127.0.0.1:{port}/api/scan/start",
                data=json.dumps({
                    "roots": [str(photo_dir)],
                    "recursive": True,
                    "generate_previews": False,
                    "files_total_hint": 25,
                }).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            start_response = urlopen(start_req)
            start_payload = json.loads(start_response.read().decode("utf-8"))
            job_id = start_payload["job_id"]

            completed_payload = None
            deadline = time.time() + 2
            while time.time() < deadline:
                status_response = urlopen(f"http://127.0.0.1:{port}/api/scan/status?job_id={job_id}")
                polled = json.loads(status_response.read().decode("utf-8"))
                if polled["status"] == "completed":
                    completed_payload = polled
                    break
                time.sleep(0.05)

            assert completed_payload is not None
            assert completed_payload["summary"]["files_seen"] == 1
            assert completed_payload["progress"]["files_total"] == 25
        finally:
            server.shutdown()

    def test_scan_async_start_applies_limit_across_multiple_roots(self, test_server) -> None:
        base_url, db_path, tmp_path = test_server
        first_root = tmp_path / "photos-a"
        second_root = tmp_path / "photos-b"
        first_root.mkdir()
        second_root.mkdir()
        create_image(first_root / "a-1.jpg")
        create_image(first_root / "a-2.jpg")
        create_image(second_root / "b-1.jpg")
        create_image(second_root / "b-2.jpg")

        start_req = Request(
            f"{base_url}/api/scan/start",
            data=json.dumps({
                "roots": [str(first_root), str(second_root)],
                "recursive": True,
                "generate_previews": False,
                "limit": 1,
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        start_response = urlopen(start_req)
        start_payload = json.loads(start_response.read().decode("utf-8"))
        job_id = start_payload["job_id"]

        completed_payload = None
        deadline = time.time() + 2
        while time.time() < deadline:
            status_response = urlopen(f"{base_url}/api/scan/status?job_id={job_id}")
            polled = json.loads(status_response.read().decode("utf-8"))
            if polled["status"] == "completed":
                completed_payload = polled
                break
            time.sleep(0.05)

        assert completed_payload is not None
        assert completed_payload["summary"]["files_seen"] == 1

        with database(db_path) as connection:
            count = connection.execute("SELECT COUNT(*) AS count FROM files").fetchone()["count"]

        assert count == 1

    def test_scan_async_start_applies_offset_across_multiple_roots(self, test_server) -> None:
        base_url, db_path, tmp_path = test_server
        first_root = tmp_path / "photos-a"
        second_root = tmp_path / "photos-b"
        first_root.mkdir()
        second_root.mkdir()
        create_image(first_root / "a-1.jpg")
        create_image(first_root / "a-2.jpg")
        create_image(second_root / "b-1.jpg")
        create_image(second_root / "b-2.jpg")

        start_req = Request(
            f"{base_url}/api/scan/start",
            data=json.dumps({
                "roots": [str(first_root), str(second_root)],
                "recursive": True,
                "generate_previews": False,
                "offset": 3,
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        start_response = urlopen(start_req)
        start_payload = json.loads(start_response.read().decode("utf-8"))
        job_id = start_payload["job_id"]

        completed_payload = None
        deadline = time.time() + 2
        while time.time() < deadline:
            status_response = urlopen(f"{base_url}/api/scan/status?job_id={job_id}")
            polled = json.loads(status_response.read().decode("utf-8"))
            if polled["status"] == "completed":
                completed_payload = polled
                break
            time.sleep(0.05)

        assert completed_payload is not None
        assert completed_payload["summary"]["files_seen"] == 1

        with database(db_path) as connection:
            rows = connection.execute("SELECT path FROM files ORDER BY path").fetchall()

        assert len(rows) == 1
        assert str(second_root.resolve()) in rows[0]["path"]

