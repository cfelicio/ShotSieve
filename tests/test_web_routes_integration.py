"""Tests for web route integration: cache, options, review, logging, and compare."""
from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace
from http import HTTPStatus
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import pytest

from shotsieve.db import connect, database, initialize_database
from shotsieve.scanner import scan_root
from shotsieve.web import build_handler

from conftest import create_image, find_free_port


class TestRouteHandlingIntegration:
    def test_cache_post_route_family_handles_missing_scope(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
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

        def fake_clear_cache_scope(*_args, **_kwargs):
            raise AssertionError("clear_cache_scope should not run for missing scope")

        monkeypatch.setattr(route_module, "send_json", fake_send_json)

        def fake_prune_missing_cache_entries(_connection, *, preview_cache_root):
            captured["preview_cache_root"] = preview_cache_root
            return 7

        deps = SimpleNamespace(
            read_json_body=lambda _handler, *, max_body_size: {"scope": "missing"},
            required_choice=lambda value, *, name, choices: value,
            database=lambda _path: _DatabaseContext(),
            get_preview_cache_root=lambda _connection, *, db_path, persist: preview_root,
            prune_missing_cache_entries=fake_prune_missing_cache_entries,
            clear_cache_scope=fake_clear_cache_scope,
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
        handler = SimpleNamespace(path="/api/cache/clear", headers={"Content-Length": "20"})

        handled = route_module._handle_cache_post_routes(handler, context, urlparse(handler.path))

        assert handled is True
        assert captured["preview_cache_root"] == preview_root
        assert captured["payload"] == {"files": 7, "scores": 0, "review": 0, "scan_runs": 0}

    def test_api_json_responses_include_basic_security_headers(self, test_server):
        base_url, _, _ = test_server
        response = urlopen(f"{base_url}/api/options")

        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.headers["X-Frame-Options"] == "DENY"

    def test_overview_route_uses_web_module_override_after_server_start(self, test_server, monkeypatch):
        base_url, _, _ = test_server
        from shotsieve import web as web_module

        monkeypatch.setattr(web_module, "review_overview", lambda _connection: {"patched": True})

        payload = json.loads(urlopen(f"{base_url}/api/overview").read().decode("utf-8"))

        assert payload == {"patched": True}

    def test_options_preview_dir_defaults_to_data_previews_next_to_db(self, test_server):
        base_url, db_path, _ = test_server
        response = urlopen(f"{base_url}/api/options")
        payload = json.loads(response.read().decode("utf-8"))
        expected_preview_dir = str((db_path.parent / "previews").resolve())
        assert payload["preview_dir"] == expected_preview_dir

    def test_options_preview_dir_uses_stored_custom_preview_root(self, test_server):
        base_url, db_path, tmp_path = test_server
        photo_dir = tmp_path / "photos"
        photo_dir.mkdir()
        custom_preview_dir = tmp_path / "custom-previews"
        create_image(photo_dir / "sample.jpg")

        from shotsieve.db import database

        with database(db_path) as connection:
            scan_root(
                connection,
                root=photo_dir,
                recursive=True,
                extensions=(".jpg",),
                preview_dir=custom_preview_dir,
            )

        response = urlopen(f"{base_url}/api/options")
        payload = json.loads(response.read().decode("utf-8"))

        assert payload["preview_dir"] == str(custom_preview_dir.resolve())

    def test_options_preview_lookup_does_not_persist_metadata(self, test_server):
        base_url, db_path, _ = test_server

        from shotsieve.db import database

        with database(db_path) as connection:
            before = connection.execute(
                "SELECT value FROM app_metadata WHERE key = 'preview_cache_root'"
            ).fetchone()

        assert before is None

        response = urlopen(f"{base_url}/api/options")
        payload = json.loads(response.read().decode("utf-8"))

        with database(db_path) as connection:
            after = connection.execute(
                "SELECT value FROM app_metadata WHERE key = 'preview_cache_root'"
            ).fetchone()

        assert response.status == 200
        assert payload["preview_dir"] == str((db_path.parent / "previews").resolve())
        assert after is None

    def test_options_preview_lookup_handles_legacy_root_with_sidecar(self, test_server):
        base_url, db_path, tmp_path = test_server
        photo_dir = tmp_path / "photos"
        photo_dir.mkdir()
        legacy_preview_dir = tmp_path / "legacy-previews"
        create_image(photo_dir / "sample.jpg")

        from shotsieve.db import database

        with database(db_path) as connection:
            scan_root(
                connection,
                root=photo_dir,
                recursive=True,
                extensions=(".jpg",),
                preview_dir=legacy_preview_dir,
            )
            connection.execute("DELETE FROM app_metadata WHERE key = 'preview_cache_root'")
            (legacy_preview_dir / "keep-me.txt").write_text("legacy", encoding="utf-8")

        response = urlopen(f"{base_url}/api/options")
        payload = json.loads(response.read().decode("utf-8"))

        assert response.status == 200
        assert payload["preview_dir"] == str(legacy_preview_dir.resolve())

    def test_score_start_uses_stored_custom_preview_root(self, test_server, monkeypatch):
        base_url, db_path, tmp_path = test_server
        photo_dir = tmp_path / "photos"
        photo_dir.mkdir()
        custom_preview_dir = tmp_path / "custom-previews"
        create_image(photo_dir / "sample.jpg")

        from shotsieve import web as web_module
        from shotsieve.db import database, get_preview_cache_root

        with database(db_path) as connection:
            scan_root(
                connection,
                root=photo_dir,
                recursive=True,
                extensions=(".jpg",),
                preview_dir=custom_preview_dir,
            )

        monkeypatch.setattr(web_module, "_require_learned_runtime", lambda **kwargs: None)

        class DummySummary:
            rows_loaded = 0
            files_considered = 0
            files_scored = 0
            learned_scored = 0
            files_skipped = 0
            files_failed = 0

        captured: dict[str, Path] = {}

        def fake_score_files(connection, **kwargs):
            captured["preview_dir"] = kwargs["preview_dir"]
            return DummySummary()

        monkeypatch.setattr(web_module, "score_files", fake_score_files)

        request = Request(
            f"{base_url}/api/score/start",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        response = urlopen(request)
        payload = json.loads(response.read().decode("utf-8"))
        job_id = payload["job_id"]

        result_payload = None
        deadline = time.time() + 2
        while time.time() < deadline:
            status_response = urlopen(f"{base_url}/api/score/status?job_id={job_id}")
            status_payload = json.loads(status_response.read().decode("utf-8"))
            if status_payload["status"] == "completed":
                result_response = urlopen(f"{base_url}/api/score/result?job_id={job_id}")
                result_payload = json.loads(result_response.read().decode("utf-8"))
                break
            time.sleep(0.05)

        assert response.status == 200
        assert payload["status"] == "running"
        assert result_payload is not None
        assert result_payload["rows_loaded"] == 0
        assert captured["preview_dir"] == custom_preview_dir.resolve()

        with database(db_path) as connection:
            assert get_preview_cache_root(connection, db_path=db_path) == custom_preview_dir.resolve()

    def test_score_start_uses_web_default_batch_size_override_after_server_start(self, test_server, monkeypatch):
        base_url, _, _ = test_server
        from shotsieve import web as web_module

        monkeypatch.setattr(web_module, "_require_learned_runtime", lambda **kwargs: None)
        monkeypatch.setattr(web_module, "DEFAULT_BATCH_SIZE", 17)

        class DummySummary:
            rows_loaded = 0
            files_considered = 0
            files_scored = 0
            learned_scored = 0
            files_skipped = 0
            files_failed = 0

        captured: dict[str, int] = {}

        def fake_score_files(connection, **kwargs):
            captured["batch_size"] = kwargs["learned_batch_size"]
            return DummySummary()

        monkeypatch.setattr(web_module, "score_files", fake_score_files)

        request = Request(
            f"{base_url}/api/score/start",
            data=b"{}",
            headers={"Content-Type": "application/json", "Origin": base_url},
            method="POST",
        )
        response = urlopen(request)
        payload = json.loads(response.read().decode("utf-8"))
        job_id = payload["job_id"]

        deadline = time.time() + 2
        while time.time() < deadline:
            status_response = urlopen(f"{base_url}/api/score/status?job_id={job_id}")
            status_payload = json.loads(status_response.read().decode("utf-8"))
            if status_payload["status"] == "completed":
                urlopen(f"{base_url}/api/score/result?job_id={job_id}").read()
                break
            time.sleep(0.05)

        assert response.status == 200
        assert payload["status"] == "running"
        assert captured["batch_size"] == 17

    def test_options_payload_defaults_to_learned_models_only(self, test_server):
        base_url, _, _ = test_server
        response = urlopen(f"{base_url}/api/options")
        payload = json.loads(response.read().decode("utf-8"))

        assert "supports_technical_only" not in payload
        assert payload["default_scoring_mode"] == payload["learned"]["default_model"]
        assert "technical-only" not in payload["learned_models"]
        assert set(payload["learned_models"]).issubset({"topiq_nr", "clipiqa", "qalign"})
        assert "topiq_nr" in payload["learned_models"]
        assert "clipiqa" in payload["learned_models"]
        assert "auto_runtime_priority" in payload["learned"]
        assert "cpu" in payload["learned"]["auto_runtime_priority"]
        assert payload["runtime_targets"] == ["auto", "cpu", "cuda", "xpu", "directml", "mps"]

    def test_options_payload_hides_qalign_for_installed_cpu_runtime(self, test_server, monkeypatch):
        base_url, _, _ = test_server
        from shotsieve import web as web_module
        import shotsieve.learned_iqa as learned_iqa_module

        class FakePyiqa:
            @staticmethod
            def list_models(metric_mode: str):
                assert metric_mode == "NR"
                return ["topiq_nr", "clipiqa", "qalign"]

        class FakeTorch:
            __version__ = "2.11.0+cpu"

            @staticmethod
            def device(name: str) -> str:
                return name

            class cuda:
                @staticmethod
                def is_available() -> bool:
                    return False

            class xpu:
                @staticmethod
                def is_available() -> bool:
                    return False

        monkeypatch.setattr(learned_iqa_module, "import_pyiqa_runtime", lambda: (FakePyiqa, FakeTorch))
        monkeypatch.setattr(web_module, "available_learned_backends", learned_iqa_module.available_learned_backends)
        monkeypatch.setattr(web_module, "runtime_curated_learned_models", learned_iqa_module.runtime_curated_learned_models)

        response = urlopen(f"{base_url}/api/options")
        payload = json.loads(response.read().decode("utf-8"))

        assert payload["learned"]["default_runtime"] == "cpu"
        assert payload["learned_models"] == ["topiq_nr", "clipiqa"]

    def test_options_payload_keeps_qalign_for_installed_accelerator_runtime(self, test_server, monkeypatch):
        base_url, _, _ = test_server
        from shotsieve import web as web_module
        import shotsieve.learned_iqa as learned_iqa_module
        import shotsieve.learned_iqa_runtime as learned_iqa_runtime_module

        class FakePyiqa:
            @staticmethod
            def list_models(metric_mode: str):
                assert metric_mode == "NR"
                return ["topiq_nr", "clipiqa", "qalign"]

        class FakeTorch:
            __version__ = "2.11.0+cu124"

            @staticmethod
            def device(name: str) -> str:
                return name

            class cuda:
                @staticmethod
                def is_available() -> bool:
                    return True

            class xpu:
                @staticmethod
                def is_available() -> bool:
                    return False

        monkeypatch.setattr(learned_iqa_module, "import_pyiqa_runtime", lambda: (FakePyiqa, FakeTorch))
        monkeypatch.setattr(web_module, "available_learned_backends", learned_iqa_module.available_learned_backends)
        monkeypatch.setattr(web_module, "runtime_curated_learned_models", learned_iqa_module.runtime_curated_learned_models)
        # On macOS the auto runtime order is ("mps", "cpu") and never checks
        # CUDA.  Force a Linux-like order so the test exercises the CUDA path.
        monkeypatch.setattr(learned_iqa_runtime_module.platform, "system", lambda: "Linux")

        response = urlopen(f"{base_url}/api/options")
        payload = json.loads(response.read().decode("utf-8"))

        assert payload["learned"]["default_runtime"] == "cuda"
        assert "qalign" in payload["learned_models"]

    def test_options_route_uses_refreshed_hardware_cache_after_invalidation(self, test_server, monkeypatch):
        base_url, _, _ = test_server
        from shotsieve import web as web_module
        import shotsieve.learned_iqa as learned_iqa_module

        class FakePyiqa:
            @staticmethod
            def list_models(metric_mode: str):
                assert metric_mode == "NR"
                return ["topiq_nr"]

        class FakeTorch:
            __version__ = "2.9.0"

            @staticmethod
            def device(name: str) -> str:
                return name

            class cuda:
                @staticmethod
                def is_available() -> bool:
                    return False

            class xpu:
                @staticmethod
                def is_available() -> bool:
                    return False

        state = {"vram_mb": 2048}

        monkeypatch.setattr(learned_iqa_module, "_cached_hw_capabilities", None)
        monkeypatch.setattr(learned_iqa_module, "_effective_cpu_count", lambda: 8)
        monkeypatch.setattr(learned_iqa_module, "detect_system_ram_mb", lambda: 16384)
        monkeypatch.setattr(learned_iqa_module, "detect_gpu_vram_mb", lambda: state["vram_mb"])
        monkeypatch.setattr(learned_iqa_module, "import_pyiqa_runtime", lambda: (FakePyiqa, FakeTorch))
        monkeypatch.setattr(web_module, "available_learned_backends", learned_iqa_module.available_learned_backends)

        learned_iqa_module.detect_hardware_capabilities()
        state["vram_mb"] = 4096
        learned_iqa_module.invalidate_hw_cache()

        response = urlopen(f"{base_url}/api/options")
        payload = json.loads(response.read().decode("utf-8"))

        assert payload["learned"]["hardware"]["vram_mb"] == 4096

    def test_unknown_get_route_returns_404(self, test_server):
        base_url, _, _ = test_server
        with pytest.raises(HTTPError) as exc_info:
            urlopen(f"{base_url}/api/nonexistent")
        assert exc_info.value.code == 404

    def test_review_file_ids_route_returns_marked_items_without_scores(self, test_server):
        base_url, db_path, tmp_path = test_server
        photo_dir = tmp_path / "photos"
        photo_dir.mkdir()
        create_image(photo_dir / "keep.jpg")
        create_image(photo_dir / "reject.jpg")

        with database(db_path) as connection:
            scan_root(
                connection,
                root=photo_dir,
                recursive=True,
                extensions=(".jpg",),
                preview_dir=tmp_path / "previews",
            )
            reject_id = connection.execute(
                "SELECT id FROM files WHERE path LIKE ? LIMIT 1",
                ("%reject.jpg",),
            ).fetchone()["id"]

        review_request = Request(
            f"{base_url}/api/review",
            data=json.dumps({
                "file_id": reject_id,
                "decision_state": "delete",
                "delete_marked": True,
                "export_marked": False,
            }).encode("utf-8"),
            headers={"Content-Type": "application/json", "Origin": base_url},
            method="POST",
        )
        urlopen(review_request).read()

        ids_payload = json.loads(urlopen(f"{base_url}/api/review/file-ids?marked=delete").read().decode("utf-8"))

        assert ids_payload["ids"] == [reject_id]
        assert isinstance(ids_payload.get("selection_revision"), str)

    def test_review_file_ids_route_requires_supported_mark(self, test_server):
        base_url, _, _ = test_server

        with pytest.raises(HTTPError) as exc_info:
            urlopen(f"{base_url}/api/review/file-ids")

        assert exc_info.value.code == HTTPStatus.BAD_REQUEST
        assert "marked" in exc_info.value.read().decode("utf-8")

    def test_review_file_ids_route_supports_pagination(self, test_server):
        base_url, db_path, tmp_path = test_server
        photo_dir = tmp_path / "photos"
        photo_dir.mkdir()
        for name in ("a.jpg", "b.jpg", "c.jpg"):
            create_image(photo_dir / name)

        with database(db_path) as connection:
            scan_root(
                connection,
                root=photo_dir,
                recursive=True,
                extensions=(".jpg",),
                preview_dir=tmp_path / "previews",
            )
            file_ids = [
                row["id"]
                for row in connection.execute("SELECT id FROM files ORDER BY id ASC").fetchall()
            ]

        for file_id in file_ids:
            review_request = Request(
                f"{base_url}/api/review",
                data=json.dumps({
                    "file_id": file_id,
                    "decision_state": "delete",
                    "delete_marked": True,
                    "export_marked": False,
                }).encode("utf-8"),
                headers={"Content-Type": "application/json", "Origin": base_url},
                method="POST",
            )
            urlopen(review_request).read()

        first_page = json.loads(
            urlopen(f"{base_url}/api/review/file-ids?marked=delete&limit=2&offset=0").read().decode("utf-8")
        )
        second_page = json.loads(
            urlopen(f"{base_url}/api/review/file-ids?marked=delete&limit=2&offset=2").read().decode("utf-8")
        )

        assert first_page["ids"] == file_ids[:2]
        assert second_page["ids"] == file_ids[2:]
        assert isinstance(first_page.get("selection_revision"), str)
        assert isinstance(second_page.get("selection_revision"), str)

    def test_unknown_get_route_logs_warning_through_app_logger(self, test_server, monkeypatch):
        base_url, _, _ = test_server
        from shotsieve import web as web_module

        messages: list[str] = []

        def fake_warning(message: str, *args, **kwargs) -> None:
            rendered = message % args if args else message
            messages.append(rendered)

        monkeypatch.setattr(web_module.log, "warning", fake_warning)

        with pytest.raises(HTTPError) as exc_info:
            urlopen(f"{base_url}/api/nonexistent")

        assert exc_info.value.code == 404
        assert any('"GET /api/nonexistent HTTP/' in message and " 404 " in message for message in messages)

    def test_successful_get_route_logs_at_debug_level(self, test_server, monkeypatch):
        base_url, _, _ = test_server
        from shotsieve import web as web_module

        messages: list[str] = []

        def fake_debug(message: str, *args, **kwargs) -> None:
            rendered = message % args if args else message
            messages.append(rendered)

        monkeypatch.setattr(web_module.log, "debug", fake_debug)

        response = urlopen(f"{base_url}/")

        assert response.status == 200
        assert any('"GET / HTTP/' in message and " 200 " in message for message in messages)

    def test_log_message_sanitizes_control_characters_before_logging(self, tmp_path: Path, monkeypatch):
        db_path = tmp_path / "data" / "shotsieve.db"
        handler_class = build_handler(db_path)
        handler = handler_class.__new__(handler_class)

        from shotsieve import web as web_module

        messages: list[str] = []

        def fake_debug(message: str, *args, **kwargs) -> None:
            rendered = message % args if args else message
            messages.append(rendered)

        monkeypatch.setattr(web_module.log, "debug", fake_debug)
        monkeypatch.setattr(handler, "address_string", lambda: "127.0.0.1")
        monkeypatch.setattr(handler, "log_date_time_string", lambda: "19/Apr/2026 00:00:00")

        handler.log_message("%s", "bad\x1b[31mline\nnext")

        assert messages
        assert "\x1b" not in messages[0]
        assert "\n" not in messages[0]
        assert "\\x1b[31m" in messages[0]
        assert "\\x0a" in messages[0]

    def test_unknown_post_route_returns_404(self, test_server):
        base_url, _, _ = test_server
        req = Request(
            f"{base_url}/api/nonexistent",
            data=b'{"test": true}',
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(HTTPError) as exc_info:
            urlopen(req)
        assert exc_info.value.code == 404

    def test_cache_clear_route_missing_scope_prunes_missing_entries(self, test_server, monkeypatch):
        base_url, db_path, _ = test_server
        from shotsieve import web as web_module

        captured: dict[str, object] = {}

        def fake_prune_missing_cache_entries(_connection, *, preview_cache_root):
            captured["preview_cache_root"] = preview_cache_root
            return 7

        def fake_clear_cache_scope(*_args, **_kwargs):
            raise AssertionError("clear_cache_scope should not run for missing scope")

        monkeypatch.setattr(web_module, "prune_missing_cache_entries", fake_prune_missing_cache_entries)
        monkeypatch.setattr(web_module, "clear_cache_scope", fake_clear_cache_scope)

        request = Request(
            f"{base_url}/api/cache/clear",
            data=json.dumps({"scope": "missing"}).encode("utf-8"),
            headers={"Content-Type": "application/json", "Origin": base_url},
            method="POST",
        )

        response = urlopen(request)
        payload = json.loads(response.read().decode("utf-8"))

        assert response.status == HTTPStatus.OK
        assert payload == {"files": 7, "scores": 0, "review": 0, "scan_runs": 0}
        assert captured["preview_cache_root"] == (db_path.parent / "previews").resolve()

    def test_files_export_route_rejects_non_string_destination_and_mode(self, test_server, monkeypatch):
        base_url, _, _ = test_server
        from shotsieve import web as web_module

        called = {"export": False}

        def fake_export_files(*_args, **_kwargs):
            called["export"] = True
            raise AssertionError("export_files should not be called for invalid input")

        monkeypatch.setattr(web_module, "export_files", fake_export_files)

        req = Request(
            f"{base_url}/api/files/export",
            data=json.dumps({"file_ids": [1], "destination": None, "mode": 123}).encode("utf-8"),
            headers={"Content-Type": "application/json", "Origin": base_url},
            method="POST",
        )

        with pytest.raises(HTTPError) as exc_info:
            urlopen(req)

        assert exc_info.value.code == HTTPStatus.BAD_REQUEST
        assert called["export"] is False

    def test_remove_cache_route_is_removed(self, test_server):
        base_url, _, _ = test_server
        req = Request(
            f"{base_url}/api/files/remove-cache",
            data=json.dumps({"file_ids": [1]}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with pytest.raises(HTTPError) as exc_info:
            urlopen(req)

        assert exc_info.value.code == HTTPStatus.NOT_FOUND

    def test_malformed_review_payload_returns_400(self, test_server):
        base_url, _, _ = test_server
        req = Request(
            f"{base_url}/api/review",
            data=b'{"delete_marked": true}',
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(HTTPError) as exc_info:
            urlopen(req)
        assert exc_info.value.code == HTTPStatus.BAD_REQUEST
        assert "file_id" in exc_info.value.read().decode("utf-8")

    def test_scan_sync_route_is_removed(self, tmp_path: Path):
        from http.server import ThreadingHTTPServer

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
            req = Request(
                f"http://127.0.0.1:{port}/api/scan",
                data=json.dumps({"roots": [str(photo_dir)], "recursive": True}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with pytest.raises(HTTPError) as exc_info:
                urlopen(req)
        finally:
            server.shutdown()

        assert exc_info.value.code == HTTPStatus.NOT_FOUND

    def test_score_sync_route_is_removed(self, tmp_path: Path, monkeypatch):
        from http.server import ThreadingHTTPServer
        from shotsieve import web as web_module

        monkeypatch.setattr(web_module, "_require_learned_runtime", lambda **kwargs: None)

        db_path = tmp_path / "data" / "shotsieve.db"
        initialize_database(db_path)
        port = find_free_port()
        server = ThreadingHTTPServer(("127.0.0.1", port), build_handler(db_path))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            req = Request(
                f"http://127.0.0.1:{port}/api/score",
                data=b"{}",
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with pytest.raises(HTTPError) as exc_info:
                urlopen(req)
        finally:
            server.shutdown()

        assert exc_info.value.code == HTTPStatus.NOT_FOUND

    def test_compare_models_sync_route_is_removed(self, tmp_path: Path, monkeypatch):
        from http.server import ThreadingHTTPServer
        from shotsieve import web as web_module

        monkeypatch.setattr(
            web_module,
            "available_learned_backends",
            lambda *, resource_profile=None: {"pyiqa": "installed"},
        )

        db_path = tmp_path / "data" / "shotsieve.db"
        initialize_database(db_path)
        port = find_free_port()
        server = ThreadingHTTPServer(("127.0.0.1", port), build_handler(db_path))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            req = Request(
                f"http://127.0.0.1:{port}/api/compare-models",
                data=json.dumps({"models": ["topiq_nr", "arniqa"], "root": None}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with pytest.raises(HTTPError) as exc_info:
                urlopen(req)
        finally:
            server.shutdown()

        assert exc_info.value.code == HTTPStatus.NOT_FOUND

    def test_compare_estimate_route_counts_files_for_selected_root(self, tmp_path: Path):
        from http.server import ThreadingHTTPServer

        db_path = tmp_path / "data" / "shotsieve.db"
        preview_dir = tmp_path / "previews"
        first_root = tmp_path / "photos-a"
        second_root = tmp_path / "photos-b"
        first_root.mkdir()
        second_root.mkdir()
        create_image(first_root / "a.jpg")
        create_image(first_root / "b.jpg")
        create_image(second_root / "c.jpg")

        initialize_database(db_path)
        with connect(db_path) as connection:
            scan_root(connection, root=first_root, recursive=True, extensions=(".jpg",), preview_dir=preview_dir)
            scan_root(connection, root=second_root, recursive=True, extensions=(".jpg",), preview_dir=preview_dir)

        port = find_free_port()
        server = ThreadingHTTPServer(("127.0.0.1", port), build_handler(db_path))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            req = Request(
                f"http://127.0.0.1:{port}/api/compare-estimate",
                data=json.dumps({"root": str(first_root)}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            response = urlopen(req)
            payload = json.loads(response.read().decode("utf-8"))
        finally:
            server.shutdown()

        assert payload == {"rows_total": 2}

    def test_compare_models_async_status_and_result_routes(self, tmp_path: Path, monkeypatch):
        from dataclasses import dataclass
        from http.server import ThreadingHTTPServer
        from shotsieve import web as web_module
        from shotsieve.scoring import AnalysisProgress

        monkeypatch.setattr(
            web_module,
            "available_learned_backends",
            lambda *, resource_profile=None: {"pyiqa": "installed"},
        )

        started_event = threading.Event()
        release_event = threading.Event()

        @dataclass
        class FakeComparison:
            model_names: list[str]
            rows: list[dict[str, object]]
            files_considered: int = 3
            files_compared: int = 3
            files_skipped: int = 0
            files_failed: int = 0
            elapsed_seconds: float = 1.8
            model_timings_seconds: dict[str, float] | None = None

        def fake_compare_models(*args, **kwargs):
            progress_callback = kwargs.get("progress_callback")
            if progress_callback:
                progress_callback(
                    AnalysisProgress(
                        model_name="topiq_nr",
                        model_index=1,
                        model_count=2,
                        files_processed=0,
                        files_total=3,
                    )
                )
            started_event.set()
            release_event.wait(timeout=2)
            if progress_callback:
                progress_callback(
                    AnalysisProgress(
                        model_name="arniqa",
                        model_index=2,
                        model_count=2,
                        files_processed=3,
                        files_total=3,
                    )
                )
            return FakeComparison(
                model_names=["topiq_nr", "arniqa"],
                rows=[
                    {
                        "path": "C:/photos/sample.jpg",
                        "topiq_nr_score": 82.0,
                        "topiq_nr_confidence": 91.0,
                        "topiq_nr_raw": 0.82,
                        "arniqa_score": 74.0,
                        "arniqa_confidence": 85.0,
                        "arniqa_raw": 0.74,
                    }
                ],
                model_timings_seconds={"topiq_nr": 0.9, "arniqa": 0.9},
            )

        monkeypatch.setattr(web_module, "compare_learned_models", fake_compare_models)

        db_path = tmp_path / "data" / "shotsieve.db"
        initialize_database(db_path)
        port = find_free_port()
        server = ThreadingHTTPServer(("127.0.0.1", port), build_handler(db_path))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            start_req = Request(
                f"http://127.0.0.1:{port}/api/compare-models/start",
                data=json.dumps({"models": ["topiq_nr", "arniqa"], "root": None}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            start_response = urlopen(start_req)
            start_payload = json.loads(start_response.read().decode("utf-8"))
            job_id = start_payload["job_id"]

            assert started_event.wait(timeout=2)

            status_response = urlopen(f"http://127.0.0.1:{port}/api/compare-models/status?job_id={job_id}")
            status_payload = json.loads(status_response.read().decode("utf-8"))
            assert status_payload["status"] in {"running", "completed"}
            assert status_payload["progress"]["model_name"] == "topiq_nr"
            assert status_payload["progress"]["files_total"] == 3

            release_event.set()

            completed_payload = None
            deadline = time.time() + 2
            while time.time() < deadline:
                status_response = urlopen(f"http://127.0.0.1:{port}/api/compare-models/status?job_id={job_id}")
                polled = json.loads(status_response.read().decode("utf-8"))
                if polled["status"] == "completed":
                    completed_payload = polled
                    break
                time.sleep(0.05)

            assert completed_payload is not None
            assert completed_payload["summary"]["files_compared"] == 3

            result_response = urlopen(f"http://127.0.0.1:{port}/api/compare-models/result?job_id={job_id}")
            result_payload = json.loads(result_response.read().decode("utf-8"))
            assert result_payload["model_names"] == ["topiq_nr", "arniqa"]
            assert result_payload["rows"][0]["arniqa_score"] == 74.0
        finally:
            release_event.set()
            server.shutdown()

    def test_compare_models_async_result_includes_pre_row_compare_failures(self, tmp_path: Path, monkeypatch):
        from http.server import ThreadingHTTPServer
        from shotsieve import web as web_module

        monkeypatch.setattr(
            web_module,
            "available_learned_backends",
            lambda *, resource_profile=None: {"pyiqa": "installed"},
        )

        def fake_compare_models(*args, **kwargs):
            return SimpleNamespace(
                model_names=["topiq_nr", "arniqa"],
                rows=[],
                files_considered=1,
                files_compared=0,
                files_skipped=0,
                files_failed=1,
                elapsed_seconds=0.6,
                model_timings_seconds={},
                compare_failures=[
                    {
                        "file_id": 3,
                        "path": "C:/photos/broken.heic",
                        "reason": "HEIF preview generation failed",
                        "stage": "preview_generation",
                    }
                ],
            )

        monkeypatch.setattr(web_module, "compare_learned_models", fake_compare_models)

        db_path = tmp_path / "data" / "shotsieve.db"
        initialize_database(db_path)
        port = find_free_port()
        server = ThreadingHTTPServer(("127.0.0.1", port), build_handler(db_path))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            start_req = Request(
                f"http://127.0.0.1:{port}/api/compare-models/start",
                data=json.dumps({"models": ["topiq_nr", "arniqa"], "root": None}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            start_response = urlopen(start_req)
            start_payload = json.loads(start_response.read().decode("utf-8"))
            job_id = start_payload["job_id"]

            result_payload = None
            deadline = time.time() + 2
            while time.time() < deadline:
                status_response = urlopen(f"http://127.0.0.1:{port}/api/compare-models/status?job_id={job_id}")
                status_payload = json.loads(status_response.read().decode("utf-8"))
                if status_payload["status"] == "completed":
                    result_response = urlopen(f"http://127.0.0.1:{port}/api/compare-models/result?job_id={job_id}")
                    result_payload = json.loads(result_response.read().decode("utf-8"))
                    break
                time.sleep(0.05)

            assert start_response.status == 200
            assert result_payload is not None
            assert result_payload["rows"] == []
            assert result_payload["files_failed"] == 1
            assert result_payload["compare_failures"] == [
                {
                    "file_id": 3,
                    "path": "C:/photos/broken.heic",
                    "reason": "HEIF preview generation failed",
                    "stage": "preview_generation",
                }
            ]
        finally:
            server.shutdown()

    def test_compare_models_async_result_includes_truncation_contract(self, tmp_path: Path, monkeypatch):
        from http.server import ThreadingHTTPServer
        from shotsieve import web as web_module

        monkeypatch.setattr(
            web_module,
            "available_learned_backends",
            lambda *, resource_profile=None: {"pyiqa": "installed"},
        )

        def fake_compare_models(*args, **kwargs):
            return SimpleNamespace(
                model_names=["topiq_nr", "arniqa"],
                rows=[
                    {
                        "file_id": 1,
                        "path": "C:/photos/sample.jpg",
                        "topiq_nr_score": 82.0,
                        "topiq_nr_confidence": 91.0,
                        "topiq_nr_raw": 0.82,
                        "arniqa_score": 74.0,
                        "arniqa_confidence": 85.0,
                        "arniqa_raw": 0.74,
                    }
                ],
                compare_failures=[],
                files_considered=1,
                files_compared=1,
                files_skipped=0,
                files_failed=0,
                elapsed_seconds=0.6,
                model_timings_seconds={"topiq_nr": 0.3, "arniqa": 0.3},
                requested_rows_total=32000,
                processed_rows_total=10000,
                truncated=True,
                max_rows=10000,
            )

        monkeypatch.setattr(web_module, "compare_learned_models", fake_compare_models)

        db_path = tmp_path / "data" / "shotsieve.db"
        initialize_database(db_path)
        port = find_free_port()
        server = ThreadingHTTPServer(("127.0.0.1", port), build_handler(db_path))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            start_req = Request(
                f"http://127.0.0.1:{port}/api/compare-models/start",
                data=json.dumps({"models": ["topiq_nr", "arniqa"], "root": None}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            start_response = urlopen(start_req)
            start_payload = json.loads(start_response.read().decode("utf-8"))
            job_id = start_payload["job_id"]

            result_payload = None
            deadline = time.time() + 2
            while time.time() < deadline:
                status_response = urlopen(f"http://127.0.0.1:{port}/api/compare-models/status?job_id={job_id}")
                status_payload = json.loads(status_response.read().decode("utf-8"))
                if status_payload["status"] == "completed":
                    result_response = urlopen(f"http://127.0.0.1:{port}/api/compare-models/result?job_id={job_id}")
                    result_payload = json.loads(result_response.read().decode("utf-8"))
                    break
                time.sleep(0.05)

            assert start_response.status == 200
            assert result_payload is not None
            assert result_payload["requested_rows_total"] == 32000
            assert result_payload["processed_rows_total"] == 10000
            assert result_payload["truncated"] is True
            assert result_payload["max_rows"] == 10000
        finally:
            server.shutdown()

