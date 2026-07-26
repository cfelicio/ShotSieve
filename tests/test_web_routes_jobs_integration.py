"""Integration tests for web jobs, options, scoring, and compare routes."""
from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace
from http import HTTPStatus
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import pytest

from shotsieve.db import connect, database, initialize_database
from shotsieve.scanner import scan_root
from shotsieve.web import build_handler

from conftest import create_image, find_free_port


class TestWebRoutesJobsIntegration:
    def test_score_start_uses_stored_custom_preview_root(self, test_server, monkeypatch):
        base_url, db_path, tmp_path = test_server
        photo_dir = tmp_path / "photos"
        photo_dir.mkdir()
        custom_preview_dir = tmp_path / "custom-previews"
        create_image(photo_dir / "sample.jpg")

        from shotsieve.db import get_preview_cache_root
        from shotsieve import web as web_module

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
            captured["raw_preview_mode"] = kwargs["raw_preview_mode"]
            return DummySummary()

        monkeypatch.setattr(web_module, "score_files", fake_score_files)

        request = Request(
            f"{base_url}/api/score/start",
            data=json.dumps({"preview_mode": "high-quality"}).encode("utf-8"),
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
        assert captured["raw_preview_mode"] == "high-quality"

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
        assert payload["default_preview_mode"] == "auto"
        assert payload["preview_modes"] == ["fast", "auto", "high-quality"]
        assert payload["raw_preview_auto_min_long_edge"] == 1024
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
        captured_preview_mode: dict[str, str] = {}

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
            captured_preview_mode["value"] = kwargs["raw_preview_mode"]
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
                data=json.dumps({"models": ["topiq_nr", "arniqa"], "root": None, "preview_mode": "fast"}).encode("utf-8"),
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
            assert captured_preview_mode["value"] == "fast"
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
