"""Integration tests for web review, export, preview, and cache routes."""
from __future__ import annotations

import json
import threading
from types import SimpleNamespace
from http import HTTPStatus
from pathlib import Path
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import pytest

from shotsieve.db import database
from shotsieve.scanner import scan_root

from conftest import create_image


class TestWebRoutesReviewIntegration:
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

    def test_analysis_diagnostics_route_returns_unscored_files_in_requested_root(self, test_server):
        base_url, db_path, tmp_path = test_server
        root_a = tmp_path / "library-a"
        root_b = tmp_path / "library-b"
        root_a.mkdir()
        root_b.mkdir()
        file_a = root_a / "attention.jpg"
        file_b = root_b / "other-library.jpg"
        create_image(file_a)
        create_image(file_b)

        with database(db_path) as connection:
            scan_root(connection, root=root_a, recursive=True, extensions=(".jpg",), preview_dir=tmp_path / "previews")
            scan_root(connection, root=root_b, recursive=True, extensions=(".jpg",), preview_dir=tmp_path / "previews")
            connection.execute(
                """
                UPDATE files
                SET analysis_status = 'failed', analysis_error = 'test model failure'
                WHERE path = ?
                """,
                (str(file_a),),
            )

        root_query = quote(str(root_a.resolve()), safe="")
        payload = json.loads(
            urlopen(f"{base_url}/api/analysis-diagnostics?root={root_query}&limit=10").read().decode("utf-8")
        )

        assert payload["total"] == 1
        assert payload["items"] == [{
            "path": str(file_a),
            "format": "jpg",
            "status": "failed",
            "error": "test model failure",
            "last_analysis_time": None,
        }]

    def test_scoped_overview_review_and_rejected_delete_leave_other_library_untouched(self, test_server):
        base_url, db_path, tmp_path = test_server
        root_a = tmp_path / "library-a"
        root_b = tmp_path / "library-b"
        root_a.mkdir()
        root_b.mkdir()
        source_a = root_a / "a.jpg"
        source_b = root_b / "b.jpg"
        create_image(source_a)
        create_image(source_b)

        with database(db_path) as connection:
            scan_root(connection, root=root_a, recursive=True, extensions=(".jpg",), preview_dir=tmp_path / "previews")
            scan_root(connection, root=root_b, recursive=True, extensions=(".jpg",), preview_dir=tmp_path / "previews")
            a_id = int(connection.execute("SELECT id FROM files WHERE path LIKE ?", ("%a.jpg",)).fetchone()["id"])
            b_id = int(connection.execute("SELECT id FROM files WHERE path LIKE ?", ("%b.jpg",)).fetchone()["id"])
            for file_id in (a_id, b_id):
                connection.execute(
                    """
                    INSERT INTO review_state(file_id, decision_state, delete_marked, export_marked, updated_time)
                    VALUES (?, 'delete', 1, 0, '2026-07-20T00:00:00+00:00')
                    """,
                    (file_id,),
                )

        root_b_text = str(root_b.resolve())
        root_b_query = quote(root_b_text, safe="")
        overview = json.loads(
            urlopen(f"{base_url}/api/overview?root={root_b_query}").read().decode("utf-8")
        )
        reviewed = json.loads(
            urlopen(f"{base_url}/api/review/file-ids?marked=delete&root={root_b_query}").read().decode("utf-8")
        )

        assert overview["active_library"] == {
            "root": root_b_text,
            "total_files": 1,
            "scored_files": 0,
            "delete_marked": 1,
            "export_marked": 0,
        }
        assert overview["catalog"]["total_files"] == 2
        assert reviewed["ids"] == [b_id]

        delete_request = Request(
            f"{base_url}/api/files/delete",
            data=json.dumps({
                "selection": {"scope": "review-state", "marked": "delete", "root": root_b_text},
                "selection_revision": reviewed["selection_revision"],
                "delete_from_disk": True,
            }).encode("utf-8"),
            headers={"Content-Type": "application/json", "Origin": base_url},
            method="POST",
        )
        deleted = json.loads(urlopen(delete_request).read().decode("utf-8"))

        assert deleted["deleted_ids"] == [b_id]
        assert not source_b.exists()
        assert source_a.exists()
        with database(db_path) as connection:
            remaining = [int(row["id"]) for row in connection.execute("SELECT id FROM files ORDER BY id").fetchall()]
        assert remaining == [a_id]

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
