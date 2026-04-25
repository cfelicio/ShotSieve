"""Tests for media streaming and preview delivery."""
from __future__ import annotations

import io
import platform
import threading
from hashlib import sha1
from http import HTTPStatus
from pathlib import Path
from urllib.request import urlopen
from urllib.error import HTTPError

import pytest
from PIL import Image

from shotsieve.db import database, initialize_database
from shotsieve.scanner import canonical_path_key, scan_root
from shotsieve.web import build_handler

from conftest import create_image, find_free_port


class TestMediaStreaming:
    def test_media_response_includes_content_length(self, test_server):
        base_url, db_path, tmp_path = test_server
        photo_dir = tmp_path / "photos"
        photo_dir.mkdir()
        create_image(photo_dir / "test.jpg")
        preview_dir = tmp_path / "data" / "previews"

        from shotsieve.db import database
        with database(db_path) as connection:
            scan_root(
                connection,
                root=photo_dir,
                recursive=True,
                extensions=(".jpg",),
                preview_dir=preview_dir,
            )
            file_id = connection.execute("SELECT id FROM files LIMIT 1").fetchone()["id"]

        response = urlopen(f"{base_url}/api/media/source?id={file_id}")
        content_length = response.headers.get("Content-Length")
        assert content_length is not None
        body = response.read()
        assert len(body) == int(content_length)

    def test_media_preview_returns_valid_image(self, test_server):
        base_url, db_path, tmp_path = test_server
        photo_dir = tmp_path / "photos"
        photo_dir.mkdir()
        create_image(photo_dir / "test.jpg")
        preview_dir = tmp_path / "data" / "previews"

        from shotsieve.db import database
        with database(db_path) as connection:
            scan_root(
                connection,
                root=photo_dir,
                recursive=True,
                extensions=(".jpg",),
                preview_dir=preview_dir,
            )
            file_id = connection.execute("SELECT id FROM files LIMIT 1").fetchone()["id"]

        response = urlopen(f"{base_url}/api/media/preview?id={file_id}")
        body = response.read()
        image = Image.open(io.BytesIO(body))
        assert image.size[0] > 0 and image.size[1] > 0

    def test_media_preview_allows_custom_preview_directory_outside_default_root(self, test_server):
        base_url, db_path, tmp_path = test_server
        photo_dir = tmp_path / "photos"
        photo_dir.mkdir()
        create_image(photo_dir / "test.jpg")
        custom_preview_dir = tmp_path / "external-previews"

        from shotsieve.db import database
        with database(db_path) as connection:
            scan_root(
                connection,
                root=photo_dir,
                recursive=True,
                extensions=(".jpg",),
                preview_dir=custom_preview_dir,
            )
            file_id = connection.execute("SELECT id FROM files LIMIT 1").fetchone()["id"]

        response = urlopen(f"{base_url}/api/media/preview?id={file_id}")
        assert response.status == HTTPStatus.OK
        body = response.read()
        image = Image.open(io.BytesIO(body))
        assert image.size[0] > 0 and image.size[1] > 0

    def test_media_preview_allows_normalized_preview_hash_for_custom_preview_directory(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        from http.server import ThreadingHTTPServer

        monkeypatch.setattr(platform, "system", lambda: "Windows")

        db_path = tmp_path / "data" / "shotsieve.db"
        preview_dir = tmp_path / "external-previews"
        preview_dir.mkdir(parents=True)
        photo_dir = tmp_path / "Photos"
        photo_dir.mkdir()
        source_path = photo_dir / "Sample.jpg"
        create_image(source_path)

        normalized_preview_name = (
            f"{sha1(canonical_path_key(source_path).encode('utf-8')).hexdigest()}.jpg"
        )
        preview_path = preview_dir / normalized_preview_name
        create_image(preview_path)

        initialize_database(db_path)
        with database(db_path) as connection:
            cursor = connection.execute(
                """
                INSERT INTO files(path, path_key, preview_path, preview_status, scan_status)
                VALUES(?, ?, ?, 'ready', 'unchanged')
                """,
                (
                    str(source_path.resolve()),
                    canonical_path_key(source_path),
                    str(preview_path.resolve()),
                ),
            )
            file_id = cursor.lastrowid

        port = find_free_port()
        server = ThreadingHTTPServer(("127.0.0.1", port), build_handler(db_path))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            response = urlopen(f"http://127.0.0.1:{port}/api/media/preview?id={file_id}")
            assert response.status == HTTPStatus.OK
            body = response.read()
            image = Image.open(io.BytesIO(body))
            assert image.size[0] > 0 and image.size[1] > 0
        finally:
            server.shutdown()

    def test_media_preview_allows_windows_normalized_hash_after_case_sensitive_policy_switch(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        from http.server import ThreadingHTTPServer

        db_path = tmp_path / "data" / "shotsieve.db"
        preview_dir = tmp_path / "external-previews"
        preview_dir.mkdir(parents=True)
        photo_dir = tmp_path / "Photos"
        photo_dir.mkdir()
        source_path = photo_dir / "Sample.jpg"
        create_image(source_path)

        monkeypatch.setattr(platform, "system", lambda: "Windows")
        windows_normalized_preview_name = (
            f"{sha1(canonical_path_key(source_path).encode('utf-8')).hexdigest()}.jpg"
        )
        preview_path = preview_dir / windows_normalized_preview_name
        create_image(preview_path)

        initialize_database(db_path)
        with database(db_path) as connection:
            cursor = connection.execute(
                """
                INSERT INTO files(path, path_key, preview_path, preview_status, scan_status)
                VALUES(?, ?, ?, 'ready', 'unchanged')
                """,
                (
                    str(source_path.resolve()),
                    canonical_path_key(source_path),
                    str(preview_path.resolve()),
                ),
            )
            file_id = cursor.lastrowid

        monkeypatch.setattr(platform, "system", lambda: "Linux")

        port = find_free_port()
        server = ThreadingHTTPServer(("127.0.0.1", port), build_handler(db_path))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            response = urlopen(f"http://127.0.0.1:{port}/api/media/preview?id={file_id}")
            assert response.status == HTTPStatus.OK
            body = response.read()
            image = Image.open(io.BytesIO(body))
            assert image.size[0] > 0 and image.size[1] > 0
        finally:
            server.shutdown()

    def test_media_preview_rejects_ambiguous_fallback_hash_after_policy_switch(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        from http.server import ThreadingHTTPServer

        db_path = tmp_path / "data" / "shotsieve.db"
        preview_dir = tmp_path / "external-previews"
        preview_dir.mkdir(parents=True)

        primary_path = tmp_path / "Photos" / "Sample.jpg"
        secondary_path = tmp_path / "Photos" / "sample.jpg"
        shared_preview_name = (
            f"{sha1(str(primary_path.resolve()).casefold().encode('utf-8')).hexdigest()}.jpg"
        )
        preview_path = preview_dir / shared_preview_name
        create_image(preview_path)

        initialize_database(db_path)
        with database(db_path) as connection:
            connection.executemany(
                """
                INSERT INTO files(path, path_key, preview_path, preview_status, scan_status)
                VALUES(?, ?, ?, 'ready', 'unchanged')
                """,
                [
                    (str(primary_path.resolve()), str(primary_path.resolve()), str(preview_path.resolve())),
                    (str(secondary_path.resolve()), str(secondary_path.resolve()), str(preview_path.resolve())),
                ],
            )
            file_id = connection.execute("SELECT id FROM files ORDER BY id LIMIT 1").fetchone()["id"]

        monkeypatch.setattr(platform, "system", lambda: "Linux")

        port = find_free_port()
        server = ThreadingHTTPServer(("127.0.0.1", port), build_handler(db_path))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with pytest.raises(HTTPError) as exc_info:
                urlopen(f"http://127.0.0.1:{port}/api/media/preview?id={file_id}")
            assert exc_info.value.code == HTTPStatus.FORBIDDEN
        finally:
            server.shutdown()

    def test_media_source_uses_webp_content_type_fallback_when_guess_missing(self, test_server, monkeypatch):
        base_url, db_path, tmp_path = test_server
        photo_dir = tmp_path / "photos"
        photo_dir.mkdir()
        webp_path = photo_dir / "sample.webp"
        webp_path.write_bytes(b"RIFF\x00\x00\x00\x00WEBPVP8 ")
        preview_dir = tmp_path / "data" / "previews"

        from shotsieve import web as web_module
        from shotsieve.db import database

        monkeypatch.setattr(web_module.mimetypes, "guess_type", lambda _: (None, None))

        with database(db_path) as connection:
            scan_root(
                connection,
                root=photo_dir,
                recursive=True,
                extensions=(".webp",),
                preview_dir=preview_dir,
                generate_previews=False,
            )
            file_id = connection.execute("SELECT id FROM files LIMIT 1").fetchone()["id"]

        response = urlopen(f"{base_url}/api/media/source?id={file_id}")
        assert response.headers.get("Content-Type") == "image/webp"

