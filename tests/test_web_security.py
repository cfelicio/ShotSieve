"""Tests for local security guards: origin, loopback, and path boundary checks."""
from __future__ import annotations

import json
from http import HTTPStatus
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import pytest

from shotsieve.scanner import scan_root
from shotsieve.web import _is_allowed_post_origin, _is_loopback_host, _is_within_any_root

from conftest import create_image


class TestLocalSecurityGuards:
    def test_security_helper_module_matches_web_guard_behavior(self, tmp_path: Path) -> None:
        from shotsieve.web_security import (
            is_allowed_post_origin,
            is_loopback_host,
            is_within_any_root,
        )

        known_root = (tmp_path / "photos").resolve()
        descendant = (known_root / "nested" / "a.jpg").resolve()
        sibling_prefix = (tmp_path / "photos-archive" / "a.jpg").resolve()

        assert is_loopback_host("127.0.0.1") is True
        assert is_loopback_host("example.com") is False
        assert is_within_any_root(descendant, [known_root]) is True
        assert is_within_any_root(sibling_prefix, [known_root]) is False
        assert is_allowed_post_origin("http://127.0.0.1:8765", "127.0.0.1:8765") is True
        assert is_allowed_post_origin("https://evil.example", "127.0.0.1:8765") is False

    def test_request_helper_module_parses_scan_and_compare_payloads(self, tmp_path: Path) -> None:
        from shotsieve.web_request import parse_compare_request, parse_scan_request

        photo_dir = tmp_path / "photos"
        photo_dir.mkdir()

        scan_request = parse_scan_request({
            "roots": [str(photo_dir)],
            "recursive": False,
            "files_total_hint": 12,
        })
        compare_request = parse_compare_request({
            "models": [" topiq_nr ", "clipiqa"],
            "offset": 2,
        }, default_batch_size=8)

        assert scan_request["roots"] == [photo_dir.resolve()]
        assert scan_request["recursive"] is False
        assert scan_request["files_total_hint"] == 12
        assert compare_request["models"] == ["topiq_nr", "clipiqa"]
        assert compare_request["offset"] == 2
        assert compare_request["batch_size"] == 8

    def test_loopback_host_detection_handles_ipv4_ipv6_and_localhost(self) -> None:
        assert _is_loopback_host("127.0.0.1") is True
        assert _is_loopback_host("::1") is True
        assert _is_loopback_host("::ffff:127.0.0.1") is True
        assert _is_loopback_host("localhost") is True
        assert _is_loopback_host("192.168.1.40") is False
        assert _is_loopback_host("example.com") is False

    def test_known_root_boundary_allows_descendant_path(self, tmp_path: Path) -> None:
        known_root = (tmp_path / "photos").resolve()
        candidate = (known_root / "nested" / "a.jpg").resolve()

        assert _is_within_any_root(candidate, [known_root]) is True

    def test_known_root_boundary_rejects_prefix_collision_path(self, tmp_path: Path) -> None:
        known_root = (tmp_path / "photos").resolve()
        sibling_prefix = (tmp_path / "photos-archive" / "a.jpg").resolve()

        assert _is_within_any_root(sibling_prefix, [known_root]) is False

    def test_cross_origin_post_is_rejected(self, test_server):
        base_url, _, _ = test_server
        req = Request(
            f"{base_url}/api/nonexistent",
            data=b'{"test": true}',
            headers={"Content-Type": "application/json", "Origin": "https://evil.example"},
            method="POST",
        )
        with pytest.raises(HTTPError) as exc_info:
            urlopen(req)

        assert exc_info.value.code == HTTPStatus.FORBIDDEN
        assert "origin" in exc_info.value.read().decode("utf-8").lower()

    def test_loopback_origin_header_is_allowed(self, test_server):
        base_url, _, _ = test_server
        req = Request(
            f"{base_url}/api/nonexistent",
            data=b'{"test": true}',
            headers={"Content-Type": "application/json", "Origin": base_url},
            method="POST",
        )
        with pytest.raises(HTTPError) as exc_info:
            urlopen(req)

        assert exc_info.value.code == HTTPStatus.NOT_FOUND

    def test_post_origin_guard_uses_web_module_override_after_server_start(self, test_server, monkeypatch):
        base_url, _, _ = test_server
        from shotsieve import web as web_module

        monkeypatch.setattr(web_module, "_is_allowed_post_origin", lambda *_args, **_kwargs: True)

        req = Request(
            f"{base_url}/api/nonexistent",
            data=b'{"test": true}',
            headers={"Content-Type": "application/json", "Origin": "https://evil.example"},
            method="POST",
        )
        with pytest.raises(HTTPError) as exc_info:
            urlopen(req)

        assert exc_info.value.code == HTTPStatus.NOT_FOUND

    def test_media_root_guard_uses_web_module_override_after_server_start(self, test_server, monkeypatch):
        base_url, db_path, tmp_path = test_server
        photo_dir = tmp_path / "photos"
        photo_dir.mkdir()
        create_image(photo_dir / "sample.jpg")
        preview_dir = tmp_path / "data" / "previews"

        from shotsieve import web as web_module
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

        monkeypatch.setattr(web_module, "_is_within_any_root", lambda *_args, **_kwargs: False)

        with pytest.raises(HTTPError) as exc_info:
            urlopen(f"{base_url}/api/media/source?id={file_id}")

        assert exc_info.value.code == HTTPStatus.FORBIDDEN

    def test_fs_list_route_uses_web_module_path_override_after_server_start(self, test_server, monkeypatch):
        base_url, _, tmp_path = test_server
        from shotsieve import web as web_module

        browse_root = tmp_path / "browse-root"
        child = browse_root / "nested"
        child.mkdir(parents=True)

        monkeypatch.setattr(web_module, "_required_path", lambda *_args, **_kwargs: browse_root.resolve())

        payload = json.loads(urlopen(f"{base_url}/api/fs/list?path=not-a-real-directory").read().decode("utf-8"))

        assert payload["path"] == str(browse_root.resolve())
        assert payload["items"] == [{"name": "nested", "path": str(child.resolve())}]

    def test_origin_default_http_port_mismatch_is_rejected(self) -> None:
        assert _is_allowed_post_origin("http://127.0.0.1", "127.0.0.1:8765") is False

    def test_origin_default_https_port_mismatch_is_rejected(self) -> None:
        assert _is_allowed_post_origin("https://localhost", "localhost:8765") is False
