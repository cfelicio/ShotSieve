"""Tests for web route integration: dispatch, logging, and security headers."""
from __future__ import annotations

import json
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import pytest

from shotsieve.web import build_handler


class TestRouteHandlingIntegration:
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

    def test_unknown_get_route_returns_404(self, test_server):
        base_url, _, _ = test_server
        with pytest.raises(HTTPError) as exc_info:
            urlopen(f"{base_url}/api/nonexistent")
        assert exc_info.value.code == 404

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
