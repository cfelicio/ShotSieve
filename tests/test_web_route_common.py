"""Focused tests for shared HTTP response mechanics."""
from __future__ import annotations

import json
from http import HTTPStatus
from pathlib import Path

import pytest

from shotsieve.web_route_common import send_bytes, send_json, send_json_error, serve_static


class _RecordingWriter:
    def __init__(self, error: BaseException | None = None) -> None:
        self.body = bytearray()
        self.error = error

    def write(self, body: bytes) -> int:
        if self.error is not None:
            raise self.error
        self.body.extend(body)
        return len(body)


class _RecordingHandler:
    def __init__(self, writer: _RecordingWriter | None = None) -> None:
        self.status: HTTPStatus | None = None
        self.headers: list[tuple[str, str]] = []
        self.headers_ended = 0
        self.wfile = writer or _RecordingWriter()

    def send_response(self, status: HTTPStatus) -> None:
        self.status = status

    def send_header(self, name: str, value: str) -> None:
        self.headers.append((name, value))

    def end_headers(self) -> None:
        self.headers_ended += 1


def _header_map(handler: _RecordingHandler) -> dict[str, str]:
    return dict(handler.headers)


def test_serve_static_writes_body_and_preserves_cache_headers(tmp_path: Path) -> None:
    asset = tmp_path / "app.js"
    body = "const café = true;\n".encode("utf-8")
    asset.write_bytes(body)
    handler = _RecordingHandler()

    serve_static(handler, asset.name, "text/javascript; charset=utf-8", static_dir=tmp_path)

    assert handler.status == HTTPStatus.OK
    assert bytes(handler.wfile.body) == body
    assert handler.headers_ended == 1
    assert _header_map(handler) == {
        "Content-Type": "text/javascript; charset=utf-8",
        "Content-Length": str(len(body)),
        "Cache-Control": "no-cache, must-revalidate",
    }


def test_send_json_writes_utf8_body_without_cache_header() -> None:
    handler = _RecordingHandler()
    payload = {"message": "café", "ok": True}
    expected_body = json.dumps(payload).encode("utf-8")

    send_json(handler, payload)

    assert handler.status == HTTPStatus.OK
    assert bytes(handler.wfile.body) == expected_body
    assert handler.headers_ended == 1
    assert _header_map(handler) == {
        "Content-Type": "application/json; charset=utf-8",
        "Content-Length": str(len(expected_body)),
    }


def test_send_bytes_writes_download_headers() -> None:
    handler = _RecordingHandler()
    body = b"csv,data\r\n"

    send_bytes(
        handler,
        body,
        content_type="text/csv; charset=utf-8",
        download_name="decisions.csv",
    )

    assert handler.status == HTTPStatus.OK
    assert bytes(handler.wfile.body) == body
    assert handler.headers_ended == 1
    assert _header_map(handler) == {
        "Content-Type": "text/csv; charset=utf-8",
        "Content-Length": str(len(body)),
        "Cache-Control": "no-cache, must-revalidate",
        "Content-Disposition": 'attachment; filename="decisions.csv"',
    }


def test_send_json_error_writes_status_and_error_body_without_cache_header() -> None:
    handler = _RecordingHandler()
    expected_body = json.dumps({"error": "bad request"}).encode("utf-8")

    send_json_error(handler, HTTPStatus.BAD_REQUEST, "bad request")

    assert handler.status == HTTPStatus.BAD_REQUEST
    assert bytes(handler.wfile.body) == expected_body
    assert handler.headers_ended == 1
    assert _header_map(handler) == {
        "Content-Type": "application/json; charset=utf-8",
        "Content-Length": str(len(expected_body)),
    }


def test_response_helpers_ignore_client_disconnects(tmp_path: Path) -> None:
    asset = tmp_path / "asset.css"
    asset.write_bytes(b"body {}")
    cases = (
        lambda handler: serve_static(handler, asset.name, "text/css", static_dir=tmp_path),
        lambda handler: send_json(handler, {"ok": True}),
        lambda handler: send_bytes(handler, b"bytes", content_type="application/octet-stream"),
        lambda handler: send_json_error(handler, HTTPStatus.BAD_REQUEST, "bad request"),
    )

    for send_response in cases:
        handler = _RecordingHandler(_RecordingWriter(BrokenPipeError()))
        send_response(handler)
        assert handler.headers_ended == 1


def test_response_helpers_reraise_non_disconnect_write_errors() -> None:
    handler = _RecordingHandler(_RecordingWriter(OSError("disk full")))

    with pytest.raises(OSError, match="disk full"):
        send_json(handler, {"ok": True})
