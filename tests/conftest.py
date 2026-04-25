"""Shared fixtures and helpers for web handler tests."""
from __future__ import annotations

import socket
import threading
from pathlib import Path

import pytest
from PIL import Image

from shotsieve.web import build_review_server


def _make_server(db_path: Path, port: int):
    return build_review_server(db_path, host="127.0.0.1", port=port)


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def read_socket_response(sock: socket.socket, *, timeout: float = 2.0) -> bytes:
    sock.settimeout(timeout)
    chunks: list[bytes] = []
    while True:
        try:
            chunk = sock.recv(4096)
        except socket.timeout:
            break
        if not chunk:
            break
        chunks.append(chunk)
    return b"".join(chunks)


@pytest.fixture()
def test_server(tmp_path: Path):
    """Start a ShotSieve server on a random port and return (base_url, db_path, tmp_path)."""
    db_path = tmp_path / "data" / "shotsieve.db"
    port = find_free_port()
    server = _make_server(db_path, port)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}", db_path, tmp_path
    server.shutdown()


def create_image(path: Path) -> None:
    image = Image.new("RGB", (120, 80), color=(40, 90, 160))
    image.save(path, format="JPEG")
