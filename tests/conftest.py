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


def _create_test_image(path: Path, *, color: tuple[int, int, int]) -> None:
    image = Image.new("RGB", (160, 120), color=color)
    image.save(path, format="JPEG")


def _build_frontend_server(tmp_path: Path, *, filenames: list[str], issue_filename: str | None = None):
    from shotsieve.db import database, initialize_database
    from shotsieve.learned_iqa import LearnedScoreResult
    from shotsieve.scanner import scan_root
    from shotsieve.scoring import score_files

    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "previews"
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir(parents=True)

    for index, filename in enumerate(filenames):
        _create_test_image(
            photo_dir / filename,
            color=((40 + index * 17) % 255, (90 + index * 31) % 255, (160 + index * 13) % 255),
        )

    initialize_database(db_path)

    class FakeLearnedBackend:
        name = "topiq_nr"
        model_version = "fake:test"

        def score_paths(self, image_paths, *, batch_size: int = 4, resource_profile: str | None = None):
            return [
                LearnedScoreResult(raw_score=0.82, normalized_score=82.0, confidence=91.0)
                for _ in image_paths
            ]

    with database(db_path) as connection:
        scan_root(
            connection,
            root=photo_dir,
            recursive=True,
            extensions=(".jpg",),
            preview_dir=preview_dir,
        )
        score_files(
            connection,
            learned_backend_name="topiq_nr",
            learned_backend_factory=lambda model_name: FakeLearnedBackend(),
        )
        if issue_filename:
            connection.execute(
                """
                UPDATE files
                   SET last_error = ?
                 WHERE path LIKE ?
                """,
                ("data corruption detected while decoding preview", f"%{issue_filename}"),
            )

    server = build_review_server(db_path, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _wait_for_shell_ready(page, timeout: float = 60000) -> None:
    page.wait_for_function(
        """
        () => {
          const modelOptions = document.querySelectorAll('#model-select option').length;
          const deviceOptions = document.querySelectorAll('#device-select option').length;
          return modelOptions >= 1 && deviceOptions >= 1;
        }
        """,
        timeout=timeout,
    )


@pytest.fixture()
def frontend_server(tmp_path: Path):
    server, thread = _build_frontend_server(
        tmp_path,
        filenames=["one.jpg", "two.jpg", "three.jpg"],
        issue_filename="three.jpg",
    )
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture()
def frontend_large_server(tmp_path: Path):
    server, thread = _build_frontend_server(
        tmp_path,
        filenames=[f"{index:03d}.jpg" for index in range(1, 66)],
    )
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture()
def chromium_page(frontend_server: str):
    playwright = pytest.importorskip("playwright.sync_api")
    expect = playwright.expect

    with playwright.sync_playwright() as runner:
        try:
            browser = runner.chromium.launch(headless=True)
        except Exception as exc:  # pragma: no cover - environment-dependent skip path
            pytest.skip(f"Playwright browser unavailable: {exc}")

        try:
            page = browser.new_page()
            page.goto(frontend_server)
            page.wait_for_selector("#tab-workspace-button")
            _wait_for_shell_ready(page)
            yield page, expect
        finally:
            browser.close()


@pytest.fixture()
def mobile_chromium_page(frontend_server: str):
    playwright = pytest.importorskip("playwright.sync_api")
    expect = playwright.expect

    with playwright.sync_playwright() as runner:
        try:
            browser = runner.chromium.launch(headless=True)
        except Exception as exc:  # pragma: no cover - environment-dependent skip path
            pytest.skip(f"Playwright browser unavailable: {exc}")

        context = None
        try:
            context = browser.new_context(**runner.devices["iPhone 13"])
            page = context.new_page()
            page.goto(frontend_server)
            page.wait_for_selector("#tab-workspace-button")
            _wait_for_shell_ready(page)
            yield page, expect
        finally:
            if context is not None:
                context.close()
            browser.close()


@pytest.fixture()
def large_chromium_page(frontend_large_server: str):
    playwright = pytest.importorskip("playwright.sync_api")
    expect = playwright.expect

    with playwright.sync_playwright() as runner:
        try:
            browser = runner.chromium.launch(headless=True)
        except Exception as exc:  # pragma: no cover - environment-dependent skip path
            pytest.skip(f"Playwright browser unavailable: {exc}")

        try:
            page = browser.new_page()
            page.goto(frontend_large_server)
            page.wait_for_selector("#tab-workspace-button")
            _wait_for_shell_ready(page)
            yield page, expect
        finally:
            browser.close()


@pytest.fixture()
def scoped_chromium_page(tmp_path: Path):
    from shotsieve.db import database, initialize_database
    from shotsieve.learned_iqa import LearnedScoreResult
    from shotsieve.scanner import scan_root
    from shotsieve.scoring import score_files
    from shotsieve.web import build_review_server

    playwright = pytest.importorskip("playwright.sync_api")
    expect = playwright.expect
    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "previews"
    root_a = tmp_path / "library-a"
    root_b = tmp_path / "library-b"
    root_a.mkdir(parents=True)
    root_b.mkdir(parents=True)
    for index in range(61):
        _create_test_image(root_a / f"a-{index:03d}.jpg", color=(80, 120, 160))
    _create_test_image(root_b / "b-001.jpg", color=(120, 80, 160))

    class FakeLearnedBackend:
        name = "topiq_nr"
        model_version = "fake:scope-test"

        def score_paths(self, image_paths, *, batch_size: int = 4, resource_profile: str | None = None):
            return [
                LearnedScoreResult(raw_score=0.82, normalized_score=82.0, confidence=91.0)
                for _ in image_paths
            ]

    initialize_database(db_path)

    with database(db_path) as connection:
        scan_root(connection, root=root_a, recursive=True, extensions=(".jpg",), preview_dir=preview_dir)
        scan_root(connection, root=root_b, recursive=True, extensions=(".jpg",), preview_dir=preview_dir)
        score_files(
            connection,
            learned_backend_name="topiq_nr",
            learned_backend_factory=lambda _model_name: FakeLearnedBackend(),
        )

    server = build_review_server(db_path, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with playwright.sync_playwright() as runner:
            try:
                browser = runner.chromium.launch(headless=True)
            except Exception as exc:  # pragma: no cover - environment-dependent skip path
                pytest.skip(f"Playwright browser unavailable: {exc}")

            try:
                page = browser.new_page()
                page.goto(f"http://127.0.0.1:{server.server_port}")
                page.wait_for_selector("#tab-workspace-button")
                _wait_for_shell_ready(page)
                yield page, expect, str(root_a.resolve()), str(root_b.resolve())
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

