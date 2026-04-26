from __future__ import annotations

import threading
from pathlib import Path
from urllib.request import urlopen

import pytest

from shotsieve.web import build_review_server


def _find_free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture()
def frontend_server(tmp_path: Path):
    db_path = tmp_path / "data" / "shotsieve.db"
    port = _find_free_port()
    server = build_review_server(db_path, host="127.0.0.1", port=port)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_reset_everything_clears_persisted_ui_state(frontend_server: str) -> None:
    state_body = urlopen(f"{frontend_server}/app-state.js").read().decode("utf-8")
    events_body = urlopen(f"{frontend_server}/app-events.js").read().decode("utf-8")

    assert "function clearUiState(" in state_body
    assert "storage.removeItem(UI_STATE_KEY);" in state_body
    assert "clearUiState({ immediate: true });" in events_body
    assert '"clear-all-cache"' in events_body
    assert "onSuccess: resetPersistedUiStateAfterFullReset" in events_body
    assert 'localStorage.removeItem("shotsieve_resource_profile");' in events_body
    assert 'profileSelect.value = "normal";' in events_body
    assert 'previewModeSelect.value = state.options?.default_preview_mode || "auto";' in events_body


def test_ui_state_is_scoped_to_database_marker(frontend_server: str) -> None:
    state_body = urlopen(f"{frontend_server}/app-state.js").read().decode("utf-8")
    app_body = urlopen(f"{frontend_server}/app.js").read().decode("utf-8")

    assert "function currentDatabaseMarker()" in state_body
    assert "documentRef.body?.dataset?.databasePath" in state_body
    assert "if (!savedDatabase || savedDatabase !== expectedDatabase)" in state_body
    assert "database: currentDatabaseMarker()," in state_body
    assert "document.body.dataset.databasePath = options.database || \"\";" in app_body
