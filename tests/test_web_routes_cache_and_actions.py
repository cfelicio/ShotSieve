from __future__ import annotations

import threading

from types import SimpleNamespace
from pathlib import Path
from urllib.parse import urlparse

import pytest


def _captured_list(store: dict[str, object], key: str) -> list[object]:
    existing = store.get(key)
    if isinstance(existing, list):
        return existing
    bucket: list[object] = []
    store[key] = bucket
    return bucket


def test_cache_post_route_family_handles_missing_scope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
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


def test_files_delete_route_accepts_review_state_selection_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
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

    batches = [[11, 12], [13], []]

    def fake_list_review_state_file_ids(_connection, **kwargs):
        _captured_list(captured, "selection_calls").append(dict(kwargs))
        return batches.pop(0)

    def fake_delete_files(_connection, **kwargs):
        _captured_list(captured, "deleted_batches").append(list(kwargs["file_ids"]))
        ids = list(kwargs["file_ids"])
        return {
            "deleted_ids": ids,
            "deleted_count": len(ids),
            "failed": [],
            "failed_count": 0,
            "delete_from_disk": kwargs["delete_from_disk"],
        }

    monkeypatch.setattr(route_module, "send_json", fake_send_json)

    deps = SimpleNamespace(
        read_json_body=lambda _handler, *, max_body_size: {
            "selection": {
                "scope": "review-state",
                "marked": "delete",
                "root": str(tmp_path / "library"),
            },
            "selection_revision": "rev-1",
            "delete_from_disk": True,
        },
        required_int_list=lambda value, *, name: (_ for _ in ()).throw(AssertionError("file_ids should not be required for filter selections")),
        required_choice=lambda value, *, name, choices: value if value in choices else (_ for _ in ()).throw(ValueError(name)),
        optional_string=lambda value: value if isinstance(value, str) else None,
        coerce_bool=lambda value, *, default: default if value is None else bool(value),
        database=lambda _path: _DatabaseContext(),
        review_selection_revision=lambda _connection, **kwargs: "rev-1",
        list_review_state_file_ids=fake_list_review_state_file_ids,
        get_preview_cache_root=lambda _connection, *, db_path, persist: preview_root,
        delete_files=fake_delete_files,
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
    handler = SimpleNamespace(path="/api/files/delete", headers={"Content-Length": "20"})

    handled = route_module._handle_file_action_post_routes(handler, context, urlparse(handler.path))

    assert handled is True
    assert captured["deleted_batches"] == [[11, 12], [13]]
    assert captured["payload"] == {
        "deleted_ids": [11, 12, 13],
        "deleted_count": 3,
        "failed": [],
        "failed_count": 0,
        "delete_from_disk": True,
    }


def test_files_delete_route_rejects_review_state_selection_without_root(tmp_path: Path):
    from shotsieve import web_routes as route_module

    deps = SimpleNamespace(
        read_json_body=lambda _handler, *, max_body_size: {
            "selection": {
                "scope": "review-state",
                "marked": "delete",
            },
            "selection_revision": "rev-1",
            "delete_from_disk": True,
        },
        required_int_list=lambda value, *, name: (_ for _ in ()).throw(AssertionError("file_ids should not be required for selection payloads")),
        required_choice=lambda value, *, name, choices: value if value in choices else (_ for _ in ()).throw(ValueError(name)),
        optional_string=lambda value: value if isinstance(value, str) else None,
        coerce_bool=lambda value, *, default: default if value is None else bool(value),
        database=lambda _path: (_ for _ in ()).throw(AssertionError("database should not be opened after invalid selection")),
        review_selection_revision=lambda _connection, **kwargs: "rev-1",
        list_review_state_file_ids=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("selection should not be materialized after invalid selection")),
        get_preview_cache_root=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("preview cache should not be queried after invalid selection")),
        delete_files=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("delete_files should not be called after invalid selection")),
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
    handler = SimpleNamespace(path="/api/files/delete", headers={"Content-Length": "20"})

    with pytest.raises(ValueError, match="selection.root is required"):
        route_module._handle_file_action_post_routes(handler, context, urlparse(handler.path))


def test_files_export_route_accepts_review_browser_selection_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
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

    batches = [[21, 22], [23], []]

    def fake_list_review_browser_file_ids(_connection, **kwargs):
        _captured_list(captured, "selection_calls").append(dict(kwargs))
        return batches.pop(0)

    def fake_export_files(_connection, **kwargs):
        _captured_list(captured, "export_batches").append(list(kwargs["file_ids"]))
        ids = list(kwargs["file_ids"])
        return SimpleNamespace(copied=len(ids), moved=0, failed=[])

    monkeypatch.setattr(route_module, "send_json", fake_send_json)

    deps = SimpleNamespace(
        read_json_body=lambda _handler, *, max_body_size: {
            "selection": {
                "scope": "review-browser",
                "marked": "all",
                "root": "C:/photos",
                "query": "keepers",
            },
            "selection_revision": "rev-1",
            "destination": str(tmp_path / "export"),
            "mode": "copy",
        },
        required_int_list=lambda value, *, name: (_ for _ in ()).throw(AssertionError("file_ids should not be required for filter selections")),
        required_choice=lambda value, *, name, choices: value if value in choices else (_ for _ in ()).throw(ValueError(name)),
        optional_string=lambda value: value if isinstance(value, str) else None,
        float_or_none=lambda value: None if value is None else float(value),
        database=lambda _path: _DatabaseContext(),
        review_selection_revision=lambda _connection, **kwargs: "rev-1",
        list_review_browser_file_ids=fake_list_review_browser_file_ids,
        get_preview_cache_root=lambda _connection, *, db_path, persist: preview_root,
        export_files=fake_export_files,
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
    handler = SimpleNamespace(path="/api/files/export", headers={"Content-Length": "20"})

    handled = route_module._handle_file_action_post_routes(handler, context, urlparse(handler.path))

    assert handled is True
    assert captured["export_batches"] == [[21, 22], [23]]
    assert captured["payload"] == {"copied": 3, "moved": 0, "failed": []}


def test_files_delete_route_accepts_file_ids_with_matching_page_revision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
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

    def fake_delete_files(_connection, **kwargs):
        ids = list(kwargs["file_ids"])
        return {
            "deleted_ids": ids,
            "deleted_count": len(ids),
            "failed": [],
            "failed_count": 0,
            "delete_from_disk": kwargs["delete_from_disk"],
        }

    monkeypatch.setattr(route_module, "send_json", fake_send_json)

    deps = SimpleNamespace(
        read_json_body=lambda _handler, *, max_body_size: {
            "file_ids": [1, 2, 3],
            "delete_from_disk": True,
            "selection_revision": "rev-match",
            "page_selection": {
                "scope": "review-browser",
                "marked": "all",
                "root": "C:/photos",
            },
        },
        required_int_list=lambda value, *, name: list(value),
        required_choice=lambda value, *, name, choices: value if value in choices else (_ for _ in ()).throw(ValueError(name)),
        optional_string=lambda value: value if isinstance(value, str) else None,
        coerce_bool=lambda value, *, default: default if value is None else bool(value),
        database=lambda _path: _DatabaseContext(),
        review_selection_revision=lambda _connection, **kwargs: "rev-match",
        get_preview_cache_root=lambda _connection, *, db_path, persist: preview_root,
        delete_files=fake_delete_files,
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
    handler = SimpleNamespace(path="/api/files/delete", headers={"Content-Length": "20"})

    handled = route_module._handle_file_action_post_routes(handler, context, urlparse(handler.path))

    assert handled is True
    result = captured["payload"]
    assert result["deleted_count"] == 3


def test_files_delete_route_rejects_file_ids_with_mismatched_page_revision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from shotsieve import web_routes as route_module

    connection = object()
    preview_root = (tmp_path / "previews").resolve()

    class _DatabaseContext:
        def __enter__(self):
            return connection

        def __exit__(self, exc_type, exc, tb):
            return False

    deps = SimpleNamespace(
        read_json_body=lambda _handler, *, max_body_size: {
            "file_ids": [1, 2],
            "delete_from_disk": True,
            "selection_revision": "rev-stale",
            "page_selection": {
                "scope": "review-browser",
                "marked": "all",
                "root": "C:/photos",
            },
        },
        required_int_list=lambda value, *, name: list(value),
        required_choice=lambda value, *, name, choices: value if value in choices else (_ for _ in ()).throw(ValueError(name)),
        optional_string=lambda value: value if isinstance(value, str) else None,
        coerce_bool=lambda value, *, default: default if value is None else bool(value),
        database=lambda _path: _DatabaseContext(),
        review_selection_revision=lambda _connection, **kwargs: "rev-current",
        get_preview_cache_root=lambda _connection, *, db_path, persist: preview_root,
        delete_files=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("delete_files should not be called with stale revision")),
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
    handler = SimpleNamespace(path="/api/files/delete", headers={"Content-Length": "20"})

    with pytest.raises(ValueError, match="Selected results changed"):
        route_module._handle_file_action_post_routes(handler, context, urlparse(handler.path))


def test_files_delete_route_rejects_file_ids_without_page_revision(tmp_path: Path):
    from shotsieve import web_routes as route_module

    connection = object()
    preview_root = (tmp_path / "previews").resolve()

    class _DatabaseContext:
        def __enter__(self):
            return connection

        def __exit__(self, exc_type, exc, tb):
            return False

    deps = SimpleNamespace(
        read_json_body=lambda _handler, *, max_body_size: {
            "file_ids": [1, 2],
            "delete_from_disk": True,
        },
        required_int_list=lambda value, *, name: list(value),
        required_choice=lambda value, *, name, choices: value if value in choices else (_ for _ in ()).throw(ValueError(name)),
        optional_string=lambda value: value if isinstance(value, str) else None,
        coerce_bool=lambda value, *, default: default if value is None else bool(value),
        database=lambda _path: _DatabaseContext(),
        review_selection_revision=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("revision should not be computed without page_selection")),
        get_preview_cache_root=lambda _connection, *, db_path, persist: preview_root,
        delete_files=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("delete_files should not be called without revision")),
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
    handler = SimpleNamespace(path="/api/files/delete", headers={"Content-Length": "20"})

    with pytest.raises(ValueError, match="selection_revision is required"):
        route_module._handle_file_action_post_routes(handler, context, urlparse(handler.path))


def test_files_delete_route_rejects_review_browser_selection_without_root(tmp_path: Path):
    from shotsieve import web_routes as route_module

    connection = object()

    class _DatabaseContext:
        def __enter__(self):
            return connection

        def __exit__(self, exc_type, exc, tb):
            return False

    deps = SimpleNamespace(
        read_json_body=lambda _handler, *, max_body_size: {
            "selection": {
                "scope": "review-browser",
                "marked": "all",
                "issues": "all",
            },
            "selection_revision": "rev-1",
            "delete_from_disk": True,
        },
        required_int_list=lambda value, *, name: (_ for _ in ()).throw(AssertionError("file_ids should not be required for selection payloads")),
        required_choice=lambda value, *, name, choices: value if value in choices else (_ for _ in ()).throw(ValueError(name)),
        optional_string=lambda value: value if isinstance(value, str) else None,
        coerce_bool=lambda value, *, default: default if value is None else bool(value),
        database=lambda _path: _DatabaseContext(),
        review_selection_revision=lambda _connection, **kwargs: "rev-1",
        get_preview_cache_root=lambda _connection, *, db_path, persist: (_ for _ in ()).throw(AssertionError("preview cache should not be queried after invalid selection")),
        delete_files=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("delete_files should not be called after invalid selection")),
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
    handler = SimpleNamespace(path="/api/files/delete", headers={"Content-Length": "20"})

    with pytest.raises(ValueError, match="selection.root is required for destructive bulk operations"):
        route_module._handle_file_action_post_routes(handler, context, urlparse(handler.path))
