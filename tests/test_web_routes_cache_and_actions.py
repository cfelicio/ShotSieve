from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

import pytest


def _captured_list(store: dict[str, object], key: str) -> list[object]:
    existing = store.get(key)
    if isinstance(existing, list):
        return existing
    bucket: list[object] = []
    store[key] = bucket
    return bucket


def test_cache_post_route_family_rejects_missing_scope(tmp_path: Path):
    from shotsieve import web_routes as route_module

    def required_choice(value, *, name, choices):
        if value not in choices:
            raise ValueError(f"{name} must be one of: {', '.join(choices)}")
        return value

    deps = SimpleNamespace(
        read_json_body=lambda _handler, *, max_body_size: {"scope": "missing"},
        required_choice=required_choice,
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

    with pytest.raises(ValueError, match="scope must be one of: scores, review, all"):
        route_module._handle_cache_post_routes(handler, context, urlparse(handler.path))


def test_missing_cache_preview_route_is_root_scoped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from shotsieve import web_routes as route_module

    captured: dict[str, object] = {}

    class _DatabaseContext:
        def __enter__(self):
            return object()

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_send_json(_handler, payload: object) -> None:
        captured["payload"] = payload

    def fake_preview(_connection, *, root: Path) -> dict[str, object]:
        captured["root"] = root
        return {"status": "empty", "root": str(root), "candidate_count": 0}

    monkeypatch.setattr(route_module, "send_json", fake_send_json)
    deps = SimpleNamespace(
        first_value=lambda params, key, default: params.get(key, [default])[0],
        optional_string=lambda value: value if isinstance(value, str) else None,
        database=lambda _path: _DatabaseContext(),
        preview_missing_cache_entries=fake_preview,
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
    handler = SimpleNamespace(
        path=f"/api/cache/missing/preview?root={str(tmp_path).replace(' ', '%20')}",
        headers={},
    )

    handled = route_module._handle_filesystem_get_routes(handler, context, urlparse(handler.path))

    assert handled is True
    assert captured["root"] == tmp_path.resolve()
    assert captured["payload"] == {"status": "empty", "root": str(tmp_path.resolve()), "candidate_count": 0}


def test_missing_cache_apply_route_passes_confirmation_token_and_ids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from shotsieve import web_routes as route_module

    captured: dict[str, object] = {}

    def fake_send_json(_handler, payload: object) -> None:
        captured["response"] = payload

    def fake_apply(context, payload):
        captured["context"] = context
        captured["payload"] = payload
        return {"status": "applied", "removed_count": 2}

    monkeypatch.setattr(route_module, "send_json", fake_send_json)
    monkeypatch.setattr(route_module, "_execute_missing_cache_apply_request", fake_apply)
    deps = SimpleNamespace(
        read_json_body=lambda _handler, *, max_body_size: {
            "root": str(tmp_path),
            "token": "preview-token",
            "candidate_ids": [4, 9],
        },
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
    handler = SimpleNamespace(path="/api/cache/missing/apply", headers={})

    handled = route_module._handle_cache_post_routes(handler, context, urlparse(handler.path))

    assert handled is True
    assert captured["context"] is context
    assert captured["payload"] == {
        "root": str(tmp_path),
        "token": "preview-token",
        "candidate_ids": [4, 9],
    }
    assert captured["response"] == {"status": "applied", "removed_count": 2}


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


def test_files_export_bulk_failure_retains_current_and_later_batch_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from shotsieve import web_routes as route_module
    from shotsieve.models import FileOperationResult, FileOperationSummary, attach_file_operation_summary

    connection = object()
    batches = [list(range(1, 501)), [501, 502], []]
    calls = 0

    def fake_list_review_browser_file_ids(_connection, **kwargs):
        return batches.pop(0)

    def fake_export_files(_connection, **kwargs):
        nonlocal calls
        calls += 1
        file_ids = list(kwargs["file_ids"])
        if calls == 1:
            result = FileOperationSummary(action="copy", contract_enabled=True)
            for file_id in file_ids:
                result.add(
                    FileOperationResult(
                        file_id=file_id,
                        source=f"photo-{file_id}.jpg",
                        destination=f"export/photo-{file_id}.jpg",
                        action="copy",
                        outcome="success",
                        stage="transfer",
                    )
                )
            result.copied = len(file_ids)
            return result

        result = FileOperationSummary(action="copy", contract_enabled=True)
        result.add(
            FileOperationResult(
                file_id=file_ids[0],
                source=f"photo-{file_ids[0]}.jpg",
                destination=f"export/photo-{file_ids[0]}.jpg",
                action="copy",
                outcome="failed",
                stage="transfer",
                error_text="simulated batch failure",
                retry_safe=False,
            )
        )
        error = RuntimeError("simulated batch failure")
        attach_file_operation_summary(error, result)
        raise error

    class _DatabaseContext:
        def __enter__(self):
            return connection

        def __exit__(self, exc_type, exc, tb):
            return False

    deps = SimpleNamespace(
        optional_string=lambda value: value if isinstance(value, str) else None,
        required_choice=lambda value, *, name, choices: value,
        float_or_none=lambda value: None if value is None else float(value),
        coerce_bool=lambda value, *, default: default if value is None else bool(value),
        database=lambda _path: _DatabaseContext(),
        review_selection_revision=lambda _connection, **kwargs: "rev-1",
        list_review_browser_file_ids=fake_list_review_browser_file_ids,
        get_preview_cache_root=lambda _connection, *, db_path, persist: tmp_path / "previews",
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

    with pytest.raises(RuntimeError, match="simulated batch failure") as exc_info:
        route_module._execute_export_request(
            context,
            {
                "selection": {
                    "scope": "review-browser",
                    "marked": "all",
                    "root": "C:/photos",
                },
                "selection_revision": "rev-1",
                "destination": str(tmp_path / "export"),
                "mode": "copy",
            },
            progress_callback=None,
            cancel_check=None,
        )

    summary = exc_info.value.file_operation_summary
    assert {item["file_id"] for item in summary["items"]} == set(range(1, 503))
    assert summary["completed_count"] == 500
    assert summary["failed_count"] == 1
    assert summary["unprocessed_count"] == 1


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
