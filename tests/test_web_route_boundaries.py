from __future__ import annotations

import inspect
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from shotsieve import web_route_common, web_route_files, web_route_jobs, web_route_review, web_route_scan
from shotsieve.job_registry import JobRegistry
from shotsieve.web import _build_route_context, _build_route_dependencies


def test_route_families_do_not_resolve_the_aggregator_through_sys_modules() -> None:
    for module in (web_route_common, web_route_files, web_route_jobs, web_route_review, web_route_scan):
        assert "_get_web_routes" not in inspect.getsource(module)
        assert 'sys.modules["shotsieve.web_routes"]' not in inspect.getsource(module)


def test_route_dependency_views_expose_only_family_dependencies() -> None:
    source = SimpleNamespace(
        database=lambda _path: None,
        score_files=lambda *_args, **_kwargs: None,
        review_overview=lambda *_args, **_kwargs: None,
    )

    views = web_route_common.build_route_dependency_views(source)

    assert views.files.database is source.database
    assert views.jobs.score_files is source.score_files
    assert views.review.review_overview is source.review_overview
    with pytest.raises(AttributeError, match="not a dependency"):
        _ = views.files.score_files


def test_handler_context_factory_keeps_legacy_dependency_container() -> None:
    db_path = Path("shotsieve.db")
    dependencies = _build_route_dependencies()
    context = _build_route_context(
        db_path,
        operation_lock=threading.Lock(),
        scan_registry=JobRegistry(max_jobs=1),
        score_registry=JobRegistry(max_jobs=1),
        compare_registry=JobRegistry(max_jobs=1),
        operation_registry=JobRegistry(max_jobs=1),
        model_registry=JobRegistry(max_jobs=1),
    )

    assert context.dependencies is not None
    assert type(context.dependencies) is type(dependencies)
    assert context.dependency_views.jobs.score_files is context.dependencies.score_files
