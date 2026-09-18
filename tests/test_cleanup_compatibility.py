"""Regression tests for retained cleanup boundaries."""
from __future__ import annotations

from shotsieve import review_cache
from shotsieve import web as web_module
from shotsieve import web_request
from shotsieve.config import BROWSER_SAFE_EXTENSIONS


def test_browser_safe_extensions_remains_a_compatibility_export() -> None:
    assert {".jpg", ".jpeg", ".png", ".webp"}.issubset(BROWSER_SAFE_EXTENSIONS)
    assert ".heic" not in BROWSER_SAFE_EXTENSIONS
    assert review_cache.BROWSER_SAFE_EXTENSIONS is BROWSER_SAFE_EXTENSIONS


def test_web_required_path_list_alias_remains_bound_to_request_helper() -> None:
    assert web_module._required_path_list is web_request.required_path_list
