"""Regression tests for retained cleanup compatibility names."""
from __future__ import annotations

from shotsieve import preview as preview_module
from shotsieve import review_cache
from shotsieve import web as web_module
from shotsieve import web_request
from shotsieve.config import BROWSER_SAFE_EXTENSIONS


def test_preview_standard_wrapper_remains_a_compatibility_shim(monkeypatch) -> None:
    sentinel = object()

    def fake_prepare(image, *, apply_exif_orientation):
        assert image == "legacy-image"
        assert apply_exif_orientation is False
        return sentinel

    monkeypatch.setattr(preview_module, "prepare_image_for_rgb", fake_prepare)

    assert preview_module._prepare_standard_preview_image("legacy-image") is sentinel


def test_browser_safe_extensions_remains_a_compatibility_export() -> None:
    assert {".jpg", ".jpeg", ".png", ".webp"}.issubset(BROWSER_SAFE_EXTENSIONS)
    assert ".heic" not in BROWSER_SAFE_EXTENSIONS
    assert review_cache.BROWSER_SAFE_EXTENSIONS is BROWSER_SAFE_EXTENSIONS


def test_web_required_path_list_alias_remains_bound_to_request_helper() -> None:
    assert web_module._required_path_list is web_request.required_path_list
