from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "model_smoke.py"


def load_smoke_module():
    spec = importlib.util.spec_from_file_location("shotsieve_model_smoke", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_model_smoke_retains_sanitized_failure_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_smoke_module()
    report_path = tmp_path / "reports" / "failure.json"

    def fail_prepare(*_args, **_kwargs):
        raise RuntimeError(
            "Hub request failed for https://example.test/download?token=private-secret "
            "while reading C:/private/photos/image.jpg"
        )

    from shotsieve import model_assets

    monkeypatch.setattr(model_assets, "apply_model_cache_dir", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(model_assets, "prepare_model", fail_prepare)
    monkeypatch.setattr(module, "installed_model_dependency_versions", lambda: {"torch": "2.13.0"})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT_PATH),
            "--model",
            "topiq_nr",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--data-dir",
            str(tmp_path / "data"),
            "--report-path",
            str(report_path),
        ],
    )

    with pytest.raises(SystemExit) as raised:
        module.main()

    assert raised.value.code == 1
    report = json.loads(report_path.read_text(encoding="utf-8"))
    serialized = json.dumps(report)
    assert report["status"] == "failed"
    assert report["diagnostic"]["category"] == "network_or_hub"
    assert "private-secret" not in serialized
    assert "photos/image.jpg" not in serialized
    assert "?token=" not in serialized
