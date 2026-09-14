from __future__ import annotations

import importlib.util
import json
import sys
import types
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


def test_runtime_evidence_captures_xpu_identity_and_peak_memory() -> None:
    module = load_smoke_module()

    class FakeXpu:
        def synchronize(self):
            return None

        def max_memory_allocated(self):
            return 2 * 1024 * 1024

        def device_count(self):
            return 1

        def get_device_name(self, index):
            assert index == 0
            return "Intel Arc test device"

    fake_torch = types.SimpleNamespace(__version__="2.14.0+xpu", xpu=FakeXpu())

    evidence = module._runtime_evidence(
        measurement={"torch_module": fake_torch},
        requested_runtime="xpu",
        actual_runtime="xpu",
        driver_version="test-driver",
        elapsed_seconds=1.23456,
    )

    assert evidence["torch_runtime"] == "2.14.0+xpu"
    assert evidence["xpu_device_count"] == 1
    assert evidence["xpu_device_name"] == "Intel Arc test device"
    assert evidence["peak_memory_mb"] == 2.0
    assert evidence["driver_version"] == "test-driver"
    assert evidence["elapsed_seconds"] == 1.235


def test_runtime_evidence_captures_rocm_identity_and_hip_version() -> None:
    module = load_smoke_module()

    class FakeProperties:
        gcnArchName = "gfx1100"

    class FakeCuda:
        def synchronize(self):
            return None

        def max_memory_allocated(self):
            return 3 * 1024 * 1024

        def device_count(self):
            return 1

        def get_device_name(self, index):
            assert index == 0
            return "Radeon RX test device"

        def get_device_properties(self, index):
            assert index == 0
            return FakeProperties()

    fake_torch = types.SimpleNamespace(
        __version__="2.9.1+rocm7.2.1",
        version=types.SimpleNamespace(hip="7.2.1"),
        cuda=FakeCuda(),
    )

    evidence = module._runtime_evidence(
        measurement={"torch_module": fake_torch},
        requested_runtime="rocm",
        actual_runtime="rocm",
        driver_version="26.2.2",
        elapsed_seconds=2.5,
    )

    assert evidence["torch_runtime"] == "2.9.1+rocm7.2.1"
    assert evidence["rocm_version"] == "7.2.1"
    assert evidence["rocm_device_count"] == 1
    assert evidence["rocm_device_name"] == "Radeon RX test device"
    assert evidence["rocm_gpu_architecture"] == "gfx1100"
    assert evidence["peak_memory_mb"] == 3.0
