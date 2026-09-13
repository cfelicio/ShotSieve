from __future__ import annotations

import errno
import json
from pathlib import Path

import pytest

from shotsieve import model_assets


class _Result:
    failed = False
    error = None


class _Backend:
    runtime = "cpu"

    def __init__(self, calls: list[tuple[str, object]]) -> None:
        self.calls = calls

    def score_paths(self, paths, *, batch_size, resource_profile):
        self.calls.append(("score", (list(paths), batch_size, resource_profile)))
        return [_Result()]

    def close(self) -> None:
        self.calls.append(("close", None))


def test_prepare_model_writes_atomic_success_record_and_releases_backend(tmp_path: Path) -> None:
    calls: list[tuple[str, object]] = []

    def factory(model: str, *, device: str):
        calls.append(("build", (model, device)))
        return _Backend(calls)

    result = model_assets.prepare_model("topiq-nr", data_dir=tmp_path, backend_factory=factory)

    assert result["state"] == "prepared"
    assert result["model"] == "topiq_nr"
    assert result["requested_runtime"] == "cpu"
    assert result["tested_runtime"] == "cpu"
    assert result["processed_counts"] == {"validation_images": 1}
    assert calls[0] == ("build", ("topiq_nr", "cpu"))
    assert calls[-1] == ("close", None)

    stored = json.loads(model_assets.preparation_record_path(tmp_path).read_text(encoding="utf-8"))
    assert stored["state"] == "prepared"
    assert stored["expected_resources"]["cache_families"]
    assert stored["disk_estimate"]["status"] == "advisory"


def test_prepare_model_failure_persists_diagnostic_after_validation_work(tmp_path: Path) -> None:
    calls: list[tuple[str, object]] = []

    class FailingBackend(_Backend):
        def score_paths(self, paths, *, batch_size, resource_profile):
            calls.append(("score", list(paths)))
            error = OSError(errno.ENOSPC, "No space left on device")
            raise error

    def factory(model: str, *, device: str):
        calls.append(("build", (model, device)))
        return FailingBackend(calls)

    with pytest.raises(OSError, match="No space"):
        model_assets.prepare_model("clipiqa", data_dir=tmp_path, backend_factory=factory)

    record = json.loads(model_assets.preparation_record_path(tmp_path).read_text(encoding="utf-8"))
    assert record["state"] == "failed"
    assert record["phase"] == "validating_initialization"
    assert record["error_report"]["category"] == "cache_no_space"
    assert record["processed_counts"] == {"validation_images": 1}
    assert calls[-1] == ("close", None)


def test_prepare_model_cancellation_is_retained_as_incomplete_diagnostic(tmp_path: Path) -> None:
    def cancel() -> None:
        raise InterruptedError("Model preparation job was cancelled by user.")

    with pytest.raises(InterruptedError):
        model_assets.prepare_model("topiq_nr", data_dir=tmp_path, cancel_check=cancel)

    record = json.loads(model_assets.preparation_record_path(tmp_path).read_text(encoding="utf-8"))
    assert record["state"] == "failed"
    assert record["cancelled"] is True
    assert record["error_report"]["category"] == "cancelled"
    assert record["recovery_action"]


def test_preparation_rejects_disabled_model_before_backend_creation(tmp_path: Path) -> None:
    called = False

    def factory(*args, **kwargs):
        nonlocal called
        called = True
        return _Backend([])

    with pytest.raises(ValueError, match="unknown or disabled"):
        model_assets.prepare_model("qalign", data_dir=tmp_path, backend_factory=factory)

    assert called is False
    assert not model_assets.preparation_record_path(tmp_path).exists()


def test_readiness_context_change_invalidates_previous_record_without_scanning_cache(tmp_path: Path) -> None:
    model_assets.prepare_model("topiq_nr", data_dir=tmp_path, backend_factory=lambda *_args, **_kwargs: _Backend([]))

    record = model_assets.read_preparation_record(tmp_path, cache_dir=tmp_path / "new-cache")

    assert record["state"] == "not_checked"
    assert record["reason"] == "preparation_context_changed"
    assert record["last_preparation"]["state"] == "prepared"


def test_model_cache_dir_sets_defaults_but_preserves_explicit_split_paths(tmp_path: Path) -> None:
    env = {"HF_HUB_CACHE": str(tmp_path / "explicit-hub")}

    paths = model_assets.apply_model_cache_dir(tmp_path / "models", environ=env)

    assert env["HF_HOME"] == str(tmp_path / "models" / "huggingface")
    assert env["HF_HUB_CACHE"] == str(tmp_path / "explicit-hub")
    assert env["TORCH_HOME"] == str(tmp_path / "models" / "torch")
    assert paths["hf_hub_cache"] == str((tmp_path / "explicit-hub").resolve())


def test_preparation_error_sanitizes_tokens_and_url_queries() -> None:
    env = {"HF_TOKEN": "super-secret", "HTTPS_PROXY": "https://user:pass@example.test:8443"}
    error = RuntimeError("Hub request https://user:pass@example.test/model?token=super-secret failed")

    report = model_assets.classify_preparation_error(error, phase="preparing_model", environ=env)

    assert "super-secret" not in json.dumps(report)
    assert "user:pass" not in json.dumps(report)
    assert "?token" not in json.dumps(report)
    assert report["category"] == "network_or_hub"


def test_model_cache_dir_respects_existing_hf_home(tmp_path):
    env = {"HF_HOME": str(tmp_path / "existing")}
    paths = model_assets.apply_model_cache_dir(tmp_path / "models", environ=env)
    assert paths["hf_hub_cache"] == str(tmp_path / "existing" / "hub")


def test_clipiqa_readiness_does_not_require_unused_hub_cache(tmp_path):
    cache = tmp_path / "torch"
    cache.mkdir()
    env = {"TORCH_HOME": str(cache), "HF_HOME": str(tmp_path / "unused-hub")}
    model_assets.prepare_model(
        "clipiqa", data_dir=tmp_path, environ=env,
        backend_factory=lambda *_args, **_kwargs: _Backend([]),
    )
    assert model_assets.read_preparation_record(tmp_path, environ=env)["state"] == "prepared"


def test_default_cache_diagnostics_resolve_upstream_locations(tmp_path):
    paths = model_assets.effective_cache_paths(environ={"XDG_CACHE_HOME": str(tmp_path)})
    assert paths["hf_home"] == str(tmp_path / "huggingface")
    assert paths["hf_hub_cache"] == str(tmp_path / "huggingface" / "hub")
    assert paths["torch_home"] == str(tmp_path / "torch")


@pytest.mark.parametrize("header", ["Authorization: Bearer", "Authorization: Basic", "Bearer"])
def test_preparation_error_redacts_authorization_credentials(header):
    report = model_assets.classify_preparation_error(
        RuntimeError(f"Request failed: {header} private-credential"),
        phase="preparing_model", environ={},
    )
    assert "private-credential" not in json.dumps(report)
