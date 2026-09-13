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


def test_model_diagnostic_includes_cache_volumes_and_known_exception_chain() -> None:
    inner = OSError("Model was not found in cache while local_files_only is enabled")
    error = RuntimeError("backend initialization failed")
    error.__cause__ = inner

    diagnostic = model_assets.build_model_diagnostic(
        error,
        phase="initializing_model",
        model_name="topiq_nr",
        requested_runtime="cuda",
        actual_runtime=None,
        cache_paths={"hf_hub_cache": "C:/model-cache/hub", "torch_home": "D:/torch"},
        environ={"HF_HUB_OFFLINE": "1"},
    )

    assert diagnostic["category"] == "missing_offline_assets"
    assert diagnostic["model"] == "topiq_nr"
    assert diagnostic["requested_runtime"] == "cuda"
    assert diagnostic["actual_runtime"] == "unknown"
    assert diagnostic["cache_volumes"]
    assert "Prepare selected model" in diagnostic["recovery_action"]


def test_prepare_model_retains_sanitized_diagnostic_when_record_writes_fail(tmp_path: Path) -> None:
    def failing_writer(_data_dir, _record):
        raise OSError("record write failed: token=private-secret")

    def factory(*_args, **_kwargs):
        raise RuntimeError("Hub request https://example.test/model?token=private-secret failed")

    with pytest.raises(RuntimeError) as exc_info:
        model_assets.prepare_model(
            "topiq_nr",
            data_dir=tmp_path,
            backend_factory=factory,
            record_writer=failing_writer,
            environ={"HF_TOKEN": "private-secret"},
        )

    diagnostic = exc_info.value.model_diagnostic
    encoded = json.dumps(diagnostic)
    assert diagnostic["category"] == "network_or_hub"
    assert diagnostic["record_write_error"] == "record write failed: token=<redacted>"
    assert "private-secret" not in encoded
    assert not model_assets.preparation_record_path(tmp_path).exists()


def test_read_preparation_record_downgrades_dead_preparation_process(tmp_path: Path, monkeypatch) -> None:
    env = {"XDG_CACHE_HOME": str(tmp_path / "cache")}
    cache_paths = model_assets.effective_cache_paths(environ=env)
    versions = model_assets._dependency_versions()
    fingerprint = model_assets._dependency_fingerprint(
        "topiq_nr",
        cache_paths=cache_paths,
        dependency_versions=versions,
        environ=env,
    )
    record = model_assets._base_record(
        "topiq_nr",
        cache_paths=cache_paths,
        dependency_versions=versions,
        dependency_fingerprint=fingerprint,
        environ=env,
    )
    record["process_id"] = 1234
    monkeypatch.setattr(model_assets, "_process_is_alive", lambda _process_id: False)
    model_assets.write_preparation_record(tmp_path, record)

    recovered = model_assets.read_preparation_record(tmp_path, environ=env)

    assert recovered["state"] == "failed"
    assert recovered["orphaned"] is True
    assert recovered["error_report"]["category"] == "interrupted_process"
    stored = json.loads(model_assets.preparation_record_path(tmp_path).read_text(encoding="utf-8"))
    assert stored["state"] == "failed"


@pytest.mark.parametrize("header", ["Authorization: Bearer", "Authorization: Basic", "Bearer"])
def test_preparation_error_redacts_authorization_credentials(header):
    report = model_assets.classify_preparation_error(
        RuntimeError(f"Request failed: {header} private-credential"),
        phase="preparing_model", environ={},
    )
    assert "private-credential" not in json.dumps(report)
