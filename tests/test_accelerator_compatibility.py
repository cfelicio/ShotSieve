from contextlib import nullcontext
import logging
from pathlib import Path
from types import SimpleNamespace
import warnings

import pytest

from shotsieve import learned_iqa_backend as backend
from shotsieve import learned_iqa_runtime as runtime


# SYCL device_architecture.def: Tiger Lake and Alchemist G10, respectively.
PRE_ARC_ARCHITECTURE = 0x0000000300000000
ARC_ARCHITECTURE = 0x000000030DC00800
LEVEL_ZERO_FAILURE = "level_zero backend failed with error: 2147483646 (UR_RESULT_ERROR_UNKNOWN)"


def _xpu_torch(architecture):
    return SimpleNamespace(
        device=str,
        xpu=SimpleNamespace(
            is_available=lambda: True,
            get_device_properties=lambda: SimpleNamespace(name="Detected Intel GPU", architecture=architecture),
        ),
    )


def _score_paths(score_batch, *, selected_runtime="xpu", load_batch=None):
    return backend.score_paths(
        SimpleNamespace(
            name="topiq_nr", input_size=384, _torch=object(),
            tensor_device=selected_runtime, runtime=selected_runtime,
            _score_tensor_batch=score_batch,
        ),
        [Path("first.jpg"), Path("second.jpg"), Path("third.jpg")],
        batch_size=2,
        recommended_cpu_workers_fn=lambda *args, **kwargs: 1,
        load_batch_tensor_fn=load_batch or (lambda paths, **kwargs: list(paths)),
        arrays_to_tensor_fn=lambda arrays, **kwargs: arrays,
        load_single_image_fn=lambda path, *args, **kwargs: path,
    )


def test_pre_arc_xpu_is_rejected_by_architecture():
    torch = _xpu_torch(PRE_ARC_ARCHITECTURE)
    usable, reason = runtime.xpu_runtime_status(torch)
    assert usable is False
    assert "Detected Intel GPU" in reason
    assert "not supported by the installed PyTorch XPU runtime" in reason
    assert runtime.has_xpu(torch) is False
    assert runtime.runtime_statuses(torch_module=torch, system_name="Windows")["xpu"] == "unavailable"


@pytest.mark.parametrize("architecture", [ARC_ARCHITECTURE, 0x0000000500400400])
def test_arc_and_newer_xpu_are_accepted_without_requiring_matrix_instructions_or_aot_kernels(architecture):
    torch = _xpu_torch(architecture)
    torch.xpu.get_arch_list = lambda: []
    assert runtime.xpu_runtime_status(torch) == (True, None)
    assert runtime.has_xpu(torch) is True
    assert runtime.resolve_device("xpu", torch_module=torch).runtime == "xpu"


@pytest.mark.parametrize("requested", [None, "auto"])
def test_auto_rejects_pre_arc_xpu_logs_one_warning_and_scores_on_cpu(requested, caplog):
    torch = _xpu_torch(PRE_ARC_ARCHITECTURE)

    def visible_with_torch_warning():
        warnings.warn("Detected Intel GPU is not officially supported by PyTorch XPU.", UserWarning)
        return True

    torch.xpu.is_available = visible_with_torch_warning
    with warnings.catch_warnings(record=True) as caught, caplog.at_level(logging.WARNING):
        resolved = runtime.resolve_device(requested, torch_module=torch, system_name="Windows")
    assert not caught
    assert resolved.runtime == "cpu"
    assert resolved.tensor_device == "cpu"
    assert "not supported" in resolved.fallback_reason
    assert [record.getMessage() for record in caplog.records] == [
        "Detected Intel GPU was detected, but it is not supported by the installed PyTorch XPU runtime. Falling back to CPU.",
    ]
    results = _score_paths(
        lambda paths: [backend.LearnedScoreResult(0.8, 80.0) for _ in paths],
        selected_runtime=resolved.runtime,
    )
    assert len(results) == 3
    assert all(not result.failed for result in results)


@pytest.mark.parametrize("requested", ["xpu", "intel"])
def test_explicit_unsupported_xpu_returns_a_controlled_error(requested, caplog):
    with pytest.raises(runtime.LearnedRuntimeUnavailableError, match="not supported by the installed PyTorch XPU runtime"):
        runtime.resolve_device(requested, torch_module=_xpu_torch(PRE_ARC_ARCHITECTURE))
    assert "Falling back to CPU" not in caplog.text


@pytest.mark.parametrize("architecture", [None, True, -1, 0x9900000000000000])
def test_xpu_unknown_architecture_does_not_rely_on_availability(architecture):
    usable, reason = runtime.xpu_runtime_status(_xpu_torch(architecture))
    assert not usable
    assert "cannot be verified" in reason


def test_xpu_probe_failure_is_controlled_and_auto_can_use_cpu():
    torch = _xpu_torch(ARC_ARCHITECTURE)

    def failed_probe():
        raise RuntimeError("driver initialization failed")

    torch.xpu.get_device_properties = failed_probe
    assert runtime.resolve_device(None, torch_module=torch).runtime == "cpu"
    with pytest.raises(runtime.LearnedRuntimeUnavailableError, match="XPU device compatibility probe failed"):
        runtime.resolve_device("xpu", torch_module=torch)


@pytest.mark.parametrize(("arches", "usable"), [
    (["gfx1100"], False),
    (["gfx1151:xnack-"], True),
    (["gfx11-generic"], True),
])
def test_rocm_compares_only_concrete_compiled_architectures(arches, usable):
    torch = SimpleNamespace(
        device=str, version=SimpleNamespace(hip="10.0.0"),
        cuda=SimpleNamespace(
            is_available=lambda: True, get_arch_list=lambda: arches,
            get_device_properties=lambda: SimpleNamespace(name="Detected AMD GPU", gcnArchName="gfx1151:xnack-"),
        ),
    )
    assert runtime.has_rocm(torch) is usable
    resolved = runtime.resolve_device(None, torch_module=torch, system_name="Linux")
    assert resolved.runtime == ("rocm" if usable else "cpu")
    if not usable:
        with pytest.raises(runtime.LearnedRuntimeUnavailableError, match="gfx1151.*no kernels"):
            runtime.resolve_device("rocm", torch_module=torch)


@pytest.mark.parametrize("failure_stage", ["batch", "transfer", "individual"])
def test_fatal_xpu_failure_stops_without_further_image_or_batch_retries(failure_stage):
    calls = []

    def score_batch(paths):
        calls.append(tuple(paths))
        if failure_stage == "individual" and len(paths) > 1:
            raise RuntimeError("image decode failed")
        raise RuntimeError(LEVEL_ZERO_FAILURE)

    def load_batch(paths, **kwargs):
        if failure_stage == "transfer":
            calls.append(tuple(paths))
            raise RuntimeError(LEVEL_ZERO_FAILURE)
        return list(paths)

    with pytest.raises(backend.LearnedBackendUnavailableError, match="Fatal xpu accelerator failure") as caught:
        _score_paths(score_batch, load_batch=load_batch)
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert len(calls) == (2 if failure_stage == "individual" else 1)


@pytest.mark.parametrize("message", [
    "image decode failed", "XPU out of memory",
    "level_zero backend failed with error: UR_RESULT_ERROR_INVALID_ARGUMENT",
])
def test_ordinary_batch_failure_still_uses_individual_fallback(message):
    calls = []

    def score_batch(paths):
        calls.append(tuple(paths))
        if len(paths) > 1:
            raise RuntimeError(message)
        return [backend.LearnedScoreResult(0.8, 80.0)]

    results = _score_paths(score_batch)
    assert calls == [
        (Path("first.jpg"), Path("second.jpg")),
        (Path("first.jpg"),), (Path("second.jpg"),), (Path("third.jpg"),),
    ]
    assert len(results) == 3
    assert all(not result.failed for result in results)


def test_fatal_cuda_failure_does_not_retry_without_autocast():
    calls = []

    def metric(*args, **kwargs):
        calls.append("forward")
        raise RuntimeError("CUDA error: device-side assert triggered")

    torch = SimpleNamespace(inference_mode=nullcontext, autocast=lambda *args, **kwargs: nullcontext(), float16=object())
    with pytest.raises(backend.LearnedBackendUnavailableError, match="Fatal cuda accelerator failure"):
        backend.score_tensor_batch(
            SimpleNamespace(_torch=torch, runtime="cuda", metric=metric), object(),
            flatten_tensor_fn=lambda value: value,
            confidence_values_fn=lambda *args, **kwargs: [],
            normalize_score_fn=lambda value, **kwargs: value,
        )
    assert calls == ["forward"]
