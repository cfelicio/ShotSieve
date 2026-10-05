from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import threading

import pytest

from shotsieve import learned_iqa_backend as backend


def _runtime_and_metric(model_name, *, enabled=True):
    """Model eager Conv/BN execution, including CLIPIQA's unregistered CLIP."""
    cudnn = SimpleNamespace(enabled=enabled)
    calls = []

    class Module:
        def __init__(self, *children):
            self.children = children

        def modules(self):
            yield self
            for child in self.children:
                yield from child.modules()

        def __call__(self, value, **kwargs):
            return self.forward(value, **kwargs)

        def forward(self, value, **kwargs):
            for child in self.children:
                value = child(value)
            return value

    class Conv(Module):
        def forward(self, value):
            calls.append(("conv", cudnn.enabled))
            return value

    class BatchNorm(Module):
        def forward(self, value):
            calls.append(("bn", cudnn.enabled))
            if cudnn.enabled:
                raise RuntimeError("MIOpenBatchNormFwdInferSpatial.cpp: HIPRTC_ERROR_COMPILATION")
            return value

    class BatchNormAct(BatchNorm):
        def forward(self, value):
            return [item + 0.25 for item in super().forward(value)]

    bn = BatchNormAct()
    backbone = Module(Conv(), bn, Conv())
    if model_name == "clipiqa":
        metric = Module()
        metric.net = Module()
        metric.net.clip_model = [backbone]
        metric.forward = lambda value, **kwargs: metric.net.clip_model[0](value)
    else:
        metric = Module(backbone)
    torch = SimpleNamespace(
        __version__="2.13.0+rocm10.0.0", version=SimpleNamespace(hip="10.0.0"),
        nn=SimpleNamespace(BatchNorm1d=BatchNorm, BatchNorm2d=BatchNorm, BatchNorm3d=BatchNorm),
        backends=SimpleNamespace(cudnn=cudnn), inference_mode=nullcontext,
    )
    return torch, metric, bn, calls


def _initialize(torch, metric, model_name, runtime):
    instance = SimpleNamespace()
    backend.initialize_backend(
        instance, model_name,
        import_pyiqa_runtime_fn=lambda: (SimpleNamespace(list_models=lambda **kwargs: [model_name]), torch),
        normalize_model_name_fn=lambda name: name,
        preferred_model_names_fn=sorted,
        resolve_device_fn=lambda *args, **kwargs: SimpleNamespace(
            runtime=runtime, metric_device="cuda", display_device=runtime, tensor_device="cuda",
        ),
        create_metric_safely_fn=lambda *args, **kwargs: metric,
    )
    return instance


@pytest.mark.parametrize("model_name", ["topiq_nr", "clipiqa"])
def test_windows_rocm_initialization_bypasses_only_batchnorm(monkeypatch, model_name):
    monkeypatch.setattr(backend, "current_system_name", lambda: "Windows")
    torch, metric, bn, calls = _runtime_and_metric(model_name)
    # Reproduce dispatch failure, then the proposed all-native control.
    with pytest.raises(RuntimeError, match="HIPRTC_ERROR_COMPILATION"):
        metric([0.25])
    torch.backends.cudnn.enabled = False
    assert metric([0.25]) == [0.5]
    torch.backends.cudnn.enabled = True
    calls.clear()

    instance = _initialize(torch, metric, model_name, "rocm")
    results = backend.score_tensor_batch(
        instance, [0.25], flatten_tensor_fn=lambda value: [value],
        confidence_values_fn=lambda *args, **kwargs: [],
        normalize_score_fn=lambda score, **kwargs: score * 100,
    )
    assert results[0].raw_score == 0.5  # Preserve BatchNorm subclass activation.
    assert calls == [("conv", True), ("bn", False), ("conv", True)]
    assert torch.backends.cudnn.enabled is True


@pytest.mark.parametrize(("system", "runtime", "model_name"), [
    ("Linux", "rocm", "topiq_nr"),
    ("Windows", "cuda", "clipiqa"),
    ("Windows", "xpu", "topiq_nr"),
    ("Darwin", "mps", "topiq_nr"),
    ("Windows", "cpu", "topiq_nr"),
    ("Windows", "rocm", "qrealign-mini"),
])
def test_other_runtimes_and_qrealign_are_not_wrapped(monkeypatch, system, runtime, model_name):
    monkeypatch.setattr(backend, "current_system_name", lambda: system)
    torch, metric, bn, calls = _runtime_and_metric(model_name)
    original_forward = bn.forward
    instance = _initialize(torch, metric, model_name, runtime)
    assert bn.forward == original_forward
    assert instance.metric is metric
    assert torch.backends.cudnn.enabled is True


@pytest.mark.parametrize("enabled", [False, True])
def test_batchnorm_dispatch_is_restored_after_exception(monkeypatch, enabled):
    monkeypatch.setattr(backend, "current_system_name", lambda: "Windows")
    torch, metric, bn, calls = _runtime_and_metric("topiq_nr", enabled=enabled)

    def fail(value):
        assert torch.backends.cudnn.enabled is False
        raise ValueError("invalid image tensor")

    bn.forward = fail
    _initialize(torch, metric, "topiq_nr", "rocm")
    with pytest.raises(ValueError, match="invalid image tensor"):
        metric([0.25])
    assert torch.backends.cudnn.enabled is enabled


def test_concurrent_qrealign_forward_cannot_observe_batchnorm_toggle(monkeypatch):
    monkeypatch.setattr(backend, "current_system_name", lambda: "Windows")
    torch, metric, bn, calls = _runtime_and_metric("topiq_nr")
    inside_bn = threading.Event()
    release_bn = threading.Event()
    qrealign_attempted = threading.Event()
    observations = []
    errors = []

    def bn_forward(value):
        inside_bn.set()
        assert release_bn.wait(5)
        return value

    bn.forward = bn_forward
    instance = _initialize(torch, metric, "topiq_nr", "rocm")

    def qrealign_metric(value, **kwargs):
        observations.append(torch.backends.cudnn.enabled)
        return value

    def run(instance):
        try:
            if instance.name == "qrealign-mini":
                qrealign_attempted.set()
            backend._score_metric_output(instance, [0.25])
        except BaseException as exc:
            errors.append(exc)

    first = threading.Thread(target=run, args=(instance,))
    second = threading.Thread(target=run, args=(SimpleNamespace(
        name="qrealign-mini", runtime="rocm", metric=qrealign_metric,
    ),))
    first.start()
    try:
        assert inside_bn.wait(5)
        second.start()
        assert qrealign_attempted.wait(5)
        assert observations == []
    finally:
        release_bn.set()
        first.join(5)
        if second.ident is not None:
            second.join(5)
    assert not first.is_alive() and not second.is_alive()
    assert not errors
    assert observations == [True]
    assert torch.backends.cudnn.enabled is True


def _score_paths(score_batch):
    return backend.score_paths(
        SimpleNamespace(
            name="topiq_nr", input_size=384, _torch=object(),
            tensor_device="cuda", runtime="rocm", _score_tensor_batch=score_batch,
        ),
        [Path("first.jpg"), Path("second.jpg"), Path("third.jpg")],
        batch_size=2, recommended_cpu_workers_fn=lambda *args, **kwargs: 1,
        load_batch_tensor_fn=lambda paths, **kwargs: list(paths),
        arrays_to_tensor_fn=lambda arrays, **kwargs: arrays,
        load_single_image_fn=lambda path, *args, **kwargs: path,
    )


@pytest.mark.parametrize("message", [
    "HIPRTC_ERROR_COMPILATION",
    "MIOpen Error: Code object build failed. Source: MIOpenBatchNormFwdInferSpatial.cpp",
    "miopenStatusUnknownError: MIOpen compilation failed: fatal error: 'type_traits' file not found",
    "miopenStatusUnknownError caused by MIOpen compilation",
])
@pytest.mark.parametrize("stage", ["batch", "individual"])
def test_rocm_compiler_failure_aborts_without_further_retries(message, stage):
    calls = []

    def score_batch(paths):
        calls.append(tuple(paths))
        if stage == "individual" and len(paths) > 1:
            raise ValueError("problematic image batch")
        try:
            raise RuntimeError(message)
        except RuntimeError as exc:
            raise RuntimeError("model forward failed") from exc

    with pytest.raises(backend.LearnedBackendUnavailableError, match="Fatal rocm accelerator failure"):
        _score_paths(score_batch)
    assert len(calls) == (1 if stage == "batch" else 2)


@pytest.mark.parametrize("message", [
    "corrupt image", "HIP out of memory", "miopenStatusUnknownError",
    "MIOpen unsupported operator", "model compilation failed",
])
def test_non_compiler_batch_failures_keep_individual_retry(message):
    calls = []

    def score_batch(paths):
        calls.append(tuple(paths))
        if len(paths) > 1:
            raise RuntimeError(message)
        return [backend.LearnedScoreResult(0.25, 25.0)]

    results = _score_paths(score_batch)
    assert [len(paths) for paths in calls] == [2, 1, 1, 1]
    assert len(results) == 3 and all(not result.failed for result in results)


@pytest.mark.parametrize("runtime", ["cpu", "cuda", "xpu", "mps"])
def test_rocm_compiler_classification_does_not_change_other_runtimes(runtime):
    assert not backend._is_fatal_accelerator_error(RuntimeError("HIPRTC_ERROR_COMPILATION"), runtime=runtime)
