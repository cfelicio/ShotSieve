from contextlib import nullcontext
import logging
import sys
import types
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast
import warnings

import numpy as np
import pytest
import shotsieve.learned_iqa as learned_iqa_module
from shotsieve.db import initialize_database
from shotsieve.config import parse_extensions
from shotsieve.learned_iqa import (
    configure_runtime_noise_controls,
    install_runtime_warning_filters,
    normalize_model_name,
    normalize_device_target,
    normalize_score,
    parse_score_range,
    resolve_device,
    runtime_statuses,
    supported_learned_models,
    supported_runtime_targets,
)


def _new_module(name: str) -> Any:
    return types.ModuleType(name)


def test_learned_iqa_split_runtime_and_catalog_modules_preserve_facade_exports() -> None:
    from shotsieve import learned_iqa_catalog as catalog_module
    from shotsieve import learned_iqa_runtime as runtime_module

    class NoCudaTorch:
        @staticmethod
        def device(name: str) -> str:
            return name

        class cuda:
            @staticmethod
            def is_available() -> bool:
                return False

        class xpu:
            @staticmethod
            def is_available() -> bool:
                return False

    assert catalog_module.normalize_model_name("Q-Align") == learned_iqa_module.normalize_model_name("Q-Align")
    assert catalog_module.supported_learned_models() == learned_iqa_module.supported_learned_models()
    assert catalog_module.supported_runtime_targets() == learned_iqa_module.supported_runtime_targets()
    assert runtime_module.resolve_device(
        None,
        torch_module=NoCudaTorch,
        import_module=lambda name: (_ for _ in ()).throw(ImportError(name)),
    ).runtime == learned_iqa_module.resolve_device(
        None,
        torch_module=NoCudaTorch,
        import_module=lambda name: (_ for _ in ()).throw(ImportError(name)),
    ).runtime
    assert callable(runtime_module.detect_hardware_capabilities)


def test_learned_iqa_split_backend_and_preprocessing_modules_preserve_facade_exports() -> None:
    from shotsieve import learned_iqa_backend as backend_module
    from shotsieve import learned_iqa_preprocessing as preprocessing_module

    result = backend_module.LearnedScoreResult(raw_score=0.2, normalized_score=20.0)

    assert isinstance(result, learned_iqa_module.LearnedScoreResult)
    assert callable(backend_module.build_learned_backend)
    assert callable(backend_module.release_learned_backend)
    assert callable(preprocessing_module.load_batch_tensor)
    assert preprocessing_module.parse_score_range("~0, ~1") == learned_iqa_module.parse_score_range("~0, ~1")
    assert preprocessing_module.normalize_score(0.2, score_range="0, 1", lower_better=False) == learned_iqa_module.normalize_score(0.2, score_range="0, 1", lower_better=False)


def test_initialize_backend_enables_cudnn_benchmark_for_cuda_runtime() -> None:
    from shotsieve import learned_iqa_backend as backend_module

    class FakePyiqa:
        __version__ = "0.1-test"

        @staticmethod
        def list_models(metric_mode: str):
            assert metric_mode == "NR"
            return ["topiq_nr"]

    fake_torch = types.SimpleNamespace(
        backends=types.SimpleNamespace(cudnn=types.SimpleNamespace(benchmark=False)),
    )
    backend = types.SimpleNamespace()
    metric = types.SimpleNamespace(
        lower_better=False,
        score_range="0, 1",
        net=types.SimpleNamespace(test_img_size=384),
    )

    def create_metric(pyiqa, model_name, *, device):
        assert fake_torch.backends.cudnn.benchmark is True
        return metric

    backend_module.initialize_backend(
        backend,
        "topiq_nr",
        import_pyiqa_runtime_fn=lambda: (FakePyiqa, fake_torch),
        normalize_model_name_fn=lambda model_name: model_name,
        preferred_model_names_fn=lambda models: sorted(models),
        resolve_device_fn=lambda device, torch_module: types.SimpleNamespace(
            runtime="cuda",
            metric_device="cuda:0",
            display_device="cuda:0",
            tensor_device="cuda:0",
        ),
        create_metric_safely_fn=create_metric,
    )

    assert fake_torch.backends.cudnn.benchmark is True
    assert backend.runtime == "cuda"


def test_initialize_backend_leaves_cudnn_benchmark_disabled_for_non_cuda_runtime() -> None:
    from shotsieve import learned_iqa_backend as backend_module

    class FakePyiqa:
        __version__ = "0.1-test"

        @staticmethod
        def list_models(metric_mode: str):
            assert metric_mode == "NR"
            return ["topiq_nr"]

    fake_torch = types.SimpleNamespace(
        backends=types.SimpleNamespace(cudnn=types.SimpleNamespace(benchmark=False)),
    )
    backend = types.SimpleNamespace()
    metric = types.SimpleNamespace(
        lower_better=False,
        score_range="0, 1",
        net=types.SimpleNamespace(test_img_size=384),
    )

    backend_module.initialize_backend(
        backend,
        "topiq_nr",
        import_pyiqa_runtime_fn=lambda: (FakePyiqa, fake_torch),
        normalize_model_name_fn=lambda model_name: model_name,
        preferred_model_names_fn=lambda models: sorted(models),
        resolve_device_fn=lambda device, torch_module: types.SimpleNamespace(
            runtime="cpu",
            metric_device="cpu",
            display_device="cpu",
            tensor_device="cpu",
        ),
        create_metric_safely_fn=lambda pyiqa, model_name, *, device: metric,
    )

    assert fake_torch.backends.cudnn.benchmark is False
    assert backend.runtime == "cpu"


def test_close_backend_restores_prior_cudnn_benchmark_value_after_cuda_backend_closes() -> None:
    from shotsieve import learned_iqa_backend as backend_module

    class FakePyiqa:
        __version__ = "0.1-test"

        @staticmethod
        def list_models(metric_mode: str):
            assert metric_mode == "NR"
            return ["topiq_nr"]

    fake_torch = types.SimpleNamespace(
        backends=types.SimpleNamespace(cudnn=types.SimpleNamespace(benchmark=False)),
    )
    backend = types.SimpleNamespace()
    metric = types.SimpleNamespace(
        lower_better=False,
        score_range="0, 1",
        net=types.SimpleNamespace(test_img_size=384),
    )

    backend_module.initialize_backend(
        backend,
        "topiq_nr",
        import_pyiqa_runtime_fn=lambda: (FakePyiqa, fake_torch),
        normalize_model_name_fn=lambda model_name: model_name,
        preferred_model_names_fn=lambda models: sorted(models),
        resolve_device_fn=lambda device, torch_module: types.SimpleNamespace(
            runtime="cuda",
            metric_device="cuda:0",
            display_device="cuda:0",
            tensor_device="cuda:0",
        ),
        create_metric_safely_fn=lambda pyiqa, model_name, *, device: metric,
    )

    class FakeGcModule:
        @staticmethod
        def collect() -> None:
            return None

    backend_module.close_backend(backend, gc_module=FakeGcModule())

    assert fake_torch.backends.cudnn.benchmark is False


def test_initialize_backend_restores_prior_cudnn_benchmark_value_when_metric_creation_fails() -> None:
    from shotsieve import learned_iqa_backend as backend_module

    class FakePyiqa:
        __version__ = "0.1-test"

        @staticmethod
        def list_models(metric_mode: str):
            assert metric_mode == "NR"
            return ["topiq_nr"]

    fake_torch = types.SimpleNamespace(
        backends=types.SimpleNamespace(cudnn=types.SimpleNamespace(benchmark=False)),
    )
    backend = types.SimpleNamespace()

    with pytest.raises(backend_module.LearnedBackendUnavailableError, match="metric init failed"):
        backend_module.initialize_backend(
            backend,
            "topiq_nr",
            import_pyiqa_runtime_fn=lambda: (FakePyiqa, fake_torch),
            normalize_model_name_fn=lambda model_name: model_name,
            preferred_model_names_fn=lambda models: sorted(models),
            resolve_device_fn=lambda device, torch_module: types.SimpleNamespace(
                runtime="cuda",
                metric_device="cuda:0",
                display_device="cuda:0",
                tensor_device="cuda:0",
            ),
            create_metric_safely_fn=lambda pyiqa, model_name, *, device: (_ for _ in ()).throw(RuntimeError("metric init failed")),
        )

    assert fake_torch.backends.cudnn.benchmark is False
    assert not hasattr(backend, "_previous_cudnn_benchmark")


def test_arrays_to_tensor_stacks_before_applying_channels_last_and_device_transfer() -> None:
    from shotsieve import learned_iqa_preprocessing as preprocessing_module

    operations: list[tuple[object, ...]] = []

    class FakeTensorDevice:
        def __init__(self, device_type: str) -> None:
            self.type = device_type

        def __str__(self) -> str:
            return f"{self.type}:0"

    class FakeTensor:
        def __init__(self, label: str) -> None:
            self.label = label

        def permute(self, *dims: int):
            operations.append((self.label, "permute", dims))
            return self

        def pin_memory(self):
            operations.append((self.label, "pin_memory"))
            return self

        def to(self, device=None, *, memory_format=None, non_blocking: bool = False):
            operations.append((self.label, "to", device, memory_format, non_blocking))
            return self

    class FakeTorch:
        channels_last = "channels_last"

        @staticmethod
        def from_numpy(array: np.ndarray) -> FakeTensor:
            label = f"image-{len([entry for entry in operations if entry[1] == 'permute']) + 1}"
            return FakeTensor(label)

        @staticmethod
        def stack(tensors: list[FakeTensor], dim: int = 0) -> FakeTensor:
            operations.append(("stack", [tensor.label for tensor in tensors], dim))
            return FakeTensor("batch")

    arrays = [
        np.zeros((4, 4, 3), dtype=np.float32),
        np.ones((4, 4, 3), dtype=np.float32),
    ]
    cuda_device = FakeTensorDevice("cuda")

    batch_tensor = cast(FakeTensor, preprocessing_module._arrays_to_tensor(
        arrays,
        torch_module=FakeTorch,
        tensor_device=cuda_device,
        use_channels_last=True,
    ))

    assert batch_tensor.label == "batch"
    assert operations == [
        ("image-1", "permute", (2, 0, 1)),
        ("image-2", "permute", (2, 0, 1)),
        ("stack", ["image-1", "image-2"], 0),
        ("batch", "to", None, "channels_last", False),
        ("batch", "pin_memory"),
        ("batch", "to", cuda_device, None, True),
    ]


@pytest.mark.parametrize("runtime", ["cpu", "directml", "mps"])
def test_arrays_to_tensor_skips_pin_memory_and_non_blocking_for_non_cuda_targets(
    runtime: str,
) -> None:
    from shotsieve import learned_iqa_preprocessing as preprocessing_module

    operations: list[tuple[object, ...]] = []

    class FakeTensorDevice:
        def __init__(self, device_type: str) -> None:
            self.type = device_type

        def __str__(self) -> str:
            return f"{self.type}:0"

    class FakeDirectMlDevice:
        def __str__(self) -> str:
            return "dml:0"

    class FakeTensor:
        def __init__(self, label: str) -> None:
            self.label = label

        def permute(self, *dims: int):
            operations.append((self.label, "permute", dims))
            return self

        def pin_memory(self):
            operations.append((self.label, "pin_memory"))
            return self

        def to(self, device=None, *, memory_format=None, non_blocking: bool = False):
            operations.append((self.label, "to", device, memory_format, non_blocking))
            return self

    class FakeTorch:
        channels_last = "channels_last"

        @staticmethod
        def from_numpy(array: np.ndarray) -> FakeTensor:
            label = f"image-{len([entry for entry in operations if entry[1] == 'permute']) + 1}"
            return FakeTensor(label)

        @staticmethod
        def stack(tensors: list[FakeTensor], dim: int = 0) -> FakeTensor:
            operations.append(("stack", [tensor.label for tensor in tensors], dim))
            return FakeTensor("batch")

    arrays = [np.zeros((4, 4, 3), dtype=np.float32)]
    tensor_device = {
        "cpu": FakeTensorDevice("cpu"),
        "directml": FakeDirectMlDevice(),
        "mps": FakeTensorDevice("mps"),
    }[runtime]

    batch_tensor = cast(FakeTensor, preprocessing_module._arrays_to_tensor(
        arrays,
        torch_module=FakeTorch,
        tensor_device=tensor_device,
        use_channels_last=False,
    ))

    assert batch_tensor.label == "batch"
    assert operations == [
        ("image-1", "permute", (2, 0, 1)),
        ("stack", ["image-1"], 0),
        ("batch", "to", tensor_device, None, False),
    ]


def test_arrays_to_tensor_skips_channels_last_without_explicit_runtime_opt_in() -> None:
    from shotsieve import learned_iqa_preprocessing as preprocessing_module

    operations: list[tuple[object, ...]] = []

    class FakeTensor:
        def __init__(self, label: str) -> None:
            self.label = label

        def permute(self, *dims: int):
            operations.append((self.label, "permute", dims))
            return self

        def to(self, device=None, *, memory_format=None):
            operations.append((self.label, "to", device, memory_format))
            return self

    class FakeTorch:
        channels_last = "channels_last"

        @staticmethod
        def from_numpy(array: np.ndarray) -> FakeTensor:
            label = f"image-{len([entry for entry in operations if entry[1] == 'permute']) + 1}"
            return FakeTensor(label)

        @staticmethod
        def stack(tensors: list[FakeTensor], dim: int = 0) -> FakeTensor:
            operations.append(("stack", [tensor.label for tensor in tensors], dim))
            return FakeTensor("batch")

    arrays = [np.zeros((4, 4, 3), dtype=np.float32)]

    batch_tensor = cast(FakeTensor, preprocessing_module._arrays_to_tensor(
        arrays,
        torch_module=FakeTorch,
        tensor_device="cpu",
        use_channels_last=False,
    ))

    assert batch_tensor.label == "batch"
    assert operations == [
        ("image-1", "permute", (2, 0, 1)),
        ("stack", ["image-1"], 0),
        ("batch", "to", "cpu", None),
    ]


def test_score_paths_only_enables_channels_last_for_cpu_and_cuda_runtimes() -> None:
    from shotsieve import learned_iqa_backend as backend_module

    image_paths = [Path("first.jpg"), Path("second.jpg")]

    for runtime, expected in (
        ("cpu", True),
        ("cuda", True),
        ("directml", False),
        ("mps", False),
    ):
        calls: list[tuple[object, ...]] = []
        backend = types.SimpleNamespace(
            name="topiq_nr",
            input_size=384,
            _torch=object(),
            tensor_device=f"device:{runtime}",
            runtime=runtime,
            _score_tensor_batch=lambda batch_tensor: [],
        )

        backend_module.score_paths(
            backend,
            image_paths,
            batch_size=2,
            recommended_cpu_workers_fn=lambda resource_profile, for_threads=False: 1,
            max_batch_sizes={"topiq_nr": 2},
            load_batch_tensor_fn=lambda paths, *, use_channels_last=False, **kwargs: calls.append(("load", runtime, tuple(paths), use_channels_last)) or object(),
            arrays_to_tensor_fn=lambda arrays, *, use_channels_last=False, **kwargs: calls.append(("prefetch", runtime, tuple(arrays), use_channels_last)) or object(),
            load_single_image_fn=lambda path, image_size: path,
        )

        assert calls == [
            ("load", runtime, tuple(image_paths), expected),
        ]


def test_score_paths_disables_next_batch_prefetch_on_cpu_but_keeps_it_for_accelerators() -> None:
    from shotsieve import learned_iqa_backend as backend_module

    image_paths = [Path("first.jpg"), Path("second.jpg")]

    for runtime, expected_calls in (
        (
            "cpu",
            [
                ("load", "cpu", (image_paths[0],), True),
                ("load", "cpu", (image_paths[1],), True),
            ],
        ),
        (
            "cuda",
            [
                ("load", "cuda", (image_paths[0],), True),
                ("prefetch", "cuda", ("decoded:second.jpg",), True),
            ],
        ),
    ):
        calls: list[tuple[object, ...]] = []
        backend = types.SimpleNamespace(
            name="topiq_nr",
            input_size=384,
            _torch=object(),
            tensor_device=f"device:{runtime}",
            runtime=runtime,
            _score_tensor_batch=lambda batch_tensor: [],
        )

        backend_module.score_paths(
            backend,
            image_paths,
            batch_size=1,
            recommended_cpu_workers_fn=lambda resource_profile, for_threads=False: 2,
            max_batch_sizes={"topiq_nr": 1},
            load_batch_tensor_fn=lambda paths, *, use_channels_last=False, **kwargs: calls.append(("load", runtime, tuple(paths), use_channels_last)) or object(),
            arrays_to_tensor_fn=lambda arrays, *, use_channels_last=False, **kwargs: calls.append(("prefetch", runtime, tuple(arrays), use_channels_last)) or object(),
            load_single_image_fn=lambda path, image_size: f"decoded:{path.name}",
        )

        assert calls == expected_calls


def test_score_tensor_batch_uses_inference_mode_for_non_directml_runtimes() -> None:
    from shotsieve import learned_iqa_backend as backend_module

    context_calls: list[str] = []

    class FakeContextManager:
        def __init__(self, label: str) -> None:
            self.label = label

        def __enter__(self) -> None:
            context_calls.append(f"enter:{self.label}")

        def __exit__(self, exc_type, exc, tb) -> None:
            context_calls.append(f"exit:{self.label}")

    class FakeTorch:
        @staticmethod
        def no_grad() -> FakeContextManager:
            return FakeContextManager("no_grad")

        @staticmethod
        def inference_mode() -> FakeContextManager:
            return FakeContextManager("inference_mode")

    backend = types.SimpleNamespace(
        _torch=FakeTorch,
        runtime="cuda",
        metric=lambda batch_tensor, return_mos=True, return_dist=True: ([0.25], [0.75]),
        score_range="0, 1",
        lower_better=False,
    )

    results = backend_module.score_tensor_batch(
        backend,
        object(),
        flatten_tensor_fn=lambda tensor: [tensor[0]],
        confidence_values_fn=lambda dist_tensor, torch_module: [dist_tensor[0]],
        normalize_score_fn=lambda raw_score, **kwargs: raw_score * 100,
    )

    assert len(results) == 1
    assert results[0].raw_score == 0.25
    assert results[0].confidence == 0.75
    assert context_calls == ["enter:inference_mode", "exit:inference_mode"]


def test_score_tensor_batch_uses_no_grad_for_directml_runtime() -> None:
    from shotsieve import learned_iqa_backend as backend_module

    context_calls: list[str] = []

    class FakeContextManager:
        def __init__(self, label: str) -> None:
            self.label = label

        def __enter__(self) -> None:
            context_calls.append(f"enter:{self.label}")

        def __exit__(self, exc_type, exc, tb) -> None:
            context_calls.append(f"exit:{self.label}")

    class FakeTorch:
        @staticmethod
        def no_grad() -> FakeContextManager:
            return FakeContextManager("no_grad")

        @staticmethod
        def inference_mode() -> FakeContextManager:
            return FakeContextManager("inference_mode")

    backend = types.SimpleNamespace(
        _torch=FakeTorch,
        runtime="directml",
        metric=lambda batch_tensor, return_mos=True, return_dist=True: ([0.25], [0.75]),
        score_range="0, 1",
        lower_better=False,
    )

    results = backend_module.score_tensor_batch(
        backend,
        object(),
        flatten_tensor_fn=lambda tensor: [tensor[0]],
        confidence_values_fn=lambda dist_tensor, torch_module: [dist_tensor[0]],
        normalize_score_fn=lambda raw_score, **kwargs: raw_score * 100,
    )

    assert len(results) == 1
    assert results[0].raw_score == 0.25
    assert results[0].confidence == 0.75
    assert context_calls == ["enter:no_grad", "exit:no_grad"]


def test_create_metric_safely_suppresses_known_model_loading_console_noise(
    capsys,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from shotsieve import learned_iqa_backend as backend_module

    monkeypatch.setattr(backend_module.threading, "active_count", lambda: 1)

    class FakePyiqa:
        @staticmethod
        def create_metric(model_name: str, *, device: str):
            assert model_name == "topiq_nr"
            assert device == "cpu"
            print("Loading pretrained model CFANet from /tmp/fake-weights.pt")
            print("Loading pretrained model CFANet from /tmp/fake-weights.pt", file=sys.stderr)
            return object()

    metric = backend_module.create_metric_safely(
        FakePyiqa,
        "topiq_nr",
        device="cpu",
        configure_runtime_noise_controls_fn=lambda: None,
        install_runtime_warning_filters_fn=lambda: None,
    )

    captured = capsys.readouterr()
    assert metric is not None
    assert captured.out == ""
    assert captured.err == ""


def test_create_metric_safely_re_emits_captured_output_when_metric_init_fails(
    capsys,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from shotsieve import learned_iqa_backend as backend_module

    monkeypatch.setattr(backend_module.threading, "active_count", lambda: 1)

    class FakePyiqa:
        @staticmethod
        def create_metric(model_name: str, *, device: str):
            print("loading weights...")
            print("native extension mismatch", file=sys.stderr)
            raise RuntimeError("metric init failed")

    with pytest.raises(RuntimeError, match="metric init failed"):
        backend_module.create_metric_safely(
            FakePyiqa,
            "topiq_nr",
            device="cpu",
            configure_runtime_noise_controls_fn=lambda: None,
            install_runtime_warning_filters_fn=lambda: None,
        )

    captured = capsys.readouterr()
    assert "loading weights..." in captured.out
    assert "native extension mismatch" in captured.err


@pytest.mark.parametrize(
    ("runtime", "expect_autocast"),
    [
        ("cuda", True),
        ("mps", True),
        ("cpu", False),
        ("directml", False),
        ("xpu", False),
    ],
)
def test_score_tensor_batch_only_uses_float16_autocast_for_cuda_and_mps(
    runtime: str,
    expect_autocast: bool,
) -> None:
    from shotsieve import learned_iqa_backend as backend_module

    context_calls: list[str] = []

    class FakeContextManager:
        def __init__(self, label: str) -> None:
            self.label = label

        def __enter__(self) -> None:
            context_calls.append(f"enter:{self.label}")

        def __exit__(self, exc_type, exc, tb) -> None:
            context_calls.append(f"exit:{self.label}")

    class FakeTorch:
        float16 = "float16"

        @staticmethod
        def no_grad() -> FakeContextManager:
            return FakeContextManager("no_grad")

        @staticmethod
        def inference_mode() -> FakeContextManager:
            return FakeContextManager("inference_mode")

        @staticmethod
        def autocast(device_type: str, *, dtype: str) -> FakeContextManager:
            return FakeContextManager(f"autocast:{device_type}:{dtype}")

    backend = types.SimpleNamespace(
        _torch=FakeTorch,
        runtime=runtime,
        name="topiq_nr",
        metric=lambda batch_tensor, return_mos=True, return_dist=True: ([0.25], [0.75]),
        score_range="0, 1",
        lower_better=False,
    )

    results = backend_module.score_tensor_batch(
        backend,
        object(),
        flatten_tensor_fn=lambda tensor: [tensor[0]],
        confidence_values_fn=lambda dist_tensor, torch_module: [dist_tensor[0]],
        normalize_score_fn=lambda raw_score, **kwargs: raw_score * 100,
    )

    assert len(results) == 1
    assert results[0].raw_score == 0.25
    assert results[0].confidence == 0.75

    expected_calls = ["enter:no_grad"] if runtime == "directml" else ["enter:inference_mode"]
    if expect_autocast:
        expected_calls.append(f"enter:autocast:{runtime}:float16")
        expected_calls.append(f"exit:autocast:{runtime}:float16")
    expected_calls.append("exit:no_grad" if runtime == "directml" else "exit:inference_mode")
    assert context_calls == expected_calls


def test_score_tensor_batch_skips_autocast_for_models_blocked_on_the_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from shotsieve import learned_iqa_backend as backend_module

    context_calls: list[str] = []

    class FakeContextManager:
        def __init__(self, label: str) -> None:
            self.label = label

        def __enter__(self) -> None:
            context_calls.append(f"enter:{self.label}")

        def __exit__(self, exc_type, exc, tb) -> None:
            context_calls.append(f"exit:{self.label}")

    class FakeTorch:
        float16 = "float16"

        @staticmethod
        def inference_mode() -> FakeContextManager:
            return FakeContextManager("inference_mode")

        @staticmethod
        def autocast(device_type: str, *, dtype: str) -> FakeContextManager:
            return FakeContextManager(f"autocast:{device_type}:{dtype}")

    monkeypatch.setattr(
        backend_module,
        "AUTOCAST_BLOCKED_MODELS_BY_RUNTIME",
        {
            "cuda": frozenset({"topiq_nr"}),
            "mps": frozenset(),
        },
    )

    backend = types.SimpleNamespace(
        _torch=FakeTorch,
        runtime="cuda",
        name="topiq_nr",
        metric=lambda batch_tensor, return_mos=True, return_dist=True: ([0.25], [0.75]),
        score_range="0, 1",
        lower_better=False,
    )

    results = backend_module.score_tensor_batch(
        backend,
        object(),
        flatten_tensor_fn=lambda tensor: [tensor[0]],
        confidence_values_fn=lambda dist_tensor, torch_module: [dist_tensor[0]],
        normalize_score_fn=lambda raw_score, **kwargs: raw_score * 100,
    )

    assert len(results) == 1
    assert results[0].raw_score == 0.25
    assert results[0].confidence == 0.75
    assert context_calls == ["enter:inference_mode", "exit:inference_mode"]


def test_score_tensor_batch_falls_back_when_autocast_context_entry_fails(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from shotsieve import learned_iqa_backend as backend_module

    metric_calls = 0

    class FakeContextManager:
        def __enter__(self) -> None:
            raise RuntimeError("autocast unavailable")

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    class FakeTorch:
        float16 = "float16"

        @staticmethod
        def inference_mode():
            return nullcontext()

        @staticmethod
        def autocast(device_type: str, *, dtype: str) -> FakeContextManager:
            return FakeContextManager()

    def fake_metric(batch_tensor, return_mos=True, return_dist=True):
        nonlocal metric_calls
        metric_calls += 1
        return [0.25], [0.75]

    backend = types.SimpleNamespace(
        _torch=FakeTorch,
        runtime="mps",
        name="topiq_nr",
        metric=fake_metric,
        score_range="0, 1",
        lower_better=False,
    )

    with caplog.at_level(logging.WARNING):
        results = backend_module.score_tensor_batch(
            backend,
            object(),
            flatten_tensor_fn=lambda tensor: [tensor[0]],
            confidence_values_fn=lambda dist_tensor, torch_module: [dist_tensor[0]],
            normalize_score_fn=lambda raw_score, **kwargs: raw_score * 100,
        )

    assert len(results) == 1
    assert results[0].raw_score == 0.25
    assert results[0].confidence == 0.75
    assert metric_calls == 1
    assert "retrying without autocast" in caplog.text.lower()


def test_score_tensor_batch_retries_without_autocast_when_forward_fails_under_autocast(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from shotsieve import learned_iqa_backend as backend_module

    autocast_state = {"enabled": False}
    metric_calls: list[bool] = []

    class FakeContextManager:
        def __enter__(self) -> None:
            autocast_state["enabled"] = True

        def __exit__(self, exc_type, exc, tb) -> None:
            autocast_state["enabled"] = False

    class FakeTorch:
        float16 = "float16"

        @staticmethod
        def inference_mode():
            return nullcontext()

        @staticmethod
        def autocast(device_type: str, *, dtype: str) -> FakeContextManager:
            return FakeContextManager()

    def fake_metric(batch_tensor, return_mos=True, return_dist=True):
        metric_calls.append(autocast_state["enabled"])
        if autocast_state["enabled"]:
            raise RuntimeError("mixed precision unsupported")
        return [0.25], [0.75]

    backend = types.SimpleNamespace(
        _torch=FakeTorch,
        runtime="cuda",
        name="topiq_nr",
        metric=fake_metric,
        score_range="0, 1",
        lower_better=False,
    )

    with caplog.at_level(logging.WARNING):
        results = backend_module.score_tensor_batch(
            backend,
            object(),
            flatten_tensor_fn=lambda tensor: [tensor[0]],
            confidence_values_fn=lambda dist_tensor, torch_module: [dist_tensor[0]],
            normalize_score_fn=lambda raw_score, **kwargs: raw_score * 100,
        )

    assert len(results) == 1
    assert results[0].raw_score == 0.25
    assert results[0].confidence == 0.75
    assert metric_calls == [True, False]
    assert "retrying without autocast" in caplog.text.lower()


def test_parse_extensions_expands_raw_alias() -> None:
    extensions = parse_extensions("raw,jpg,heif")

    assert ".cr2" in extensions
    assert ".dng" in extensions
    assert ".jpg" in extensions
    assert ".heif" in extensions


def test_learned_score_range_normalization_helpers() -> None:
    assert parse_score_range("~0, ~1") == (0.0, 1.0)
    assert normalize_score(0.82, score_range="~0, ~1", lower_better=False) == 82.0
    assert normalize_score(20.0, score_range="0, 100", lower_better=True) == 80.0


def test_learned_model_catalog_exposes_all_supported_backends() -> None:
    models = supported_learned_models()
    runtimes = supported_runtime_targets()

    assert "topiq_nr" in models
    assert "topiq_nr-flive" in models
    assert "topiq_nr-spaq" in models
    assert "arniqa" in models
    assert "arniqa-spaq" in models
    assert "qalign" in models
    assert "tres" in models
    assert "clipiqa" in models
    assert "qualiclip" in models
    assert "musiq" not in models
    assert "musiq-spaq" not in models
    assert "maniqa" not in models
    assert "nima" not in models
    assert "directml" in runtimes
    assert "intel" in runtimes
    assert "amd" in runtimes
    assert "mps" in runtimes
    assert "apple" in runtimes


def test_learned_model_aliases_and_runtime_resolution() -> None:
    class NoCudaTorch:
        @staticmethod
        def device(name: str) -> str:
            return name

        class cuda:
            @staticmethod
            def is_available() -> bool:
                return False

        class xpu:
            @staticmethod
            def is_available() -> bool:
                return False

    class CudaTorch:
        @staticmethod
        def device(name: str) -> str:
            return name

        class cuda:
            @staticmethod
            def is_available() -> bool:
                return True

        class xpu:
            @staticmethod
            def is_available() -> bool:
                return False

    class XpuTorch:
        @staticmethod
        def device(name: str) -> str:
            return name

        class cuda:
            @staticmethod
            def is_available() -> bool:
                return False

        class xpu:
            @staticmethod
            def is_available() -> bool:
                return True

    class FakeDirectMlModule:
        @staticmethod
        def default_device() -> int:
            return 0

        @staticmethod
        def device(index: int) -> str:
            return f"dml:{index}"

    class MpsTorch:
        @staticmethod
        def device(name: str) -> str:
            return name

        class cuda:
            @staticmethod
            def is_available() -> bool:
                return False

        class xpu:
            @staticmethod
            def is_available() -> bool:
                return False

        class backends:
            class mps:
                @staticmethod
                def is_available() -> bool:
                    return True

    def import_directml(name: str):
        if name == "torch_directml":
            return FakeDirectMlModule
        raise ImportError(name)

    def import_missing(name: str):
        raise ImportError(name)

    assert normalize_model_name("TOPIQ-NR") == "topiq_nr"
    assert normalize_model_name("topiq_nr_spaq") == "topiq_nr-spaq"
    assert normalize_model_name("Q-Align") == "qalign"
    assert normalize_model_name("Quali-Clip") == "qualiclip"
    assert normalize_device_target("NVIDIA") == "cuda"
    assert normalize_device_target("AMD", system_name="Windows") == "directml"
    assert normalize_device_target("AMD", system_name="Linux") == "amd"
    assert normalize_device_target("Apple", system_name="Darwin") == "mps"
    assert normalize_device_target("Intel") == "intel"
    assert resolve_device(None, torch_module=NoCudaTorch, import_module=import_missing).runtime == "cpu"
    assert resolve_device("auto", torch_module=NoCudaTorch, import_module=import_missing).runtime == "cpu"
    assert resolve_device("cuda", torch_module=NoCudaTorch, import_module=import_missing).runtime == "cpu"
    assert resolve_device(None, torch_module=CudaTorch, import_module=import_missing, system_name="Linux").runtime == "cuda"
    assert resolve_device("cpu", torch_module=CudaTorch, import_module=import_missing).runtime == "cpu"
    assert resolve_device("intel", torch_module=XpuTorch, import_module=import_missing).runtime == "xpu"
    assert resolve_device("amd", torch_module=NoCudaTorch, import_module=import_directml, system_name="Windows").runtime == "directml"
    assert resolve_device("apple", torch_module=MpsTorch, import_module=import_missing, system_name="Darwin").runtime == "mps"
    assert resolve_device("auto", torch_module=MpsTorch, import_module=import_missing, system_name="Darwin").runtime == "mps"

    statuses = runtime_statuses(torch_module=NoCudaTorch, import_module=import_directml, system_name="Windows")
    assert statuses == {
        "cpu": "available",
        "cuda": "unavailable",
        "xpu": "unavailable",
        "directml": "available",
        "mps": "unsupported",
    }

    mac_statuses = runtime_statuses(torch_module=MpsTorch, import_module=import_missing, system_name="Darwin")
    assert mac_statuses == {
        "cpu": "available",
        "cuda": "unavailable",
        "xpu": "unsupported",
        "directml": "unsupported",
        "mps": "available",
    }


def test_runtime_noise_controls_set_env_and_logger_levels(monkeypatch) -> None:
    monkeypatch.delenv("HF_HUB_DISABLE_PROGRESS_BARS", raising=False)
    monkeypatch.delenv("TRANSFORMERS_VERBOSITY", raising=False)
    monkeypatch.delenv("TOKENIZERS_PARALLELISM", raising=False)

    configure_runtime_noise_controls()

    import os
    assert os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS") == "1"
    assert os.environ.get("TRANSFORMERS_VERBOSITY") == "error"
    assert os.environ.get("TOKENIZERS_PARALLELISM") == "false"
    assert logging.getLogger("pyiqa").getEffectiveLevel() >= logging.WARNING
    assert logging.getLogger("huggingface_hub").getEffectiveLevel() >= logging.ERROR


def test_runtime_warning_filter_suppresses_known_noisy_messages() -> None:
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        install_runtime_warning_filters()

        warnings.warn(
            "'torch.load' received a zip file that looks like a TorchScript archive dispatching to 'torch.jit.load'",
            UserWarning,
        )
        warnings.warn(
            "The following generation flags are not valid and may be ignored: ['temperature', 'top_p']",
            UserWarning,
        )
        warnings.warn("`use_return_dict` is deprecated! Use `return_dict` instead!", UserWarning)
        warnings.warn(
            "Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.",
            UserWarning,
        )
        warnings.warn("unrelated warning should still be visible", UserWarning)

    visible_messages = [
        str(entry.message)
        for entry in captured
        if issubclass(entry.category, UserWarning)
    ]
    assert visible_messages == ["unrelated warning should still be visible"]


def test_initialize_database_releases_file_handle_on_return() -> None:
    with TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "shotsieve.db"
        initialize_database(db_path)

        db_path.unlink()

        assert not db_path.exists()


def test_score_paths_returns_explicit_failure_instead_of_fake_midscore(monkeypatch, tmp_path: Path) -> None:
    backend = learned_iqa_module.PyiqaBackend.__new__(learned_iqa_module.PyiqaBackend)
    backend.name = "topiq_nr"
    backend.input_size = 384
    backend._torch = object()
    backend.tensor_device = "cpu"

    image_path = tmp_path / "sample.jpg"
    image_path.write_bytes(b"not-an-image")

    def _always_fail_load(*args, **kwargs):
        raise RuntimeError("forced failure")

    monkeypatch.setattr(learned_iqa_module, "load_batch_tensor", _always_fail_load)

    results = backend.score_paths([image_path], batch_size=1)

    assert len(results) == 1
    assert results[0].raw_score is None
    assert results[0].normalized_score is None
    assert results[0].confidence is None
    assert results[0].error == "forced failure"


def test_pkg_resources_packaging_compat_shim_sets_attribute_when_missing_without_warning() -> None:
    pkg_resources_module = types.SimpleNamespace()
    packaging_module = types.SimpleNamespace(__name__="packaging")

    def fake_import_module(name: str):
        if name == "pkg_resources":
            warnings.warn("pkg_resources is deprecated as an API", UserWarning)
            return pkg_resources_module
        if name == "packaging":
            return packaging_module
        raise ImportError(name)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        learned_iqa_module.ensure_pkg_resources_packaging_compat(import_module=fake_import_module)

    visible_messages = [
        str(entry.message)
        for entry in captured
        if issubclass(entry.category, UserWarning)
    ]

    assert getattr(pkg_resources_module, "packaging", None) is packaging_module
    assert visible_messages == []


def test_import_pyiqa_runtime_suppresses_pkg_resources_warning_during_compat_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from shotsieve import learned_iqa_runtime as runtime_module

    fake_pyiqa = _new_module("pyiqa")
    fake_pyiqa.list_models = lambda metric_mode: ["topiq_nr"]

    fake_torch = _new_module("torch")
    fake_torch.__version__ = "2.11.0+cpu"
    fake_torch.device = lambda name: name
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    fake_torch.xpu = types.SimpleNamespace(is_available=lambda: False)
    fake_torch.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: False)
    )

    pkg_resources_module = types.SimpleNamespace()
    packaging_module = types.SimpleNamespace(__name__="packaging")

    def fake_import_module(name: str):
        if name == "pyiqa":
            return fake_pyiqa
        if name == "torch":
            return fake_torch
        if name == "pkg_resources":
            warnings.warn("pkg_resources is deprecated as an API", UserWarning)
            return pkg_resources_module
        if name == "packaging":
            return packaging_module
        raise ImportError(name)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        pyiqa_module, torch_module = runtime_module.import_pyiqa_runtime(
            import_module=fake_import_module
        )

    visible_messages = [
        str(entry.message)
        for entry in captured
        if issubclass(entry.category, UserWarning)
    ]

    assert pyiqa_module is fake_pyiqa
    assert torch_module is fake_torch
    assert getattr(pkg_resources_module, "packaging", None) is packaging_module
    assert visible_messages == []


def test_import_pyiqa_runtime_uses_injected_import_module_for_optional_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from shotsieve import learned_iqa_runtime as runtime_module

    fake_pyiqa = _new_module("pyiqa")
    fake_torch = _new_module("torch")
    import_calls: list[str] = []
    ensure_calls: list[object] = []

    def fake_import_module(name: str):
        import_calls.append(name)
        if name == "pyiqa":
            return fake_pyiqa
        if name == "torch":
            return fake_torch
        raise ImportError(name)

    def fake_ensure_pkg_resources_packaging_compat(*, import_module) -> None:
        ensure_calls.append(import_module)

    monkeypatch.setattr(
        runtime_module,
        "ensure_pkg_resources_packaging_compat",
        fake_ensure_pkg_resources_packaging_compat,
    )

    pyiqa_module, torch_module = runtime_module.import_pyiqa_runtime(
        import_module=fake_import_module
    )

    assert pyiqa_module is fake_pyiqa
    assert torch_module is fake_torch
    assert import_calls == ["pyiqa", "torch"]
    assert ensure_calls == [fake_import_module]


def test_qalign_not_runtime_compatible_on_cpu_or_directml() -> None:
    assert learned_iqa_module.is_model_runtime_compatible("qalign", torch_version="2.6.0") is True
    assert learned_iqa_module.is_model_runtime_compatible("qalign", torch_version="2.11.0+cpu") is True
    assert learned_iqa_module.is_model_runtime_compatible("qalign", torch_version="2.3.1") is True
    assert learned_iqa_module.is_model_runtime_compatible("qalign", torch_version="2.11.0+cpu", runtime="cpu") is False
    assert learned_iqa_module.is_model_runtime_compatible("qalign", torch_version="2.11.0+cpu", runtime="directml") is False
    assert learned_iqa_module.is_model_runtime_compatible("qalign", torch_version=None) is True


@pytest.mark.parametrize("runtime", ["cpu", "directml"])
def test_pyiqa_backend_blocks_qalign_on_explicit_incompatible_runtime(
    monkeypatch: pytest.MonkeyPatch,
    runtime: str,
) -> None:
    class FakePyiqa:
        __version__ = "0.1-test"

        @staticmethod
        def list_models(metric_mode: str):
            assert metric_mode == "NR"
            return ["topiq_nr", "clipiqa", "qalign"]

    class FakeTorch:
        __version__ = "2.11.0+cpu"

        @staticmethod
        def device(name: str) -> str:
            return name

        class cuda:
            @staticmethod
            def is_available() -> bool:
                return False

        class xpu:
            @staticmethod
            def is_available() -> bool:
                return False

    monkeypatch.setattr(learned_iqa_module, "import_pyiqa_runtime", lambda: (FakePyiqa, FakeTorch))
    monkeypatch.setattr(
        learned_iqa_module,
        "resolve_device",
        lambda device, torch_module: types.SimpleNamespace(
            runtime=runtime,
            metric_device=runtime,
            tensor_device=runtime,
            display_device=runtime,
        ),
    )

    with pytest.raises(
        learned_iqa_module.LearnedBackendUnavailableError,
        match=f"runtime '{runtime}'",
    ):
        learned_iqa_module.PyiqaBackend("qalign", device=runtime)


@pytest.mark.parametrize("runtime", ["cpu", "directml"])
def test_resolve_learned_model_version_blocks_qalign_on_explicit_incompatible_runtime(
    monkeypatch: pytest.MonkeyPatch,
    runtime: str,
) -> None:
    class FakePyiqa:
        __version__ = "0.1-test"

        @staticmethod
        def list_models(metric_mode: str):
            assert metric_mode == "NR"
            return ["topiq_nr", "clipiqa", "qalign"]

    class FakeTorch:
        __version__ = "2.11.0+cpu"

        @staticmethod
        def device(name: str) -> str:
            return name

        class cuda:
            @staticmethod
            def is_available() -> bool:
                return False

        class xpu:
            @staticmethod
            def is_available() -> bool:
                return False

    monkeypatch.setattr(learned_iqa_module, "import_pyiqa_runtime", lambda: (FakePyiqa, FakeTorch))
    monkeypatch.setattr(
        learned_iqa_module,
        "resolve_device",
        lambda device, torch_module: types.SimpleNamespace(
            runtime=runtime,
            metric_device=runtime,
            tensor_device=runtime,
            display_device=runtime,
        ),
    )

    with pytest.raises(
        learned_iqa_module.LearnedBackendUnavailableError,
        match=f"runtime '{runtime}'",
    ):
        learned_iqa_module.resolve_learned_model_version("qalign", device=runtime)


def test_runtime_compatible_model_names_filters_qalign_from_cpu_and_directml() -> None:
    models = ["topiq_nr", "clipiqa", "qalign"]

    filtered_directml = learned_iqa_module.runtime_compatible_model_names(
        models,
        torch_version="2.11.0+cpu",
        runtime="directml",
    )
    filtered_cpu = learned_iqa_module.runtime_compatible_model_names(
        models,
        torch_version="2.11.0+cpu",
        runtime="cpu",
    )

    assert "qalign" not in filtered_directml
    assert filtered_directml == ["topiq_nr", "clipiqa"]
    assert "qalign" not in filtered_cpu
    assert filtered_cpu == ["topiq_nr", "clipiqa"]


def test_available_backends_excludes_qalign_on_cpu_runtime_even_on_older_torch(monkeypatch) -> None:
    class FakePyiqa:
        @staticmethod
        def list_models(metric_mode: str):
            assert metric_mode == "NR"
            return ["topiq_nr", "arniqa", "qalign"]

    class FakeTorch:
        __version__ = "2.3.1"

        @staticmethod
        def device(name: str) -> str:
            return name

        class cuda:
            @staticmethod
            def is_available() -> bool:
                return False

        class xpu:
            @staticmethod
            def is_available() -> bool:
                return False

    monkeypatch.setattr(learned_iqa_module, "import_pyiqa_runtime", lambda: (FakePyiqa, FakeTorch))

    payload = learned_iqa_module.available_learned_backends()

    available_text = str(payload["modern_models_available"] or "")
    available = available_text.split(",") if available_text else []
    assert "qalign" not in available


def test_available_backends_excludes_qalign_on_cpu_runtime_with_new_torch(monkeypatch) -> None:
    class FakePyiqa:
        @staticmethod
        def list_models(metric_mode: str):
            assert metric_mode == "NR"
            return ["topiq_nr", "clipiqa", "qalign"]

    class FakeTorch:
        __version__ = "2.11.0+cpu"

        @staticmethod
        def device(name: str) -> str:
            return name

        class cuda:
            @staticmethod
            def is_available() -> bool:
                return False

        class xpu:
            @staticmethod
            def is_available() -> bool:
                return False

    monkeypatch.setattr(learned_iqa_module, "import_pyiqa_runtime", lambda: (FakePyiqa, FakeTorch))

    payload = learned_iqa_module.available_learned_backends()
    available_text = str(payload["modern_models_available"] or "")
    available = available_text.split(",") if available_text else []

    assert "qalign" not in available


def test_torch_load_cve_bypass_patches_and_restores_torch_load(monkeypatch: pytest.MonkeyPatch) -> None:
    call_log: list[dict] = []

    def fake_load(*args, **kwargs):
        call_log.append(kwargs.copy())
        return None

    fake_torch = types.SimpleNamespace(load=fake_load)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    with learned_iqa_module._bypass_torch_load_cve_check():
        fake_torch.load("dummy.bin")

    assert call_log[0]["weights_only"] is False

    assert fake_torch.load is fake_load


def test_torch_load_cve_bypass_respects_explicit_weights_only_argument(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_log: list[dict] = []

    def fake_load(*args, **kwargs):
        call_log.append(kwargs.copy())
        return None

    fake_torch = types.SimpleNamespace(load=fake_load)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    with learned_iqa_module._bypass_torch_load_cve_check():
        fake_torch.load("dummy.bin", weights_only=True)

    assert call_log[0]["weights_only"] is True
    assert fake_torch.load is fake_load


def test_detect_hardware_capabilities_cache_can_be_invalidated(monkeypatch: pytest.MonkeyPatch) -> None:
    state = {"vram_mb": 2048}

    monkeypatch.setattr(learned_iqa_module, "_cached_hw_capabilities", None)
    monkeypatch.setattr(learned_iqa_module, "_effective_cpu_count", lambda: 8)
    monkeypatch.setattr(learned_iqa_module, "detect_system_ram_mb", lambda: 16384)
    monkeypatch.setattr(learned_iqa_module, "detect_gpu_vram_mb", lambda: state["vram_mb"])

    first = learned_iqa_module.detect_hardware_capabilities()
    state["vram_mb"] = 4096
    second = learned_iqa_module.detect_hardware_capabilities()

    assert first == {"cpu_count": 8, "ram_mb": 16384, "vram_mb": 2048}
    assert second is first
    assert second["vram_mb"] == 2048

    learned_iqa_module.invalidate_hw_cache()
    refreshed = learned_iqa_module.detect_hardware_capabilities()

    assert refreshed == {"cpu_count": 8, "ram_mb": 16384, "vram_mb": 4096}
    assert refreshed is not first


def test_available_backends_fallback_includes_runtime_and_hardware_when_pyiqa_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        learned_iqa_module,
        "import_pyiqa_runtime",
        lambda: (_ for _ in ()).throw(ImportError("pyiqa missing")),
    )
    monkeypatch.setattr(
        learned_iqa_module,
        "_runtime_status_text_from_torch_import",
        lambda import_module=learned_iqa_module.importlib.import_module, system_name=None: (
            "cuda:available,xpu:unavailable,directml:not-installed,mps:unsupported,cpu:available"
        ),
    )
    monkeypatch.setattr(
        learned_iqa_module,
        "detect_hardware_capabilities",
        lambda: {"cpu_count": 16, "ram_mb": 65536, "vram_mb": 24576},
    )

    payload = learned_iqa_module.available_learned_backends(resource_profile="normal")

    assert payload["pyiqa"] == "not-installed"
    assert payload["runtime_status"] == "cuda:available,xpu:unavailable,directml:not-installed,mps:unsupported,cpu:available"
    assert payload["hardware"] == {"cpu_count": 16, "ram_mb": 65536, "vram_mb": 24576}
    assert payload["resource_profile"] == "normal"
    assert isinstance(payload["recommended_batch_sizes"], dict)
    assert payload["recommended_batch_sizes"]["topiq_nr"] >= 1


def test_available_backends_tolerates_runtime_probe_exceptions(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakePyiqa:
        @staticmethod
        def list_models(metric_mode: str):
            assert metric_mode == "NR"
            return ["topiq_nr"]

    class FakeTorch:
        __version__ = "2.9.0"

        @staticmethod
        def device(name: str) -> str:
            return name

        class cuda:
            @staticmethod
            def is_available() -> bool:
                raise RuntimeError("cuda probe failed")

        class xpu:
            @staticmethod
            def is_available() -> bool:
                raise RuntimeError("xpu probe failed")

    monkeypatch.setattr(learned_iqa_module, "import_pyiqa_runtime", lambda: (FakePyiqa, FakeTorch))

    payload = learned_iqa_module.available_learned_backends()

    assert payload["pyiqa"] == "installed"
    assert payload["default_runtime"] == "cpu"
    assert payload["default_device"] == "cpu"


def test_runtime_statuses_tolerates_directml_import_runtime_error() -> None:
    class NoAccelTorch:
        @staticmethod
        def device(name: str) -> str:
            return name

        class cuda:
            @staticmethod
            def is_available() -> bool:
                return False

        class xpu:
            @staticmethod
            def is_available() -> bool:
                return False

    def import_directml_runtime_error(name: str):
        if name == "torch_directml":
            raise RuntimeError("incompatible torch_directml runtime")
        raise ImportError(name)

    statuses = runtime_statuses(
        torch_module=NoAccelTorch,
        import_module=import_directml_runtime_error,
        system_name="Windows",
    )

    assert statuses["directml"] == "not-installed"


def test_available_backends_tolerates_incomplete_torch_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakePyiqa:
        @staticmethod
        def list_models(metric_mode: str):
            assert metric_mode == "NR"
            return ["topiq_nr"]

    class IncompleteTorch:
        __version__ = "0.0-test"

    monkeypatch.setattr(
        learned_iqa_module,
        "import_pyiqa_runtime",
        lambda: (FakePyiqa, IncompleteTorch),
    )

    payload = learned_iqa_module.available_learned_backends()

    assert payload["pyiqa"] == "unavailable"
    assert payload["default_runtime"] == "cpu"
    assert payload["default_model"] == "topiq_nr"