import types
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

import numpy as np
import pytest
import shotsieve.learned_iqa as learned_iqa_module
from shotsieve.db import initialize_database
from shotsieve.config import parse_extensions
from shotsieve.learned_iqa import (
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


