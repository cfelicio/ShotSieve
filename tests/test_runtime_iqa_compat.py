import logging
import sys
import types
from typing import Any
import warnings

import pytest
import shotsieve.learned_iqa as learned_iqa_module
from shotsieve.learned_iqa import (
    configure_runtime_noise_controls,
    install_runtime_warning_filters,
    runtime_statuses,
)




def _new_module(name: str) -> Any:
    return types.ModuleType(name)

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