from contextlib import nullcontext
import logging
import types
from pathlib import Path
from typing import Any

import pytest




def _new_module(name: str) -> Any:
    return types.ModuleType(name)

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


