from __future__ import annotations

from importlib.machinery import ModuleSpec
import sys
from types import ModuleType, SimpleNamespace

import pytest

from shotsieve import desktop
from shotsieve import learned_iqa_runtime as runtime


def _torch_runtime(name: str, *, available: bool = True) -> ModuleType:
    torch = ModuleType("torch")
    torch.__spec__ = ModuleSpec("torch", loader=None)
    torch.device = str
    torch.version = SimpleNamespace(hip="7.2.1" if name == "rocm" else None)
    properties = SimpleNamespace(total_memory=8192 * 1024 * 1024)
    torch.cuda = SimpleNamespace(
        is_available=lambda: available and name in {"cuda", "rocm"},
        get_arch_list=lambda: ["gfx1100", "gfx1151"] if name == "rocm" else ["sm_120"],
        get_device_capability=lambda: (11, 0) if name == "rocm" else (12, 0),
        get_device_properties=lambda _index: properties,
    )
    torch.xpu = SimpleNamespace(
        is_available=lambda: available and name == "xpu",
        get_device_properties=lambda _index: properties,
    )
    torch.backends = SimpleNamespace(
        mps=SimpleNamespace(is_available=lambda: available and name == "mps"),
    )
    return torch


_TARGETS = (
    ("Windows", "windows-nvidia-cuda", "cuda", "cuda"),
    ("Linux", "linux-nvidia-cuda", "cuda", "cuda"),
    ("Windows", "windows-amd-rocm", "rocm", "cuda"),
    ("Linux", "linux-amd-rocm", "rocm", "cuda"),
    ("Windows", "windows-intel-xpu", "xpu", "xpu"),
    ("Linux", "linux-intel-xpu", "xpu", "xpu"),
    ("Darwin", "macos-apple-mps", "mps", "mps"),
)


@pytest.mark.parametrize(("system", "target", "name", "tensor_device"), _TARGETS)
def test_accelerator_identity_flows_from_launcher_to_model_device(
    monkeypatch, system, target, name, tensor_device,
) -> None:
    torch = _torch_runtime(name)
    monkeypatch.setitem(sys.modules, "torch", torch)

    assert desktop.runtime_bundle_has_usable_torch(target) is True
    for requested in (name, "auto"):
        resolved = runtime.resolve_device(requested, torch_module=torch, system_name=system)
        assert resolved.runtime == name
        assert resolved.display_device == name
        assert resolved.tensor_device == tensor_device
        assert resolved.metric_device == tensor_device
        assert resolved.fallback_reason is None

    statuses = runtime.runtime_statuses(torch_module=torch, system_name=system)
    assert statuses[name] == "available"
    if name == "rocm":
        assert statuses["cuda"] == "unavailable"
        assert runtime.has_cuda(torch) is False
        # AMD architecture strings must never enter the NVIDIA SM check.
        assert runtime.has_rocm(torch) is True


@pytest.mark.parametrize(("system", "target", "name", "tensor_device"), _TARGETS)
def test_missing_accelerator_falls_back_only_for_auto(
    monkeypatch, system, target, name, tensor_device,
) -> None:
    torch = _torch_runtime(name, available=False)
    monkeypatch.setitem(sys.modules, "torch", torch)

    assert desktop.runtime_bundle_has_usable_torch(target) is False
    assert runtime.resolve_device("auto", torch_module=torch, system_name=system).runtime == "cpu"
    with pytest.raises(runtime.LearnedRuntimeUnavailableError):
        runtime.resolve_device(name, torch_module=torch, system_name=system)


@pytest.mark.parametrize(("name", "expected_mb"), (
    ("cuda", 8192), ("rocm", 8192), ("xpu", 8192), ("mps", 24576),
))
def test_memory_detection_uses_the_selected_torch_backend(monkeypatch, name, expected_mb) -> None:
    def unexpected_fallback():
        pytest.fail("Torch memory properties should take precedence over host heuristics")

    monkeypatch.setattr(runtime, "_detect_vram_windows_registry", unexpected_fallback)
    monkeypatch.setattr(runtime, "_detect_vram_linux_nvidia_smi", unexpected_fallback)
    monkeypatch.setattr(runtime, "_detect_vram_linux_amd", unexpected_fallback)
    assert runtime.detect_gpu_vram_mb(
        torch_module=_torch_runtime(name), detect_system_ram_mb_fn=lambda: 32768,
    ) == expected_mb


@pytest.mark.parametrize(("system", "target", "name", "tensor_device"), _TARGETS)
@pytest.mark.parametrize("available", (True, False))
def test_sidecar_reprobe_keeps_initialized_native_torch(
    monkeypatch, system, target, name, tensor_device, available,
) -> None:
    torch = _torch_runtime(name, available=available)
    torch_nn = ModuleType("torch.nn")
    torchvision = ModuleType("torchvision")
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.nn", torch_nn)
    monkeypatch.setitem(sys.modules, "torchvision", torchvision)

    for _ in range(2):
        assert desktop.runtime_bundle_has_usable_torch(target, force_reload=True) is available
        assert sys.modules["torch"] is torch
        assert sys.modules["torch.nn"] is torch_nn
        assert sys.modules["torchvision"] is torchvision


def test_failed_torch_import_can_clear_orphan_submodules(monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setitem(sys.modules, "torch.nn", ModuleType("torch.nn"))
    desktop._clear_failed_torch_imports()
    assert "torch.nn" not in sys.modules


def test_missing_runtime_diagnostics_respect_apple_platform(monkeypatch) -> None:
    def missing_import(_name):
        raise ImportError("not installed")

    monkeypatch.setattr(runtime, "detect_hardware_capabilities", lambda: {})
    payload = runtime.unavailable_backend_payload(
        status="unavailable", import_module=missing_import, system_name="Darwin",
    )
    assert payload["auto_runtime_priority"] == "mps,cpu"
    assert "mps:unavailable" in payload["runtime_status"]
    assert "rocm:unsupported" in payload["runtime_status"]
