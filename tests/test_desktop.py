from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from pathlib import Path
from typing import cast

import shotsieve.desktop as desktop_module
import pytest
from shotsieve import runtime_support


def test_resolve_cuda_runtime_target_id_only_returns_cuda_targets() -> None:
    assert desktop_module._resolve_cuda_runtime_target_id("windows-nvidia") == "windows-nvidia"
    assert desktop_module._resolve_cuda_runtime_target_id("linux-nvidia") == "linux-nvidia"
    assert desktop_module._resolve_cuda_runtime_target_id("windows-cpu") is None
    assert desktop_module._resolve_cuda_runtime_target_id(None) is None


def test_default_data_dir_prefers_local_app_data(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    local_app_data = tmp_path / "AppData" / "Local"
    monkeypatch.setenv("LOCALAPPDATA", str(local_app_data))
    monkeypatch.delenv("APPDATA", raising=False)
    installed_module = tmp_path / "venv" / "Lib" / "site-packages" / "shotsieve" / "desktop.py"
    installed_module.parent.mkdir(parents=True)
    installed_module.write_text("# installed module\n", encoding="utf-8")
    monkeypatch.setattr(desktop_module, "__file__", str(installed_module))

    result = desktop_module.default_data_dir()

    assert result == local_app_data.resolve() / desktop_module.APP_DIRNAME


def test_default_data_dir_prefers_repo_local_data_for_source_checkout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    local_app_data = tmp_path / "AppData" / "Local"
    monkeypatch.setenv("LOCALAPPDATA", str(local_app_data))

    project_root = tmp_path / "shotsieve-src"
    module_path = project_root / "src" / "shotsieve" / "desktop.py"
    module_path.parent.mkdir(parents=True)
    module_path.write_text("# source module\n", encoding="utf-8")
    (project_root / "src" / "shotsieve" / "__init__.py").write_text("", encoding="utf-8")
    (project_root / "pyproject.toml").write_text("[project]\nname='shotsieve'\n", encoding="utf-8")
    monkeypatch.setattr(desktop_module, "__file__", str(module_path))

    result = desktop_module.default_data_dir()

    assert result == (project_root / "data").resolve()
    assert result != local_app_data.resolve() / desktop_module.APP_DIRNAME


def test_default_data_dir_prefers_portable_internal_dir_for_frozen_build(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    local_app_data = tmp_path / "AppData" / "Local"
    monkeypatch.setenv("LOCALAPPDATA", str(local_app_data))

    portable_dir = tmp_path / "portable"
    portable_dir.mkdir(parents=True)
    portable_exe = portable_dir / "ShotSieve-NVIDIA.exe"
    portable_exe.write_bytes(b"binary")

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(portable_exe), raising=False)

    result = desktop_module.default_data_dir()

    assert result == (portable_dir / "data").resolve()
    assert result != local_app_data.resolve() / desktop_module.APP_DIRNAME


def test_main_uses_default_data_dir_and_no_browser(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    project_root = tmp_path / "shotsieve-src"
    module_path = project_root / "src" / "shotsieve" / "desktop.py"
    module_path.parent.mkdir(parents=True)
    module_path.write_text("# source module\n", encoding="utf-8")
    (project_root / "src" / "shotsieve" / "__init__.py").write_text("", encoding="utf-8")
    (project_root / "pyproject.toml").write_text("[project]\nname='shotsieve'\n", encoding="utf-8")
    monkeypatch.setattr(desktop_module, "__file__", str(module_path))
    called: dict[str, object] = {}

    def fake_serve_review_ui(*, db_path: Path, host: str, port: int, open_browser: bool) -> None:
        called.update({
            "db_path": db_path,
            "host": host,
            "port": port,
            "open_browser": open_browser,
        })

    monkeypatch.setattr(desktop_module, "serve_review_ui", fake_serve_review_ui)
    monkeypatch.setattr(desktop_module, "maybe_prepare_cuda_torch_runtime", lambda data_dir: None)
    monkeypatch.setattr(desktop_module, "maybe_prepare_learned_iqa_runtime", lambda data_dir: None)
    monkeypatch.setattr(sys, "argv", ["shotsieve-desktop", "--port", "9001", "--no-browser"])

    desktop_module.main()

    expected_data_dir = (project_root / "data").resolve()
    assert expected_data_dir.exists()
    assert called["db_path"] == expected_data_dir / "shotsieve.db"
    assert called["host"] == "127.0.0.1"
    assert called["port"] == 9001
    assert called["open_browser"] is False


def test_main_uses_custom_data_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    custom_data_dir = tmp_path / "custom-data"
    called: dict[str, object] = {}

    def fake_serve_review_ui(*, db_path: Path, host: str, port: int, open_browser: bool) -> None:
        called.update({
            "db_path": db_path,
            "host": host,
            "port": port,
            "open_browser": open_browser,
        })

    monkeypatch.setattr(desktop_module, "serve_review_ui", fake_serve_review_ui)
    monkeypatch.setattr(
        sys,
        "argv",
        ["shotsieve-desktop", "--data-dir", str(custom_data_dir), "--host", "0.0.0.0"],
    )

    desktop_module.main()

    assert custom_data_dir.resolve().exists()
    assert called["db_path"] == custom_data_dir.resolve() / "shotsieve.db"
    assert called["host"] == "0.0.0.0"
    assert called["port"] == 8765
    assert called["open_browser"] is True


def test_runtime_target_id_from_executable_name_detects_windows_nvidia(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", r"C:\\tmp\\ShotSieve-NVIDIA.exe", raising=False)

    target_id = desktop_module.runtime_target_id_from_executable_name(system_name="Windows")

    assert target_id == "windows-nvidia"


def test_maybe_prepare_cuda_torch_runtime_installs_sidecar_when_auto_enabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)

    monkeypatch.setenv("SHOTSIEVE_BOOTSTRAP_AUTO_INSTALL_TORCH", "1")
    monkeypatch.setattr(desktop_module, "runtime_target_id_from_executable_name", lambda system_name=None: "windows-nvidia")
    monkeypatch.setattr(desktop_module, "runtime_bundle_has_usable_cuda_torch", lambda force_reload=False: False)

    calls: list[tuple[str, Path, bool]] = []

    def fake_install_torch_sidecar(
        *, runtime: str, site_packages: Path, output_func=print, force_reinstall: bool = False
    ) -> bool:
        calls.append((runtime, site_packages, force_reinstall))
        (site_packages / "torch").mkdir(parents=True, exist_ok=True)
        (site_packages / "torch" / "__init__.py").write_text("", encoding="utf-8")
        return True

    monkeypatch.setattr(desktop_module, "install_torch_sidecar", fake_install_torch_sidecar)

    desktop_module.maybe_prepare_cuda_torch_runtime(data_dir)

    assert len(calls) == 1
    assert calls[0][0] == "cuda"
    assert str(calls[0][1]) in sys.path
    assert calls[0][2] is False


def test_maybe_prepare_cuda_torch_runtime_invalidates_hardware_cache_after_successful_install(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)

    monkeypatch.setenv("SHOTSIEVE_BOOTSTRAP_AUTO_INSTALL_TORCH", "1")
    monkeypatch.setattr(desktop_module, "runtime_target_id_from_executable_name", lambda system_name=None: "windows-nvidia")
    monkeypatch.setattr(desktop_module, "runtime_bundle_has_usable_cuda_torch", lambda force_reload=False: False)
    monkeypatch.setattr(desktop_module, "_sidecar_torch_has_usable_cuda", lambda path: True)

    invalidations: list[str] = []
    monkeypatch.setattr(desktop_module, "invalidate_hw_cache", lambda: invalidations.append("torch"), raising=False)

    def fake_install_torch_sidecar(
        *, runtime: str, site_packages: Path, output_func=print, force_reinstall: bool = False
    ) -> bool:
        (site_packages / "torch").mkdir(parents=True, exist_ok=True)
        (site_packages / "torch" / "__init__.py").write_text("", encoding="utf-8")
        return True

    monkeypatch.setattr(desktop_module, "install_torch_sidecar", fake_install_torch_sidecar)

    desktop_module.maybe_prepare_cuda_torch_runtime(data_dir)

    assert invalidations == ["torch"]


def test_maybe_prepare_cuda_torch_runtime_non_interactive_defaults_to_auto_install(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)

    monkeypatch.delenv("SHOTSIEVE_BOOTSTRAP_AUTO_INSTALL_TORCH", raising=False)
    monkeypatch.setattr(desktop_module, "runtime_target_id_from_executable_name", lambda system_name=None: "windows-nvidia")
    monkeypatch.setattr(desktop_module, "runtime_bundle_has_usable_cuda_torch", lambda force_reload=False: False)
    monkeypatch.setattr(desktop_module, "_is_interactive_console", lambda: False)

    calls: list[tuple[str, Path, bool]] = []

    def fake_install_torch_sidecar(
        *, runtime: str, site_packages: Path, output_func=print, force_reinstall: bool = False
    ) -> bool:
        calls.append((runtime, site_packages, force_reinstall))
        (site_packages / "torch").mkdir(parents=True, exist_ok=True)
        (site_packages / "torch" / "__init__.py").write_text("", encoding="utf-8")
        return True

    monkeypatch.setattr(desktop_module, "install_torch_sidecar", fake_install_torch_sidecar)

    messages: list[str] = []
    desktop_module.maybe_prepare_cuda_torch_runtime(data_dir, output_func=messages.append)

    assert len(calls) == 1
    assert calls[0][0] == "cuda"
    assert calls[0][2] is False
    assert any("PyTorch" in message for message in messages)


def test_runtime_pythonpath_updates_prepend_sidecar_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    existing_a = str((tmp_path / "existing-a").resolve())
    existing_b = str((tmp_path / "existing-b").resolve())
    sidecar = (tmp_path / "runtime" / "site-packages" / "windows-nvidia").resolve()

    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([existing_a, existing_b]))
    monkeypatch.setattr(sys, "path", [existing_a, existing_b], raising=False)

    desktop_module._prepend_runtime_pythonpath(sidecar)

    updated_parts = os.environ["PYTHONPATH"].split(os.pathsep)
    assert updated_parts[0] == str(sidecar)
    assert updated_parts[1] == existing_a
    assert updated_parts[2] == existing_b
    assert sys.path[0] == str(sidecar)


def test_runtime_pythonpath_updates_use_shared_composer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    existing = str((tmp_path / "existing-a").resolve())
    sidecar = (tmp_path / "runtime" / "site-packages" / "windows-nvidia").resolve()

    monkeypatch.setenv("PYTHONPATH", existing)
    monkeypatch.setattr(sys, "path", [existing], raising=False)

    compose_calls: list[tuple[str | None, Path]] = []

    def fake_compose_pythonpath(*, existing: str | None, prepend_path: Path) -> str:
        compose_calls.append((existing, prepend_path))
        return "shared-pythonpath"

    monkeypatch.setattr(runtime_support, "compose_pythonpath", fake_compose_pythonpath)

    desktop_module._prepend_runtime_pythonpath(sidecar)

    assert compose_calls == [(existing, sidecar)]
    assert os.environ["PYTHONPATH"] == "shared-pythonpath"
    assert sys.path[0] == str(sidecar)


def test_desktop_runtime_support_wrappers_delegate_to_shared_module(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(runtime_support, "path_has_torch", lambda path: calls.append(("torch", path)) or True)
    monkeypatch.setattr(runtime_support, "path_has_pyiqa", lambda path: calls.append(("pyiqa", path)) or False)
    monkeypatch.setattr(runtime_support, "parse_env_bool", lambda value: calls.append(("bool", value)) or True)
    monkeypatch.setattr(runtime_support, "is_interactive_console", lambda: calls.append(("interactive", None)) or False)

    def fake_confirm(prompt: str, *, input_func=input) -> bool:
        calls.append(("confirm", prompt))
        return True

    monkeypatch.setattr(runtime_support, "confirm", fake_confirm)
    monkeypatch.setattr(
        runtime_support,
        "compose_pythonpath",
        lambda *, existing, prepend_path: calls.append(("pythonpath", (existing, prepend_path))) or "shared-pythonpath",
    )

    assert desktop_module._path_has_torch(tmp_path) is True
    assert desktop_module._path_has_pyiqa(tmp_path) is False
    assert desktop_module._parse_env_bool("yes") is True
    assert desktop_module._is_interactive_console() is False
    assert desktop_module._confirm("Install now?") is True
    assert desktop_module._compose_pythonpath(existing="existing", prepend_path=tmp_path) == "shared-pythonpath"

    assert calls == [
        ("torch", tmp_path),
        ("pyiqa", tmp_path),
        ("bool", "yes"),
        ("interactive", None),
        ("confirm", "Install now?"),
        ("pythonpath", ("existing", tmp_path)),
    ]


def test_runtime_bundle_has_usable_cuda_torch_false_when_import_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(desktop_module.importlib.util, "find_spec", lambda name: object() if name == "torch" else None)

    def fake_import_module(name: str):
        if name == "torch":
            raise ImportError("boom")
        return object()

    monkeypatch.setattr(desktop_module.importlib, "import_module", fake_import_module)

    assert desktop_module.runtime_bundle_has_usable_cuda_torch() is False


def test_runtime_bundle_has_usable_cuda_torch_false_when_cuda_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(desktop_module.importlib.util, "find_spec", lambda name: object() if name == "torch" else None)
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setattr(desktop_module.importlib, "import_module", lambda name: fake_torch)

    assert desktop_module.runtime_bundle_has_usable_cuda_torch() is False


def test_runtime_bundle_has_usable_cuda_torch_true_when_cuda_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(desktop_module.importlib.util, "find_spec", lambda name: object() if name == "torch" else None)
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
    monkeypatch.setattr(desktop_module.importlib, "import_module", lambda name: fake_torch)

    assert desktop_module.runtime_bundle_has_usable_cuda_torch() is True


def test_runtime_has_learned_iqa_false_when_import_errors_with_missing_stdlib(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(desktop_module.importlib.util, "find_spec", lambda name: object() if name == "pyiqa" else None)

    def fake_import_module(name: str):
        if name == "pyiqa":
            raise ModuleNotFoundError("No module named 'modulefinder'")
        return object()

    monkeypatch.setattr(desktop_module.importlib, "import_module", fake_import_module)

    assert desktop_module._runtime_has_learned_iqa() is False


def test_learned_iqa_runtime_diagnostic_suggests_sympy_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(desktop_module.importlib.util, "find_spec", lambda name: object() if name == "pyiqa" else None)

    def fake_import_module(name: str):
        if name == "pyiqa":
            exc = ModuleNotFoundError("No module named 'sympy'")
            exc.name = "sympy"
            raise exc
        return object()

    monkeypatch.setattr(desktop_module.importlib, "import_module", fake_import_module)

    diagnostic = desktop_module._learned_iqa_runtime_import_diagnostic()

    assert diagnostic is not None
    assert "suggested package: sympy" in diagnostic


def test_runtime_has_learned_iqa_refreshes_import_caches_before_detection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {"invalidated": False}

    def fake_invalidate_caches() -> None:
        state["invalidated"] = True

    def fake_find_spec(name: str):
        if name != "pyiqa":
            return None
        return object() if state["invalidated"] else None

    def fake_import_module(name: str):
        if name == "pyiqa" and state["invalidated"]:
            return object()
        raise ImportError(name)

    monkeypatch.setattr(desktop_module.importlib, "invalidate_caches", fake_invalidate_caches)
    monkeypatch.setattr(desktop_module.importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setattr(desktop_module.importlib, "import_module", fake_import_module)

    assert desktop_module._runtime_has_learned_iqa() is True


def test_runtime_has_learned_iqa_clears_stale_module_cache_before_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel_key = "pyiqa"
    module_cache = cast(dict[str, object], sys.modules)
    had_original = sentinel_key in module_cache
    original_value = module_cache.get(sentinel_key)
    module_cache[sentinel_key] = None

    def fake_find_spec(name: str):
        return object() if name == "pyiqa" else None

    def fake_import_module(name: str):
        if name != "pyiqa":
            raise ImportError(name)
        if "pyiqa" in sys.modules and sys.modules.get("pyiqa") is None:
            raise ModuleNotFoundError("No module named 'pyiqa'")
        return object()

    monkeypatch.setattr(desktop_module.importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setattr(desktop_module.importlib, "import_module", fake_import_module)

    try:
        assert desktop_module._runtime_has_learned_iqa() is True
    finally:
        if had_original:
            module_cache[sentinel_key] = original_value
        else:
            module_cache.pop(sentinel_key, None)


def test_sidecar_cuda_probe_uses_runtime_import_check(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    site_packages = (tmp_path / "runtime" / "site-packages" / "windows-nvidia").resolve()
    site_packages.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(desktop_module, "runtime_bundle_has_usable_cuda_torch", lambda force_reload=False: force_reload)

    assert desktop_module._sidecar_torch_has_usable_cuda(site_packages) is True


def test_maybe_prepare_cuda_torch_runtime_attempts_repair_install_when_sidecar_exists_but_unusable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)

    monkeypatch.delenv("SHOTSIEVE_BOOTSTRAP_AUTO_INSTALL_TORCH", raising=False)
    monkeypatch.setattr(desktop_module, "_is_interactive_console", lambda: False)
    monkeypatch.setattr(desktop_module, "runtime_target_id_from_executable_name", lambda system_name=None: "windows-nvidia")

    check_calls: list[bool] = []

    def fake_runtime_bundle_has_usable_cuda_torch(*, force_reload: bool = False) -> bool:
        check_calls.append(force_reload)
        return False

    monkeypatch.setattr(desktop_module, "runtime_bundle_has_usable_cuda_torch", fake_runtime_bundle_has_usable_cuda_torch)

    runtime_root = (data_dir / "runtime").resolve()
    site_packages = desktop_module.sidecar_site_packages_dir(runtime_root, "windows-nvidia")
    (site_packages / "torch").mkdir(parents=True, exist_ok=True)
    (site_packages / "torch" / "__init__.py").write_text("", encoding="utf-8")

    install_calls: list[tuple[str, Path, bool]] = []

    monkeypatch.setattr(desktop_module, "_sidecar_torch_has_usable_cuda", lambda path: False)

    def fake_install_torch_sidecar(
        *, runtime: str, site_packages: Path, output_func=print, force_reinstall: bool = False
    ) -> bool:
        install_calls.append((runtime, site_packages, force_reinstall))
        return True

    monkeypatch.setattr(desktop_module, "install_torch_sidecar", fake_install_torch_sidecar)

    messages: list[str] = []
    desktop_module.maybe_prepare_cuda_torch_runtime(data_dir, output_func=messages.append)

    assert check_calls == [False]
    assert len(install_calls) == 1
    assert install_calls[0][0] == "cuda"
    assert install_calls[0][2] is True
    assert any("attempting repair installation" in message.casefold() for message in messages)


def test_maybe_prepare_cuda_torch_runtime_skips_reinstall_when_explicitly_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)

    monkeypatch.setenv("SHOTSIEVE_BOOTSTRAP_AUTO_INSTALL_TORCH", "0")
    monkeypatch.setattr(desktop_module, "runtime_target_id_from_executable_name", lambda system_name=None: "windows-nvidia")
    monkeypatch.setattr(desktop_module, "runtime_bundle_has_usable_cuda_torch", lambda force_reload=False: False)
    monkeypatch.setattr(desktop_module, "_sidecar_torch_has_usable_cuda", lambda path: False)

    runtime_root = (data_dir / "runtime").resolve()
    site_packages = desktop_module.sidecar_site_packages_dir(runtime_root, "windows-nvidia")
    (site_packages / "torch").mkdir(parents=True, exist_ok=True)
    (site_packages / "torch" / "__init__.py").write_text("", encoding="utf-8")

    install_calls: list[tuple[str, Path, bool]] = []

    def fake_install_torch_sidecar(
        *, runtime: str, site_packages: Path, output_func=print, force_reinstall: bool = False
    ) -> bool:
        install_calls.append((runtime, site_packages, force_reinstall))
        return True

    monkeypatch.setattr(desktop_module, "install_torch_sidecar", fake_install_torch_sidecar)

    messages: list[str] = []
    desktop_module.maybe_prepare_cuda_torch_runtime(data_dir, output_func=messages.append)

    assert install_calls == []
    assert any("skipping reinstall" in message.casefold() for message in messages)


def test_main_invokes_first_run_learned_runtime_preparation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    local_app_data = tmp_path / "AppData" / "Local"
    monkeypatch.setenv("LOCALAPPDATA", str(local_app_data))

    called: dict[str, object] = {
        "torch": False,
        "learned": False,
        "served": False,
    }

    def fake_prepare_cuda(data_dir: Path) -> None:
        called["torch"] = True

    def fake_prepare_learned(data_dir: Path) -> None:
        called["learned"] = True

    def fake_serve_review_ui(*, db_path: Path, host: str, port: int, open_browser: bool) -> None:
        called["served"] = True

    monkeypatch.setattr(desktop_module, "maybe_prepare_cuda_torch_runtime", fake_prepare_cuda)
    monkeypatch.setattr(desktop_module, "maybe_prepare_learned_iqa_runtime", fake_prepare_learned, raising=False)
    monkeypatch.setattr(desktop_module, "serve_review_ui", fake_serve_review_ui)
    monkeypatch.setattr(sys, "argv", ["shotsieve-desktop", "--no-browser"])

    desktop_module.main()

    assert called["torch"] is True
    assert called["learned"] is True
    assert called["served"] is True


def test_maybe_prepare_learned_iqa_runtime_reports_diagnostic_details_when_post_install_probe_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    monkeypatch.setenv("SHOTSIEVE_BOOTSTRAP_AUTO_INSTALL_LEARNED_IQA", "1")
    monkeypatch.setattr(desktop_module, "runtime_target_id_from_executable_name", lambda system_name=None: "windows-nvidia")

    probe_calls = {"count": 0}

    def fake_runtime_has_learned_iqa() -> bool:
        probe_calls["count"] += 1
        return False

    monkeypatch.setattr(desktop_module, "_runtime_has_learned_iqa", fake_runtime_has_learned_iqa)

    def fake_install_learned_iqa_sidecar(
        *, runtime: str, site_packages: Path, output_func=print, force_reinstall: bool = False
    ) -> bool:
        (site_packages / "pyiqa").mkdir(parents=True, exist_ok=True)
        (site_packages / "pyiqa" / "__init__.py").write_text("", encoding="utf-8")
        return True

    monkeypatch.setattr(desktop_module, "install_learned_iqa_sidecar", fake_install_learned_iqa_sidecar)
    monkeypatch.setattr(
        desktop_module,
        "_learned_iqa_runtime_import_diagnostic",
        lambda: "ModuleNotFoundError: No module named 'yaml' (install package: pyyaml)",
        raising=False,
    )

    messages: list[str] = []
    desktop_module.maybe_prepare_learned_iqa_runtime(data_dir, output_func=messages.append)

    assert probe_calls["count"] >= 2
    assert any("initialization still failed" in message.casefold() for message in messages)
    assert any("No module named 'yaml'" in message for message in messages)
    assert any("pip-install.log" in message for message in messages)


def test_maybe_prepare_learned_iqa_runtime_invalidates_hardware_cache_after_successful_install(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)

    monkeypatch.setenv("SHOTSIEVE_BOOTSTRAP_AUTO_INSTALL_LEARNED_IQA", "1")
    monkeypatch.setattr(desktop_module, "runtime_target_id_from_executable_name", lambda system_name=None: "windows-nvidia")

    probe_calls = {"count": 0}

    def fake_runtime_has_learned_iqa() -> bool:
        probe_calls["count"] += 1
        return probe_calls["count"] >= 2

    monkeypatch.setattr(desktop_module, "_runtime_has_learned_iqa", fake_runtime_has_learned_iqa)

    invalidations: list[str] = []
    monkeypatch.setattr(desktop_module, "invalidate_hw_cache", lambda: invalidations.append("learned"), raising=False)

    def fake_install_learned_iqa_sidecar(
        *, runtime: str, site_packages: Path, output_func=print, force_reinstall: bool = False
    ) -> bool:
        (site_packages / "pyiqa").mkdir(parents=True, exist_ok=True)
        (site_packages / "pyiqa" / "__init__.py").write_text("", encoding="utf-8")
        return True

    monkeypatch.setattr(desktop_module, "install_learned_iqa_sidecar", fake_install_learned_iqa_sidecar)

    desktop_module.maybe_prepare_learned_iqa_runtime(data_dir)

    assert invalidations == ["learned"]
