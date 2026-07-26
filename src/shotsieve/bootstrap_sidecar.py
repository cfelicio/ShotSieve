from __future__ import annotations

from collections.abc import Callable
import contextlib
import importlib
import io
import os
import pkgutil
import sys
import traceback
import warnings
from pathlib import Path
from typing import Any

from shotsieve import runtime_support


DEFAULT_TORCH_AUTO_INSTALL_ENV = "SHOTSIEVE_BOOTSTRAP_AUTO_INSTALL_TORCH"
DEFAULT_TORCH_SITE_PACKAGES_DIRNAME = "site-packages"
DISTUTILS_REPLACEMENT_WARNING_PATTERN = r"Setuptools is replacing distutils\..*"
PIP_UNEXPECTED_IMPORT_WARNING_PATTERN = r"DEPRECATION: Unexpected import of '.*' after pip install started\..*"


def _battr(name: str, fallback: Any) -> Any:
    mod = sys.modules.get("shotsieve.bootstrap")
    if mod is not None and hasattr(mod, name):
        return getattr(mod, name)
    return fallback


@contextlib.contextmanager
def _suppress_distutils_replacement_warning():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=DISTUTILS_REPLACEMENT_WARNING_PATTERN,
            category=UserWarning,
        )
        yield


@contextlib.contextmanager
def _suppress_embedded_pip_warnings():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=DISTUTILS_REPLACEMENT_WARNING_PATTERN,
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=PIP_UNEXPECTED_IMPORT_WARNING_PATTERN,
            category=Warning,
        )
        yield


def sidecar_site_packages_dir(runtime_root: Path, target_id: str) -> Path:
    return runtime_root / DEFAULT_TORCH_SITE_PACKAGES_DIRNAME / target_id


def _path_has_torch(path: Path) -> bool:
    return runtime_support.path_has_torch(path)


def _path_has_pyiqa(path: Path) -> bool:
    return runtime_support.path_has_pyiqa(path)


def runtime_bundle_contains_torch(install_dir: Path) -> bool:
    candidates = (
        install_dir,
        install_dir / "_internal",
        install_dir / "Lib" / "site-packages",
        install_dir / "lib" / "site-packages",
    )
    check_func = _battr("_path_has_torch", _path_has_torch)
    for candidate in candidates:
        if check_func(candidate):
            return True
    return False


def _parse_env_bool(value: str | None) -> bool | None:
    return runtime_support.parse_env_bool(value)


def _is_interactive_console() -> bool:
    return runtime_support.is_interactive_console()


def _confirm(prompt: str, *, input_func=input) -> bool:
    return runtime_support.confirm(prompt, input_func=input_func)


def _torch_install_index_args(runtime: str) -> list[str]:
    normalized = runtime.casefold()
    if normalized == "cuda":
        return [
            "--index-url",
            "https://download.pytorch.org/whl/cu126",
            "--trusted-host",
            "download.pytorch.org",
        ]
    if normalized == "cpu":
        return [
            "--index-url",
            "https://download.pytorch.org/whl/cpu",
            "--trusted-host",
            "download.pytorch.org",
        ]
    return []


def _patch_distlib_finder_for_frozen() -> None:
    if not getattr(sys, "frozen", False):
        return

    imp_mod = _battr("importlib", importlib)
    pkg_mod = _battr("pkgutil", pkgutil)
    suppress_func = _battr("_suppress_distutils_replacement_warning", _suppress_distutils_replacement_warning)

    try:
        with suppress_func():
            distlib_resources = imp_mod.import_module("pip._vendor.distlib.resources")
            distlib_package = imp_mod.import_module("pip._vendor.distlib")
    except Exception:
        return

    register_finder = getattr(distlib_resources, "register_finder", None)
    resource_finder = getattr(distlib_resources, "ResourceFinder", None)
    loader = getattr(distlib_package, "__loader__", None)

    if not callable(register_finder) or resource_finder is None:
        return

    finder_registry = getattr(distlib_resources, "_finder_registry", None)
    loader_types: set[type] = set()
    if loader is not None:
        loader_types.add(type(loader))

    get_loader = getattr(pkg_mod, "get_loader", None)
    if callable(get_loader):
        try:
            pkgutil_loader = get_loader("pip._vendor.distlib")
        except Exception:
            pkgutil_loader = None
    else:
        pkgutil_loader = None
    if pkgutil_loader is not None:
        loader_types.add(type(pkgutil_loader))

    try:
        pyi_importers = imp_mod.import_module("pyimod02_importers")
    except Exception:
        pyi_importers = None
    if pyi_importers is not None:
        for loader_name in ("PyiFrozenImporter", "FrozenImporter"):
            loader_type = getattr(pyi_importers, loader_name, None)
            if isinstance(loader_type, type):
                loader_types.add(loader_type)

    for loader_type in loader_types:
        if isinstance(finder_registry, dict) and loader_type in finder_registry:
            continue
        try:
            register_finder(loader_type, resource_finder)
        except Exception:
            continue

    original_finder = getattr(distlib_resources, "finder", None)
    distlib_exception = getattr(distlib_resources, "DistlibException", Exception)
    if getattr(original_finder, "__shotsieve_patched__", False):
        return

    if not callable(original_finder):
        return

    def _finder_with_fallback(package: str):
        try:
            return original_finder(package)
        except distlib_exception:
            return resource_finder(package)

    setattr(_finder_with_fallback, "__shotsieve_patched__", True)
    try:
        setattr(distlib_resources, "finder", _finder_with_fallback)
    except Exception:
        return


def _patch_pip_scriptmaker_for_embedded_install() -> None:
    imp_mod = _battr("importlib", importlib)
    suppress_func = _battr("_suppress_distutils_replacement_warning", _suppress_distutils_replacement_warning)
    try:
        with suppress_func():
            wheel_module = imp_mod.import_module("pip._internal.operations.install.wheel")
    except Exception:
        return

    script_maker = getattr(wheel_module, "PipScriptMaker", None)
    if not isinstance(script_maker, type):
        return

    if getattr(script_maker, "__shotsieve_disable_launchers_patch__", False):
        return

    original_init = getattr(script_maker, "__init__", None)
    if not callable(original_init):
        return

    def _patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        try:
            self.add_launchers = False
        except Exception:
            pass

    try:
        setattr(script_maker, "__init__", _patched_init)
        setattr(script_maker, "__shotsieve_disable_launchers_patch__", True)
    except Exception:
        return


def _load_embedded_pip_main() -> Callable[[list[str]], object] | None:
    imp_mod = _battr("importlib", importlib)
    suppress_func = _battr("_suppress_distutils_replacement_warning", _suppress_distutils_replacement_warning)
    try:
        with suppress_func():
            pip_module = imp_mod.import_module("pip._internal.cli.main")
    except Exception:
        return None

    pip_main = getattr(pip_module, "main", None)
    return pip_main if callable(pip_main) else None


def _coerce_pip_main_return_code(result: object) -> int:
    if result is None:
        return 0
    if isinstance(result, int):
        return result
    return 1


def _install_torch_sidecar_with_embedded_pip(
    *,
    runtime: str,
    site_packages: Path,
    force_reinstall: bool = False,
    output_func=print,
) -> bool | None:
    suppress_pip = _battr("_suppress_embedded_pip_warnings", _suppress_embedded_pip_warnings)
    with suppress_pip():
        p_distlib = _battr("_patch_distlib_finder_for_frozen", _patch_distlib_finder_for_frozen)
        p_script = _battr("_patch_pip_scriptmaker_for_embedded_install", _patch_pip_scriptmaker_for_embedded_install)
        p_main = _battr("_load_embedded_pip_main", _load_embedded_pip_main)
        p_distlib()
        p_script()
        pip_main = p_main()
    if pip_main is None:
        return None

    site_packages.mkdir(parents=True, exist_ok=True)
    pip_log_path = site_packages / "pip-install.log"
    try:
        pip_log_path.touch(exist_ok=True)
    except OSError:
        pass

    def _append_debug_log(
        *,
        package_name: str,
        install_args: list[str],
        return_code: int,
        stdout_text: str,
        stderr_text: str,
        exception_text: str | None,
    ) -> None:
        lines = [
            f"=== embedded pip install: {package_name} ===",
            f"args: {' '.join(install_args)}",
            f"exit_code: {return_code}",
        ]
        if stdout_text:
            lines.extend(["--- stdout ---", stdout_text.rstrip("\n")])
        if stderr_text:
            lines.extend(["--- stderr ---", stderr_text.rstrip("\n")])
        if exception_text:
            lines.extend(["--- exception ---", exception_text.rstrip("\n")])
        lines.append("")

        try:
            with pip_log_path.open("a", encoding="utf-8", errors="replace") as log_file:
                log_file.write("\n".join(lines))
        except OSError:
            pass

    def _run_pip_install(package_name: str) -> int:
        idx_args_func = _battr("_torch_install_index_args", _torch_install_index_args)
        install_args = [
            "install",
            "--disable-pip-version-check",
            "--upgrade",
            "--no-cache-dir",
            "--no-deps",
            "--log",
            str(pip_log_path),
            "--target",
            str(site_packages),
            package_name,
            *idx_args_func(runtime),
        ]
        if force_reinstall:
            install_args.insert(1, "--force-reinstall")

        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        exception_text: str | None = None

        coerce_func = _battr("_coerce_pip_main_return_code", _coerce_pip_main_return_code)
        try:
            with suppress_pip():
                with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
                    return_code = coerce_func(pip_main(install_args))
        except SystemExit as exc:
            code = exc.code
            if isinstance(code, int):
                return_code = code
            else:
                return_code = 0 if code in {None, ""} else 1
        except Exception:
            return_code = 1
            exception_text = traceback.format_exc()

        _append_debug_log(
            package_name=package_name,
            install_args=install_args,
            return_code=return_code,
            stdout_text=stdout_buffer.getvalue(),
            stderr_text=stderr_buffer.getvalue(),
            exception_text=exception_text,
        )
        return return_code

    has_torch_func = _battr("_path_has_torch", _path_has_torch)
    torch_return_code = _run_pip_install("torch")
    if torch_return_code != 0:
        if has_torch_func(site_packages):
            output_func(
                "PyTorch installation completed with a non-fatal cleanup issue. "
                "Continuing with the detected runtime package."
            )
            return True

        output_func(
            "PyTorch installation step 'torch' failed with exit code "
            f"{torch_return_code}. Check {pip_log_path} for details."
        )
        output_func("PyTorch installation failed. The app will continue without GPU-accelerated learned models.")
        return False

    torchvision_return_code = _run_pip_install("torchvision")
    if torchvision_return_code != 0:
        output_func(
            "Torchvision installation failed with exit code "
            f"{torchvision_return_code}, but PyTorch was installed. "
            f"Check {pip_log_path} for details. Continuing with available learned-model support."
        )

    return has_torch_func(site_packages)


def install_torch_sidecar(
    *,
    runtime: str,
    site_packages: Path,
    output_func=print,
    force_reinstall: bool = False,
) -> bool:
    sidecar_func = _battr("_install_torch_sidecar_with_embedded_pip", _install_torch_sidecar_with_embedded_pip)
    embedded_install_result = sidecar_func(
        runtime=runtime,
        site_packages=site_packages,
        force_reinstall=force_reinstall,
        output_func=output_func,
    )
    if embedded_install_result is None:
        output_func(
            "Bundled pip runtime installer is unavailable in this build. "
            "The app will continue without GPU-accelerated learned models."
        )
        return False

    return embedded_install_result


def _learned_iqa_packages_for_runtime(runtime: str) -> list[str]:
    normalized = runtime.casefold()
    packages = [
        "pyiqa",
        "opencv-python-headless",
        "pyyaml",
        "sympy",
        "requests",
        "tqdm",
        "scipy",
        "huggingface-hub",
        "pandas",
        "icecream",
    ]
    if normalized == "directml":
        packages.append("torch-directml")
    return packages


def _install_learned_iqa_sidecar_with_embedded_pip(
    *,
    runtime: str,
    site_packages: Path,
    force_reinstall: bool = False,
    output_func=print,
) -> bool | None:
    suppress_pip = _battr("_suppress_embedded_pip_warnings", _suppress_embedded_pip_warnings)
    with suppress_pip():
        p_distlib = _battr("_patch_distlib_finder_for_frozen", _patch_distlib_finder_for_frozen)
        p_script = _battr("_patch_pip_scriptmaker_for_embedded_install", _patch_pip_scriptmaker_for_embedded_install)
        p_main = _battr("_load_embedded_pip_main", _load_embedded_pip_main)
        p_distlib()
        p_script()
        pip_main = p_main()
    if pip_main is None:
        return None

    site_packages.mkdir(parents=True, exist_ok=True)
    pip_log_path = site_packages / "pip-install.log"
    try:
        pip_log_path.touch(exist_ok=True)
    except OSError:
        pass

    def _append_debug_log(
        *,
        package_name: str,
        install_args: list[str],
        return_code: int,
        stdout_text: str,
        stderr_text: str,
        exception_text: str | None,
    ) -> None:
        lines = [
            f"=== embedded pip install: {package_name} ===",
            f"args: {' '.join(install_args)}",
            f"exit_code: {return_code}",
        ]
        if stdout_text:
            lines.extend(["--- stdout ---", stdout_text.rstrip("\n")])
        if stderr_text:
            lines.extend(["--- stderr ---", stderr_text.rstrip("\n")])
        if exception_text:
            lines.extend(["--- exception ---", exception_text.rstrip("\n")])
        lines.append("")

        try:
            with pip_log_path.open("a", encoding="utf-8", errors="replace") as log_file:
                log_file.write("\n".join(lines))
        except OSError:
            pass

    def _run_pip_install(package_name: str) -> int:
        install_args = [
            "install",
            "--disable-pip-version-check",
            "--upgrade",
            "--no-cache-dir",
        ]
        if package_name == "pyiqa":
            install_args.append("--no-deps")

        install_args.extend(
            [
                "--log",
                str(pip_log_path),
                "--target",
                str(site_packages),
                package_name,
            ]
        )
        if force_reinstall:
            install_args.insert(1, "--force-reinstall")

        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        exception_text: str | None = None

        coerce_func = _battr("_coerce_pip_main_return_code", _coerce_pip_main_return_code)
        try:
            with suppress_pip():
                with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
                    return_code = coerce_func(pip_main(install_args))
        except SystemExit as exc:
            code = exc.code
            if isinstance(code, int):
                return_code = code
            else:
                return_code = 0 if code in {None, ""} else 1
        except Exception:
            return_code = 1
            exception_text = traceback.format_exc()

        _append_debug_log(
            package_name=package_name,
            install_args=install_args,
            return_code=return_code,
            stdout_text=stdout_buffer.getvalue(),
            stderr_text=stderr_buffer.getvalue(),
            exception_text=exception_text,
        )
        return return_code

    package_results: dict[str, int] = {}
    pkg_list_func = _battr("_learned_iqa_packages_for_runtime", _learned_iqa_packages_for_runtime)
    for package_name in pkg_list_func(runtime):
        package_results[package_name] = _run_pip_install(package_name)

    has_pyiqa_func = _battr("_path_has_pyiqa", _path_has_pyiqa)
    pyiqa_return_code = package_results.get("pyiqa", 1)
    if pyiqa_return_code != 0 and not has_pyiqa_func(site_packages):
        output_func(
            "Learned IQA installation step 'pyiqa' failed with exit code "
            f"{pyiqa_return_code}. Check {pip_log_path} for details."
        )
        output_func("Learned IQA dependency installation failed. The app will continue with learned backends disabled.")
        return False

    for package_name, return_code in package_results.items():
        if package_name == "pyiqa" or return_code == 0:
            continue
        output_func(
            f"Dependency '{package_name}' installation failed with exit code {return_code}. "
            f"Check {pip_log_path} for details. Continuing with available learned-model support."
        )

    return has_pyiqa_func(site_packages)


def install_learned_iqa_sidecar(
    *,
    runtime: str,
    site_packages: Path,
    output_func=print,
    force_reinstall: bool = False,
) -> bool:
    sidecar_func = _battr("_install_learned_iqa_sidecar_with_embedded_pip", _install_learned_iqa_sidecar_with_embedded_pip)
    embedded_install_result = sidecar_func(
        runtime=runtime,
        site_packages=site_packages,
        force_reinstall=force_reinstall,
        output_func=output_func,
    )
    if embedded_install_result is None:
        output_func(
            "Bundled pip runtime installer is unavailable in this build. "
            "The app will continue with learned backends disabled."
        )
        return False

    return embedded_install_result


def _compose_pythonpath(*, existing: str | None, prepend_path: Path) -> str:
    return runtime_support.compose_pythonpath(existing=existing, prepend_path=prepend_path)


def maybe_prepare_torch_runtime(
    asset,
    *,
    install_dir: Path,
    runtime_root: Path,
    input_func=input,
    output_func=print,
) -> dict[str, str]:
    contains_func = _battr("runtime_bundle_contains_torch", runtime_bundle_contains_torch)
    if contains_func(install_dir):
        return {}

    sidecar_dir_func = _battr("sidecar_site_packages_dir", sidecar_site_packages_dir)
    has_torch_func = _battr("_path_has_torch", _path_has_torch)
    site_packages = sidecar_dir_func(runtime_root, asset.id)
    if has_torch_func(site_packages):
        compose_func = _battr("_compose_pythonpath", _compose_pythonpath)
        return {"PYTHONPATH": compose_func(existing=os.environ.get("PYTHONPATH"), prepend_path=site_packages)}

    parse_env_func = _battr("_parse_env_bool", _parse_env_bool)
    auto_install = parse_env_func(os.environ.get(DEFAULT_TORCH_AUTO_INSTALL_ENV))
    if auto_install is None:
        is_console_func = _battr("_is_interactive_console", _is_interactive_console)
        if not is_console_func():
            return {}
        confirm_func = _battr("_confirm", _confirm)
        auto_install = confirm_func(
            "PyTorch was not detected for this runtime. Download and install it now? [y/N]: ",
            input_func=input_func,
        )

    if not auto_install:
        output_func("Continuing without runtime PyTorch installation.")
        return {}

    output_func("Installing PyTorch runtime dependencies. This may take a few minutes...")
    install_func = _battr("install_torch_sidecar", install_torch_sidecar)
    installed = install_func(runtime=asset.runtime, site_packages=site_packages)
    if not installed:
        return {}

    compose_func = _battr("_compose_pythonpath", _compose_pythonpath)
    return {"PYTHONPATH": compose_func(existing=os.environ.get("PYTHONPATH"), prepend_path=site_packages)}
