from __future__ import annotations

import argparse
import importlib
import importlib.util
import inspect
import os
import platform
import sys
import traceback
from pathlib import Path

from shotsieve import runtime_support
from shotsieve.bootstrap import install_learned_iqa_sidecar, install_torch_sidecar, sidecar_site_packages_dir
from shotsieve.learned_iqa import invalidate_hw_cache
from shotsieve.web import serve_review_ui


APP_DIRNAME = "ShotSieve"
PORTABLE_DATA_DIRNAME = "data"
TORCH_AUTO_INSTALL_ENV = "SHOTSIEVE_BOOTSTRAP_AUTO_INSTALL_TORCH"
LEARNED_IQA_AUTO_INSTALL_ENV = "SHOTSIEVE_BOOTSTRAP_AUTO_INSTALL_LEARNED_IQA"
LEARNED_IQA_MISSING_MODULE_PACKAGE_HINTS = {
    "yaml": "pyyaml",
    "cv2": "opencv-python-headless",
    "pil": "Pillow",
    "huggingface_hub": "huggingface-hub",
    "sympy": "sympy",
}


def default_data_dir() -> Path:
    if getattr(sys, "frozen", False):
        executable_dir = Path(sys.executable).resolve().parent
        return (executable_dir / PORTABLE_DATA_DIRNAME).resolve()

    source_checkout_root = runtime_support.source_checkout_root(__file__, package_name="shotsieve")
    if source_checkout_root is not None:
        return (source_checkout_root / PORTABLE_DATA_DIRNAME).resolve()

    local_app_data = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
    if local_app_data:
        return Path(local_app_data).expanduser().resolve() / APP_DIRNAME
    return (Path.home() / f".{APP_DIRNAME.casefold()}").resolve()


def runtime_target_id_from_executable_name(*, system_name: str | None = None) -> str | None:
    runtime_name_source = sys.executable if getattr(sys, "frozen", False) else sys.argv[0]
    runtime_name = Path(runtime_name_source).name.casefold()
    system = (system_name or platform.system()).casefold()

    if system == "windows":
        prefix = "windows"
    elif system == "linux":
        prefix = "linux"
    elif system == "darwin":
        prefix = "macos"
    else:
        return None

    if "nvidia" in runtime_name or "cuda" in runtime_name:
        return f"{prefix}-nvidia"
    if prefix == "windows" and ("directml" in runtime_name or "-dml" in runtime_name):
        return "windows-dml"
    if prefix == "macos" and "mps" in runtime_name:
        return "macos-mps"
    if "cpu" in runtime_name:
        return f"{prefix}-cpu"

    return None


def target_is_cuda_runtime(target_id: str | None) -> bool:
    return bool(target_id and target_id.endswith("-nvidia"))


def _resolve_cuda_runtime_target_id(target_id: str | None) -> str | None:
    return target_id if target_is_cuda_runtime(target_id) else None


def _clear_torch_module_cache() -> None:
    for module_name in list(sys.modules):
        if (
            module_name == "torch"
            or module_name.startswith("torch.")
            or module_name == "torchvision"
            or module_name.startswith("torchvision.")
            or module_name == "torchaudio"
            or module_name.startswith("torchaudio.")
            or module_name == "functorch"
            or module_name.startswith("functorch.")
        ):
            sys.modules.pop(module_name, None)


def _clear_pyiqa_module_cache() -> None:
    for module_name in list(sys.modules):
        if module_name == "pyiqa" or module_name.startswith("pyiqa."):
            sys.modules.pop(module_name, None)


def _learned_iqa_runtime_import_diagnostic() -> str | None:
    _clear_pyiqa_module_cache()

    try:
        importlib.invalidate_caches()
    except Exception:
        pass

    try:
        spec = importlib.util.find_spec("pyiqa")
    except Exception as exc:
        return f"find_spec('pyiqa') failed: {type(exc).__name__}: {exc}"

    if spec is None:
        return "pyiqa module was not discoverable on sys.path after installation."

    try:
        importlib.import_module("pyiqa")
    except Exception as exc:
        details: list[str] = [f"{type(exc).__name__}: {exc}"]

        missing_module_name = getattr(exc, "name", None)
        if isinstance(missing_module_name, str) and missing_module_name:
            module_root = missing_module_name.split(".", 1)[0].casefold()
            suggested_package = LEARNED_IQA_MISSING_MODULE_PACKAGE_HINTS.get(module_root)
            if suggested_package:
                details.append(
                    f"missing module '{missing_module_name}' (suggested package: {suggested_package})"
                )
            else:
                details.append(f"missing module '{missing_module_name}'")

        trace_text = traceback.format_exc(limit=6).strip()
        if trace_text:
            details.append(trace_text)
        return " | ".join(details)

    return None


def runtime_bundle_has_usable_cuda_torch(*, force_reload: bool = False) -> bool:
    if importlib.util.find_spec("torch") is None:
        return False

    if force_reload:
        _clear_torch_module_cache()

    try:
        torch_module = importlib.import_module("torch")
    except Exception:
        return False

    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not hasattr(cuda, "is_available"):
        return False

    try:
        return bool(cuda.is_available())
    except Exception:
        return False


def _path_has_torch(path: Path) -> bool:
    return runtime_support.path_has_torch(path)


def _path_has_pyiqa(path: Path) -> bool:
    return runtime_support.path_has_pyiqa(path)


def _parse_env_bool(value: str | None) -> bool | None:
    return runtime_support.parse_env_bool(value)


def _is_interactive_console() -> bool:
    return runtime_support.is_interactive_console()


def _confirm(prompt: str, *, input_func=input) -> bool:
    return runtime_support.confirm(prompt, input_func=input_func)


def _compose_pythonpath(*, existing: str | None, prepend_path: Path) -> str:
    return runtime_support.compose_pythonpath(existing=existing, prepend_path=prepend_path)


def _call_prepare_learned_iqa_runtime(data_dir: Path, *, assume_install_consent: bool) -> None:
    prepare_fn = maybe_prepare_learned_iqa_runtime
    try:
        parameters = inspect.signature(prepare_fn).parameters
    except (TypeError, ValueError):
        parameters = {}

    supports_assume_install_consent = "assume_install_consent" in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )

    if supports_assume_install_consent:
        prepare_fn(data_dir, assume_install_consent=assume_install_consent)
        return

    prepare_fn(data_dir)


def _prepend_runtime_pythonpath(path: Path) -> None:
    path_text = str(path)
    os.environ["PYTHONPATH"] = _compose_pythonpath(existing=os.environ.get("PYTHONPATH"), prepend_path=path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)


def _sidecar_torch_has_usable_cuda(site_packages: Path) -> bool:
    _ = site_packages
    return runtime_bundle_has_usable_cuda_torch(force_reload=True)


def _runtime_has_learned_iqa() -> bool:
    return _learned_iqa_runtime_import_diagnostic() is None


def _runtime_name_from_target_id(target_id: str | None) -> str:
    normalized = (target_id or "").strip().casefold()
    if normalized.endswith("-nvidia"):
        return "cuda"
    if normalized.endswith("-dml"):
        return "directml"
    if normalized.endswith("-cpu"):
        return "cpu"
    if normalized.endswith("-mps"):
        return "default"
    return "default"


def _fallback_runtime_target_id(*, system_name: str | None = None) -> str:
    system = (system_name or platform.system()).casefold()
    if system == "windows":
        return "windows-cpu"
    if system == "linux":
        return "linux-cpu"
    if system == "darwin":
        return "macos-cpu"
    return "windows-cpu"


def maybe_prepare_learned_iqa_runtime(
    data_dir: Path,
    *,
    target_id: str | None = None,
    assume_install_consent: bool = False,
    input_func=input,
    output_func=print,
) -> None:
    if _runtime_has_learned_iqa():
        return

    resolved_target_id = target_id or runtime_target_id_from_executable_name() or _fallback_runtime_target_id()
    runtime_root = (data_dir / "runtime").resolve()
    site_packages = sidecar_site_packages_dir(runtime_root, resolved_target_id)

    sidecar_has_pyiqa = _path_has_pyiqa(site_packages)
    if sidecar_has_pyiqa:
        _prepend_runtime_pythonpath(site_packages)
        if _runtime_has_learned_iqa():
            return

    auto_install = _parse_env_bool(os.environ.get(LEARNED_IQA_AUTO_INSTALL_ENV))
    if auto_install is None:
        if assume_install_consent:
            output_func(
                "PyTorch runtime was installed for this session. Continuing with learned IQA runtime installation..."
            )
            auto_install = True
        elif not _is_interactive_console():
            if sidecar_has_pyiqa:
                output_func(
                    "Learned IQA sidecar exists but is unavailable in this session; attempting repair installation..."
                )
            else:
                output_func("Learned IQA dependencies were not detected. Attempting automatic runtime installation...")
            auto_install = True
        else:
            prompt = (
                "Learned IQA sidecar is present but unavailable. Repair installation now? [y/N]: "
                if sidecar_has_pyiqa
                else "Learned IQA dependencies were not detected. Download and install now? [y/N]: "
            )
            auto_install = _confirm(prompt, input_func=input_func)

    if not auto_install:
        if sidecar_has_pyiqa:
            output_func("Learned IQA sidecar is present but unavailable in this session. Skipping reinstall.")
            diagnostic = _learned_iqa_runtime_import_diagnostic()
            if diagnostic:
                output_func(f"Learned IQA runtime diagnostic: {diagnostic}")
                output_func(f"Learned IQA pip log: {site_packages / 'pip-install.log'}")
        else:
            output_func("Continuing without learned IQA dependency installation.")
        return

    output_func("Installing learned IQA runtime dependencies. This may take a few minutes...")
    installed = install_learned_iqa_sidecar(
        runtime=_runtime_name_from_target_id(resolved_target_id),
        site_packages=site_packages,
        output_func=output_func,
        force_reinstall=sidecar_has_pyiqa,
    )
    if installed and _path_has_pyiqa(site_packages):
        _prepend_runtime_pythonpath(site_packages)
        invalidate_hw_cache()
        if not _runtime_has_learned_iqa():
            output_func(
                "Learned IQA dependencies were installed, but initialization still failed in this session. "
                "The app will continue with learned backends disabled."
            )
            diagnostic = _learned_iqa_runtime_import_diagnostic()
            if diagnostic:
                output_func(f"Learned IQA runtime diagnostic: {diagnostic}")
            output_func(f"Learned IQA sidecar path: {site_packages}")
            output_func(f"Learned IQA pip log: {site_packages / 'pip-install.log'}")


def maybe_prepare_cuda_torch_runtime(data_dir: Path, *, target_id: str | None = None, input_func=input, output_func=print) -> bool:
    resolved_target_id = _resolve_cuda_runtime_target_id(target_id or runtime_target_id_from_executable_name())
    if resolved_target_id is None:
        return False

    if runtime_bundle_has_usable_cuda_torch():
        return False

    runtime_root = (data_dir / "runtime").resolve()
    site_packages = sidecar_site_packages_dir(runtime_root, resolved_target_id)
    sidecar_has_torch = _path_has_torch(site_packages)
    if sidecar_has_torch:
        _prepend_runtime_pythonpath(site_packages)
        if _sidecar_torch_has_usable_cuda(site_packages):
            return False

    auto_install = _parse_env_bool(os.environ.get(TORCH_AUTO_INSTALL_ENV))
    if auto_install is None:
        if not _is_interactive_console():
            if sidecar_has_torch:
                output_func(
                    "Runtime PyTorch is already installed but CUDA is unavailable in this session; "
                    "attempting repair installation..."
                )
            else:
                output_func("PyTorch was not detected for this CUDA runtime. Attempting automatic runtime installation...")
            auto_install = True
        else:
            prompt = (
                "Runtime PyTorch was found but CUDA is unavailable. Repair runtime installation now? [y/N]: "
                if sidecar_has_torch
                else "PyTorch was not detected for this CUDA runtime. Download and install it now? [y/N]: "
            )
            auto_install = _confirm(prompt, input_func=input_func)

    if not auto_install:
        if sidecar_has_torch:
            output_func("Runtime PyTorch is already installed but CUDA is unavailable in this session. Skipping reinstall.")
        else:
            output_func("Continuing without runtime PyTorch installation.")
        return False

    output_func("Installing PyTorch runtime dependencies. This may take a few minutes...")
    installed = install_torch_sidecar(
        runtime="cuda",
        site_packages=site_packages,
        output_func=output_func,
        force_reinstall=sidecar_has_torch,
    )
    if installed and _path_has_torch(site_packages):
        _prepend_runtime_pythonpath(site_packages)
        invalidate_hw_cache()
        if not _sidecar_torch_has_usable_cuda(site_packages):
            output_func(
                "PyTorch runtime is installed, but CUDA remains unavailable. "
                "The app will continue without GPU-accelerated learned models."
            )
            return False
        return True

    return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="shotsieve-desktop")
    parser.add_argument("--data-dir", default=None, help="Directory for the local ShotSieve cache and previews")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface for the local review server")
    parser.add_argument("--port", type=int, default=8765, help="Port for the local review server")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open the default browser")
    return parser


def main() -> None:
    import multiprocessing
    multiprocessing.freeze_support()

    parser = build_parser()
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve() if args.data_dir else default_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    installed_torch_runtime = maybe_prepare_cuda_torch_runtime(data_dir)
    _call_prepare_learned_iqa_runtime(data_dir, assume_install_consent=installed_torch_runtime)
    db_path = data_dir / "shotsieve.db"

    serve_review_ui(
        db_path=db_path,
        host=args.host,
        port=args.port,
        open_browser=not args.no_browser,
    )


if __name__ == "__main__":
    main()