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

from shotsieve.bootstrap_sidecar import (
    dispatch_sidecar_install_command,
    install_learned_iqa_sidecar,
    install_torch_sidecar,
    sidecar_site_packages_dir,
    torch_sidecar_is_valid,
)
from shotsieve.learned_iqa import invalidate_hw_cache
from shotsieve.learned_iqa_runtime import cuda_runtime_is_usable, has_mps, has_rocm, has_xpu
from shotsieve.model_assets import apply_model_cache_dir, recover_orphaned_preparation
from shotsieve.release_targets import canonical_release_target_id
from shotsieve.runtime_support import (
    compose_pythonpath,
    confirm,
    is_interactive_console,
    parse_env_bool,
    path_has_pyiqa,
    path_has_torch,
    prepare_runtime_dll_search_path,
    source_checkout_root,
)
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
    "facexlib": "facexlib",
    "tqdm": "tqdm",
}


def default_data_dir() -> Path:
    if getattr(sys, "frozen", False):
        executable_dir = Path(sys.executable).resolve().parent
        return (executable_dir / PORTABLE_DATA_DIRNAME).resolve()

    source_root = source_checkout_root(__file__, package_name="shotsieve")
    if source_root is not None:
        return (source_root / PORTABLE_DATA_DIRNAME).resolve()

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

    if "nvidia-cuda" in runtime_name:
        return f"{prefix}-nvidia-cuda"
    if "intel-xpu" in runtime_name:
        return f"{prefix}-intel-xpu"
    if "amd-rocm" in runtime_name:
        return f"{prefix}-amd-rocm"
    if prefix == "macos" and "apple-mps" in runtime_name:
        return "macos-apple-mps"
    if "cpu" in runtime_name:
        return f"{prefix}-cpu"

    return None


def _clear_failed_torch_imports() -> None:
    """Remove orphan imports after a failure, never reload an initialized Torch.

    Dropping a successfully imported torch does not unload its native extension
    or operator registrations. Reimporting it can break CPU fallback as well as
    GPU initialization. Replacing an already-loaded build requires a restart.
    """
    if sys.modules.get("torch") is not None:
        return
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


def _clear_transformers_module_cache() -> None:
    for module_name in list(sys.modules):
        if module_name == "transformers" or module_name.startswith("transformers."):
            sys.modules.pop(module_name, None)


def _clear_tqdm_module_cache() -> None:
    """Drop a host/frozen tqdm module before loading the sidecar runtime."""
    for module_name in list(sys.modules):
        if module_name == "tqdm" or module_name.startswith("tqdm."):
            sys.modules.pop(module_name, None)


def _learned_iqa_runtime_import_diagnostic() -> str | None:
    _clear_pyiqa_module_cache()
    _clear_transformers_module_cache()
    _clear_tqdm_module_cache()

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

    try:
        transformers_module = importlib.import_module("transformers")
        # Q-ReAlign imports this lazy Transformers class only when its model is
        # constructed. Probe it here so a stale or internally inconsistent
        # sidecar is repaired before the user reaches model preparation.
        getattr(transformers_module, "AutoModelForImageTextToText")
    except Exception as exc:
        details = [f"{type(exc).__name__}: {exc}"]
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

    try:
        # TOPIQ's architecture module imports FaceRestoreHelper even for the
        # non-face topiq_nr model.  Probe registration here so an older
        # sidecar missing facexlib is repaired before model preparation.
        importlib.import_module("pyiqa.archs.topiq_arch")
    except Exception as exc:
        details = [f"{type(exc).__name__}: {exc}"]
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
    if force_reload:
        importlib.invalidate_caches()
        _clear_failed_torch_imports()

    if importlib.util.find_spec("torch") is None:
        return False

    try:
        torch_module = importlib.import_module("torch")
    except Exception:
        return False

    return cuda_runtime_is_usable(torch_module)


def runtime_bundle_has_usable_torch(target_id: str | None, *, force_reload: bool = False) -> bool:
    """Probe the already-importable runtime for the selected target family."""
    normalized_target = canonical_release_target_id(target_id or "")
    if normalized_target.endswith("-nvidia-cuda"):
        return runtime_bundle_has_usable_cuda_torch(force_reload=force_reload)

    if force_reload:
        importlib.invalidate_caches()
        _clear_failed_torch_imports()

    if importlib.util.find_spec("torch") is None:
        return False

    try:
        torch_module = importlib.import_module("torch")
    except Exception:
        return False

    runtime = _runtime_name_from_target_id(normalized_target)
    if runtime == "xpu":
        return has_xpu(torch_module)
    if runtime == "rocm":
        return has_rocm(torch_module)
    if runtime == "mps":
        return has_mps(torch_module)
    return True


def _call_prepare_learned_iqa_runtime(
    data_dir: Path,
    *,
    assume_install_consent: bool,
    torch_available: bool | None = None,
) -> None:
    prepare_fn = maybe_prepare_learned_iqa_runtime
    try:
        parameters = inspect.signature(prepare_fn).parameters
    except (TypeError, ValueError):
        parameters = {}

    supports_assume_install_consent = "assume_install_consent" in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )
    supports_torch_available = "torch_available" in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )

    kwargs: dict[str, object] = {}
    if supports_assume_install_consent:
        kwargs["assume_install_consent"] = assume_install_consent
    if supports_torch_available and torch_available is not None:
        kwargs["torch_available"] = torch_available
    if kwargs:
        prepare_fn(data_dir, **kwargs)
        return

    prepare_fn(data_dir)


def _prepend_runtime_pythonpath(path: Path) -> None:
    path_text = str(path)
    os.environ["PYTHONPATH"] = compose_pythonpath(existing=os.environ.get("PYTHONPATH"), prepend_path=path)
    prepare_runtime_dll_search_path(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)


def _sidecar_torch_has_usable_cuda(site_packages: Path) -> bool:
    _ = site_packages
    return runtime_bundle_has_usable_cuda_torch(force_reload=True)


def _sidecar_torch_has_usable_runtime(site_packages: Path, target_id: str) -> bool:
    _prepend_runtime_pythonpath(site_packages)
    return runtime_bundle_has_usable_torch(target_id, force_reload=True)


def _target_torch_package_is_available(data_dir: Path, target_id: str) -> bool:
    """Return package availability separately from accelerator availability."""
    if runtime_bundle_has_usable_torch(target_id):
        return True
    runtime_root = (data_dir / "runtime").resolve()
    runtime_name = _runtime_name_from_target_id(target_id)
    site_packages = sidecar_site_packages_dir(runtime_root, target_id)
    return torch_sidecar_is_valid(site_packages, target_id=target_id, runtime=runtime_name) and path_has_torch(site_packages)


def _runtime_has_learned_iqa() -> bool:
    return _learned_iqa_runtime_import_diagnostic() is None


def _runtime_name_from_target_id(target_id: str | None) -> str:
    normalized = (target_id or "").strip().casefold()
    if normalized.endswith("-nvidia-cuda"):
        return "cuda"
    if normalized.endswith("-intel-xpu"):
        return "xpu"
    if normalized.endswith("-amd-rocm"):
        return "rocm"
    if normalized.endswith("-cpu"):
        return "cpu"
    if normalized.endswith("-apple-mps"):
        return "mps"
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
    torch_available: bool | None = None,
    assume_install_consent: bool = False,
    force_install: bool = False,
    input_func=input,
    output_func=print,
) -> bool:
    if torch_available is False:
        output_func(
            "Learned IQA remains unavailable because the target PyTorch sidecar is not installed. "
            "The catalog and Review UI will continue to work."
        )
        return False

    if _runtime_has_learned_iqa():
        return True

    resolved_target_id = canonical_release_target_id(
        target_id or runtime_target_id_from_executable_name() or _fallback_runtime_target_id()
    )
    runtime_root = (data_dir / "runtime").resolve()
    site_packages = sidecar_site_packages_dir(runtime_root, resolved_target_id)

    sidecar_has_pyiqa = path_has_pyiqa(site_packages)
    if sidecar_has_pyiqa:
        _prepend_runtime_pythonpath(site_packages)
        if _runtime_has_learned_iqa():
            return True

    configured_install = parse_env_bool(os.environ.get(LEARNED_IQA_AUTO_INSTALL_ENV))
    auto_install = True if force_install else configured_install
    if auto_install is None:
        if assume_install_consent:
            output_func(
                "PyTorch runtime was installed for this session. Continuing with learned IQA runtime installation..."
            )
            auto_install = True
        elif not is_interactive_console():
            if sidecar_has_pyiqa:
                output_func(
                    "Learned IQA sidecar exists but is unavailable in this session; skipping automatic repair. "
                    "Set "
                    f"{LEARNED_IQA_AUTO_INSTALL_ENV}=1."
                )
            else:
                output_func(
                    "Learned IQA dependencies were not detected. Skipping automatic installation. "
                    "Set "
                    f"{LEARNED_IQA_AUTO_INSTALL_ENV}=1."
                )
            auto_install = False
        else:
            prompt = (
                "Learned IQA sidecar is present but unavailable. Repair installation now? [y/N]: "
                if sidecar_has_pyiqa
                else "Learned IQA dependencies were not detected. Download and install now? [y/N]: "
            )
            auto_install = confirm(prompt, input_func=input_func)

    if not auto_install:
        if sidecar_has_pyiqa:
            output_func("Learned IQA sidecar is present but unavailable in this session. Skipping reinstall.")
            diagnostic = _learned_iqa_runtime_import_diagnostic()
            if diagnostic:
                output_func(f"Learned IQA runtime diagnostic: {diagnostic}")
                output_func(f"Learned IQA pip log: {site_packages / 'pip-install.log'}")
        else:
            output_func("Continuing without learned IQA dependency installation.")
        return False

    output_func("Installing learned IQA runtime dependencies. This may take a few minutes...")
    installed = install_learned_iqa_sidecar(
        runtime=_runtime_name_from_target_id(resolved_target_id),
        site_packages=site_packages,
        output_func=output_func,
        force_reinstall=sidecar_has_pyiqa,
    )
    if installed and path_has_pyiqa(site_packages):
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
            return False

        return True

    return False


def maybe_prepare_torch_runtime(
    data_dir: Path,
    *,
    target_id: str | None = None,
    force_install: bool = False,
    input_func=input,
    output_func=print,
) -> bool:
    """Prepare the target selected by the frozen launcher name."""
    requested_target_id = target_id or runtime_target_id_from_executable_name() or _fallback_runtime_target_id()
    resolved_target_id = canonical_release_target_id(requested_target_id)
    runtime = _runtime_name_from_target_id(resolved_target_id)

    if runtime_bundle_has_usable_torch(resolved_target_id):
        return False

    runtime_root = (data_dir / "runtime").resolve()
    site_packages = sidecar_site_packages_dir(runtime_root, resolved_target_id)
    sidecar_has_torch = torch_sidecar_is_valid(
        site_packages,
        target_id=resolved_target_id,
        runtime=runtime,
    ) and path_has_torch(site_packages)
    if sidecar_has_torch:
        if runtime == "cuda":
            _prepend_runtime_pythonpath(site_packages)
            sidecar_usable = _sidecar_torch_has_usable_cuda(site_packages)
        else:
            sidecar_usable = _sidecar_torch_has_usable_runtime(site_packages, resolved_target_id)
        if sidecar_usable:
            return False

    configured_install = parse_env_bool(os.environ.get(TORCH_AUTO_INSTALL_ENV))
    auto_install = True if force_install else configured_install
    if auto_install is None:
        if not is_interactive_console():
            if sidecar_has_torch:
                output_func(
                    f"Runtime PyTorch is already installed but {runtime} is unavailable in this session; "
                    "skipping automatic repair. Set "
                    f"{TORCH_AUTO_INSTALL_ENV}=1."
                )
            else:
                output_func(
                    f"PyTorch was not detected for this {runtime} runtime. Skipping automatic installation. "
                    "Set "
                    f"{TORCH_AUTO_INSTALL_ENV}=1."
                )
            auto_install = False
        else:
            prompt = (
                f"Runtime PyTorch was found but {runtime} is unavailable. Repair runtime installation now? [y/N]: "
                if sidecar_has_torch
                else f"PyTorch was not detected for this {runtime} runtime. Download and install it now? [y/N]: "
            )
            auto_install = confirm(prompt, input_func=input_func)

    if not auto_install:
        if sidecar_has_torch:
            output_func(
                f"Runtime PyTorch is already installed but {runtime} is unavailable in this session. "
                "Skipping reinstall."
            )
        else:
            output_func("Continuing without runtime PyTorch installation.")
        return False

    output_func("Installing PyTorch runtime dependencies. This may take a few minutes...")
    installed = install_torch_sidecar(
        runtime=runtime,
        site_packages=site_packages,
        output_func=output_func,
        force_reinstall=sidecar_has_torch,
    )
    if installed and path_has_torch(site_packages):
        _prepend_runtime_pythonpath(site_packages)
        invalidate_hw_cache()
        if runtime == "cuda":
            sidecar_usable = _sidecar_torch_has_usable_cuda(site_packages)
        else:
            sidecar_usable = _sidecar_torch_has_usable_runtime(site_packages, resolved_target_id)
        if runtime not in {"cpu", "default"} and not sidecar_usable:
            output_func(
                f"PyTorch runtime is installed, but {runtime} remains unavailable. "
                "The app will continue without GPU-accelerated learned models."
            )
            return False
        return True

    return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="shotsieve-desktop")
    parser.add_argument("--data-dir", default=None, help="Directory for the local ShotSieve cache and previews")
    parser.add_argument(
        "--model-cache-dir",
        default=None,
        help="Optional root for model downloads (preserves explicit HF_HOME, HF_HUB_CACHE, and TORCH_HOME)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host interface for the local review server")
    parser.add_argument("--port", type=int, default=8765, help="Port for the local review server")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open the default browser")
    return parser


def main() -> None:
    import multiprocessing
    multiprocessing.freeze_support()

    helper_exit_code = dispatch_sidecar_install_command()
    if helper_exit_code is not None:
        raise SystemExit(helper_exit_code)

    parser = build_parser()
    args = parser.parse_args()

    apply_model_cache_dir(args.model_cache_dir)
    data_dir = Path(args.data_dir).expanduser().resolve() if args.data_dir else default_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    recover_orphaned_preparation(data_dir)
    installed_torch_runtime = maybe_prepare_torch_runtime(data_dir)
    detected_target = canonical_release_target_id(
        runtime_target_id_from_executable_name() or _fallback_runtime_target_id()
    )
    torch_available = bool(
        installed_torch_runtime
        or _target_torch_package_is_available(data_dir, detected_target)
    )
    prepare_call_kwargs: dict[str, object] = {
        "assume_install_consent": bool(installed_torch_runtime),
    }
    try:
        prepare_parameters = inspect.signature(_call_prepare_learned_iqa_runtime).parameters
    except (TypeError, ValueError):
        prepare_parameters = {}
    if "torch_available" in prepare_parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in prepare_parameters.values()
    ):
        prepare_call_kwargs["torch_available"] = torch_available
    _call_prepare_learned_iqa_runtime(data_dir, **prepare_call_kwargs)
    db_path = data_dir / "shotsieve.db"

    serve_review_ui(
        db_path=db_path,
        host=args.host,
        port=args.port,
        open_browser=not args.no_browser,
    )


if __name__ == "__main__":
    main()
