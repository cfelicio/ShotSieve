from __future__ import annotations

import contextlib
import csv
import errno
import hashlib
import importlib
import io
import json
import os
import platform
import pkgutil
import re
import secrets
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
import urllib.request
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from shotsieve.dependency_constraints import (
    COMMON_MODEL_REQUIREMENTS,
    PYTORCH_CPU_INDEX_URL,
    PYTORCH_CUDA_INDEX_URL,
    PYTORCH_XPU_INDEX_URL,
    ROCM_PYTHON_INDEX_URL,
    ROCM_SELECTOR_REQUIREMENT,
    ROCM_TORCH_REQUIREMENTS,
    TORCH_REQUIREMENTS,
    XPU_TORCH_REQUIREMENTS,
)
from shotsieve.release_targets import canonical_release_target_id
from shotsieve.runtime_support import (
    compose_runtime_dll_path,
    compose_pythonpath,
    confirm,
    is_interactive_console,
    parse_env_bool,
    path_has_pyiqa,
    path_has_torch,
)

DEFAULT_TORCH_AUTO_INSTALL_ENV = "SHOTSIEVE_BOOTSTRAP_AUTO_INSTALL_TORCH"
DEFAULT_TORCH_SITE_PACKAGES_DIRNAME = "site-packages"
SIDECAR_STATE_FILENAME = ".shotsieve-runtime.json"
SIDECAR_LOCK_SUFFIX = ".install.lock"
# Version 2 records the dependency-complete Torch sidecar install.  Version 1
# sidecars may contain an XPU torch DLL without the Intel runtime wheels that
# DLL requires, so they must be repaired once after this change.
SIDECAR_STATE_VERSION = 2
SIDECAR_LOCK_TIMEOUT_SECONDS = 300.0
SIDECAR_LOCK_POLL_SECONDS = 0.2
SIDECAR_INSTALL_COMMAND = "--_shotsieve-install-sidecar"
DISTUTILS_REPLACEMENT_WARNING_PATTERN = r"Setuptools is replacing distutils\..*"
PIP_UNEXPECTED_IMPORT_WARNING_PATTERN = r"DEPRECATION: Unexpected import of '.*' after pip install started\..*"

# The learned-IQA packages are installed after the Torch sidecar and some of
# them declare Torch as a dependency.  They therefore use --no-deps below so
# pip cannot replace an already-loaded Torch DLL on Windows.  The initial
# Torch sidecar install is staged separately and must resolve dependencies:
# Intel's XPU wheel needs the SYCL/compiler/oneMKL runtime wheels to load.
_LEARNED_IQA_NO_DEPS_PACKAGES = frozenset({
    "pyiqa",
    "timm",
    "openai-clip",
    "accelerate",
    # PyIQA imports FaceRestoreHelper from facexlib while registering TOPIQ,
    # including for the non-face topiq_nr model.  Keep its torch dependency
    # out of the already-loaded runtime sidecar.
    "facexlib",
})

# openai-clip is a source-only package from 2022.  Its isolated build invokes
# the frozen executable as if it were a normal pip runner, which makes the
# embedded installer reject the build-dependency arguments.  The bundled
# setuptools is sufficient when build isolation is disabled.
_LEARNED_IQA_NO_BUILD_ISOLATION_PACKAGES = frozenset({"openai-clip"})

# PyPI publishes openai-clip 1.0.1 as a source archive only.  A normal Python
# interpreter can build it, but a PyInstaller executable is also sys.executable
# and is therefore incorrectly invoked by pip's PEP 517 subprocess.  The
# frozen-runtime fallback below installs the pure-Python package files directly
# after verifying the pinned archive digest.
_OPENAI_CLIP_SOURCE_URL = (
    "https://files.pythonhosted.org/packages/3f/81/26d701ef9fface424b4ca808a5c5674df645ac46447720a540143d11c41e/"
    "openai-clip-1.0.1.tar.gz"
)
_OPENAI_CLIP_SOURCE_SHA256 = "cd40bf2f205c096c49524fcbff484339f793b52afd6e7ffad80a2fe108151721"
_OPENAI_CLIP_SOURCE_ROOT = "openai-clip-1.0.1"
_OPENAI_CLIP_DIST_INFO = "openai_clip-1.0.1.dist-info"

@dataclass(frozen=True, slots=True)
class TorchInstallPlan:
    """The complete source contract for one target's torch sidecar."""

    target_id: str
    platform: str
    runtime: str
    packages: tuple[str, ...]
    index_args: tuple[str, ...]

    @property
    def source_fingerprint(self) -> str:
        payload = json.dumps(
            {
                "target_id": self.target_id,
                "platform": self.platform,
                "runtime": self.runtime,
                "packages": self.packages,
                "index_args": self.index_args,
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def to_json(self) -> dict[str, object]:
        return {
            "target_id": self.target_id,
            "platform": self.platform,
            "runtime": self.runtime,
            "packages": list(self.packages),
            "index_args": list(self.index_args),
            "source_fingerprint": self.source_fingerprint,
        }


def _distribution_name(requirement: str) -> str:
    return requirement.split("==", 1)[0].split("[", 1)[0].strip().casefold()


def _learned_iqa_install_options(requirement: str) -> list[str]:
    distribution_name = _distribution_name(requirement)
    options: list[str] = []
    if distribution_name in _LEARNED_IQA_NO_DEPS_PACKAGES:
        options.append("--no-deps")
    if distribution_name in _LEARNED_IQA_NO_BUILD_ISOLATION_PACKAGES:
        options.append("--no-build-isolation")
    return options


def _install_openai_clip_source(site_packages: Path) -> None:
    """Install the pinned pure-Python openai-clip source package in place."""
    request = urllib.request.Request(
        _OPENAI_CLIP_SOURCE_URL,
        headers={"User-Agent": "ShotSieve-runtime/0.5.0"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        archive_bytes = response.read()

    digest = hashlib.sha256(archive_bytes).hexdigest()
    if digest != _OPENAI_CLIP_SOURCE_SHA256:
        raise RuntimeError(
            "The downloaded openai-clip archive failed its SHA-256 verification."
        )

    extracted_files = 0
    source_root = PurePosixPath(_OPENAI_CLIP_SOURCE_ROOT)
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as source_archive:
        for member in source_archive.getmembers():
            member_path = PurePosixPath(member.name)
            try:
                relative_path = member_path.relative_to(source_root)
            except ValueError:
                continue
            if not relative_path.parts or relative_path.parts[0] != "clip":
                continue
            if not member.isfile() or any(part in {"", ".", ".."} for part in relative_path.parts):
                continue

            destination = site_packages.joinpath(*relative_path.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = source_archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"The openai-clip archive member '{member.name}' could not be read.")
            with source, destination.open("wb") as output:
                shutil.copyfileobj(source, output)
            extracted_files += 1

    if extracted_files == 0 or not (site_packages / "clip" / "__init__.py").is_file():
        raise RuntimeError("The openai-clip archive did not contain its clip package.")

    dist_info = site_packages / _OPENAI_CLIP_DIST_INFO
    dist_info.mkdir(parents=True, exist_ok=True)
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\n"
        "Name: openai-clip\n"
        "Version: 1.0.1\n"
        "Summary: OpenAI CLIP\n"
        "Requires-Dist: ftfy\n"
        "Requires-Dist: regex\n"
        "Requires-Dist: tqdm\n",
        encoding="utf-8",
    )
    (dist_info / "WHEEL").write_text(
        "Wheel-Version: 1.0\n"
        "Generator: ShotSieve frozen-runtime bootstrap\n"
        "Root-Is-Purelib: true\n"
        "Tag: py3-none-any\n",
        encoding="utf-8",
    )
    (dist_info / "top_level.txt").write_text("clip\n", encoding="utf-8")


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


def _target_parts(target_id: str | None, runtime: str | None) -> tuple[str, str, str]:
    raw_target = str(target_id or "").strip().casefold()
    canonical_target = canonical_release_target_id(raw_target) if raw_target else ""
    normalized_runtime = str(runtime or "").strip().casefold()

    if canonical_target:
        if canonical_target.startswith("windows-"):
            target_platform = "windows"
        elif canonical_target.startswith("linux-"):
            target_platform = "linux"
        elif canonical_target.startswith("macos-"):
            target_platform = "macos"
        else:
            target_platform = platform.system().casefold()
        if canonical_target.endswith("-nvidia-cuda"):
            normalized_runtime = "cuda"
        elif canonical_target.endswith("-intel-xpu"):
            normalized_runtime = "xpu"
        elif canonical_target.endswith("-amd-rocm"):
            normalized_runtime = "rocm"
        elif canonical_target.endswith("-apple-mps"):
            normalized_runtime = "mps"
        elif canonical_target.endswith("-cpu"):
            normalized_runtime = "cpu"
    else:
        target_platform = platform.system().casefold()
        if target_platform == "darwin":
            target_platform = "macos"

    if target_platform == "darwin":
        target_platform = "macos"
    if not canonical_target:
        platform_prefix = {
            "windows": "windows",
            "linux": "linux",
            "macos": "macos",
        }.get(target_platform, target_platform)
        suffix = {
            "cuda": "nvidia-cuda",
            "xpu": "intel-xpu",
            "rocm": "amd-rocm",
            "mps": "apple-mps",
            "cpu": "cpu",
        }.get(normalized_runtime, "cpu")
        canonical_target = f"{platform_prefix}-{suffix}"

    return canonical_target, target_platform, normalized_runtime or "cpu"


def torch_install_plan(
    *,
    target_id: str | None = None,
    runtime: str | None = None,
    platform_name: str | None = None,
) -> TorchInstallPlan:
    """Resolve the exact pinned torch source for a target.

    The plan is intentionally explicit: vendor-specific ROCm URLs and indexes
    and the XPU index are part of the plan, so a missing vendor source is never
    replaced with a generic PyPI resolution.
    """
    resolved_target, target_platform, resolved_runtime = _target_parts(target_id, runtime)
    if platform_name:
        target_platform = platform_name.strip().casefold()
        if target_platform == "darwin":
            target_platform = "macos"

    if resolved_runtime == "cuda":
        packages = TORCH_REQUIREMENTS
        index_args = ("--index-url", PYTORCH_CUDA_INDEX_URL, "--trusted-host", "download.pytorch.org")
    elif resolved_runtime == "xpu":
        packages = XPU_TORCH_REQUIREMENTS
        index_args = (
            "--index-url",
            PYTORCH_XPU_INDEX_URL,
            "--extra-index-url",
            "https://pypi.org/simple",
        )
    elif resolved_runtime == "rocm":
        if target_platform not in {"windows", "linux"}:
            raise ValueError(f"ROCm is not supported for target platform '{target_platform}'.")
        packages = ROCM_TORCH_REQUIREMENTS
        index_args = (
            "--index-url",
            ROCM_PYTHON_INDEX_URL,
            "--extra-index-url",
            "https://pypi.org/simple",
        )
    elif resolved_runtime == "cpu" and target_platform in {"windows", "linux"}:
        packages = TORCH_REQUIREMENTS
        index_args = ("--index-url", PYTORCH_CPU_INDEX_URL, "--trusted-host", "download.pytorch.org")
    elif resolved_runtime in {"cpu", "mps", "default"}:
        packages = TORCH_REQUIREMENTS
        index_args = ()
    else:
        raise ValueError(f"Unsupported torch runtime '{resolved_runtime}'.")

    return TorchInstallPlan(
        target_id=resolved_target,
        platform=target_platform,
        runtime=resolved_runtime,
        packages=tuple(packages),
        index_args=tuple(index_args),
    )


def sidecar_state_path(site_packages: Path) -> Path:
    return site_packages / SIDECAR_STATE_FILENAME


def _read_sidecar_state(site_packages: Path) -> dict[str, object] | None:
    try:
        payload = json.loads(sidecar_state_path(site_packages).read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _write_json_atomically(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_sidecar_state(site_packages: Path, plan: TorchInstallPlan, *, learned_iqa: bool = False) -> None:
    payload = _read_sidecar_state(site_packages) or {}
    payload.update(
        {
            "schema": SIDECAR_STATE_VERSION,
            "kind": "runtime",
            "complete": True,
            "installed_at": time.time(),
            "plan": plan.to_json(),
        }
    )
    payload["learned_iqa_complete" if learned_iqa else "torch_complete"] = True
    _write_json_atomically(sidecar_state_path(site_packages), payload)


def torch_sidecar_is_valid(site_packages: Path, *, target_id: str | None = None, runtime: str | None = None) -> bool:
    """Reject interrupted or mismatched current-version sidecar installs."""
    if not (site_packages / "torch" / "__init__.py").is_file():
        return False

    state = _read_sidecar_state(site_packages)
    if state is None:
        return False

    plan = torch_install_plan(target_id=target_id or site_packages.name, runtime=runtime)
    stored_plan = state.get("plan")
    return bool(
        state.get("schema") == SIDECAR_STATE_VERSION
        and state.get("kind") == "runtime"
        and state.get("complete") is True
        and state.get("torch_complete") is True
        and isinstance(stored_plan, dict)
        and stored_plan.get("target_id") == plan.target_id
        and stored_plan.get("source_fingerprint") == plan.source_fingerprint
    )


@contextlib.contextmanager
def _sidecar_install_lock(site_packages: Path):
    """Use an atomic lock file so two launches cannot write one sidecar."""
    lock_path = site_packages.with_name(f".{site_packages.name}{SIDECAR_LOCK_SUFFIX}")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + SIDECAR_LOCK_TIMEOUT_SECONDS
    handle = None
    while handle is None:
        try:
            handle = lock_path.open("x", encoding="utf-8")
            handle.write(f"pid={os.getpid()}\n")
            handle.flush()
        except FileExistsError:
            # A process killed during a download cannot run the finally block.
            # Reclaim only a lock whose recorded owner is no longer alive;
            # active installs continue to block until the normal timeout.
            try:
                owner_text = lock_path.read_text(encoding="utf-8")
                owner_value = owner_text.partition("=")[2].strip()
                if owner_value:
                    os.kill(int(owner_value), 0)
            except (FileNotFoundError, ProcessLookupError):
                try:
                    lock_path.unlink(missing_ok=True)
                except OSError:
                    pass
            except ValueError:
                pass
            except OSError as exc:
                if exc.errno == errno.ESRCH:
                    try:
                        lock_path.unlink(missing_ok=True)
                    except OSError:
                        pass
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for sidecar install lock '{lock_path}'.")
            time.sleep(SIDECAR_LOCK_POLL_SECONDS)
    try:
        yield
    finally:
        try:
            handle.close()
        finally:
            lock_path.unlink(missing_ok=True)


def runtime_bundle_contains_torch(install_dir: Path) -> bool:
    candidates = (
        install_dir,
        install_dir / "_internal",
        install_dir / "Lib" / "site-packages",
        install_dir / "lib" / "site-packages",
    )
    for candidate in candidates:
        if path_has_torch(candidate):
            return True
    return False


def _torch_install_index_args(runtime: str) -> list[str]:
    return list(torch_install_plan(runtime=runtime).index_args)


def _torch_packages_for_runtime(runtime: str) -> tuple[str, ...]:
    return torch_install_plan(runtime=runtime).packages


def _patch_distlib_finder_for_frozen() -> None:
    if not getattr(sys, "frozen", False):
        return

    imp_mod = importlib
    pkg_mod = pkgutil
    suppress_func = _suppress_distutils_replacement_warning

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
    imp_mod = importlib
    suppress_func = _suppress_distutils_replacement_warning
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
    imp_mod = importlib
    suppress_func = _suppress_distutils_replacement_warning
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


def _commit_staged_sidecar(staging_dir: Path, site_packages: Path) -> None:
    """Publish a completed install without exposing its partial contents."""
    previous_dir = site_packages.with_name(f".{site_packages.name}.previous-{os.getpid()}")
    if previous_dir.exists():
        shutil.rmtree(previous_dir, ignore_errors=True)

    moved_previous = False
    try:
        if site_packages.exists():
            site_packages.rename(previous_dir)
            moved_previous = True
        staging_dir.rename(site_packages)
    except Exception:
        if moved_previous and not site_packages.exists() and previous_dir.exists():
            previous_dir.rename(site_packages)
        raise
    finally:
        if previous_dir.exists():
            shutil.rmtree(previous_dir, ignore_errors=True)


def _create_sidecar_staging_dir(parent_dir: Path) -> Path:
    """Create a short temporary sibling for Windows' legacy path limit.

    Torch includes deeply nested headers. A descriptive temporary name based
    on the target id can push those paths past MAX_PATH while copying an
    existing sidecar, even when the installed path itself is still usable.
    Every release target id is longer than this eight-character name.
    """
    parent_dir.mkdir(parents=True, exist_ok=True)
    for _ in range(128):
        staging_dir = parent_dir / f".s{secrets.token_hex(3)}"
        try:
            staging_dir.mkdir()
        except FileExistsError:
            continue
        return staging_dir
    raise FileExistsError(f"Could not allocate a temporary sidecar staging directory in {parent_dir}.")


def _run_sidecar_install_subprocess(
    *,
    operation: str,
    runtime: str,
    site_packages: Path,
    force_reinstall: bool,
    output_func=print,
) -> bool:
    """Run embedded pip outside the long-lived app process.

    pip installs a process-wide audit hook that warns about later imports and
    will reject them starting in pip 26.3. Isolating each sidecar install keeps
    that hook inside a short-lived helper instead of the desktop application.
    """
    helper_args = [
        SIDECAR_INSTALL_COMMAND,
        operation,
        runtime,
        str(site_packages),
        "1" if force_reinstall else "0",
    ]
    if getattr(sys, "frozen", False):
        command = [sys.executable, *helper_args]
        env = None
    else:
        command = [sys.executable, "-m", "shotsieve.bootstrap_sidecar", *helper_args]
        env = os.environ.copy()
        package_root = str(Path(__file__).resolve().parent.parent)
        existing_pythonpath = env.get("PYTHONPATH")
        pythonpath_entries = [package_root]
        if existing_pythonpath:
            pythonpath_entries.append(existing_pythonpath)
        env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)

    try:
        completed = subprocess.run(
            command,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except OSError as exc:
        output_func(f"Could not start the {operation} sidecar installer: {exc}")
        return False

    for stream in (completed.stdout, completed.stderr):
        if stream:
            message = stream.rstrip()
            if message:
                output_func(message)
    if completed.returncode != 0:
        output_func(
            f"The {operation} sidecar installer exited with code "
            f"{completed.returncode}."
        )
        return False
    return True


def dispatch_sidecar_install_command(argv: list[str] | None = None) -> int | None:
    """Dispatch the private helper command used by source and frozen builds."""
    command_args = list(sys.argv[1:] if argv is None else argv)
    if not command_args or command_args[0] != SIDECAR_INSTALL_COMMAND:
        return None
    if len(command_args) != 5:
        print(f"Invalid {SIDECAR_INSTALL_COMMAND} arguments.", file=sys.stderr)
        return 2

    _, operation, runtime, raw_site_packages, raw_force_reinstall = command_args
    if operation not in {"torch", "learned-iqa"} or raw_force_reinstall not in {"0", "1"}:
        print(f"Invalid {SIDECAR_INSTALL_COMMAND} arguments.", file=sys.stderr)
        return 2

    installer = {
        "torch": _install_torch_sidecar_with_embedded_pip,
        "learned-iqa": _install_learned_iqa_sidecar_with_embedded_pip,
    }[operation]
    try:
        installed = installer(
            runtime=runtime,
            site_packages=Path(raw_site_packages),
            force_reinstall=raw_force_reinstall == "1",
        )
    except Exception:
        traceback.print_exc()
        return 1
    if installed is None:
        print("Bundled pip runtime installer is unavailable in this build.", file=sys.stderr)
        return 1
    return 0 if installed else 1


def _install_torch_sidecar_with_embedded_pip(
    *,
    runtime: str,
    site_packages: Path,
    force_reinstall: bool = False,
    output_func=print,
) -> bool | None:
    suppress_pip = _suppress_embedded_pip_warnings
    with suppress_pip():
        p_distlib = _patch_distlib_finder_for_frozen
        p_script = _patch_pip_scriptmaker_for_embedded_install
        p_main = _load_embedded_pip_main
        p_distlib()
        p_script()
        pip_main = p_main()
    if pip_main is None:
        return None

    site_packages.parent.mkdir(parents=True, exist_ok=True)
    pip_log_path = site_packages / "pip-install.log"
    site_packages.mkdir(parents=True, exist_ok=True)
    try:
        pip_log_path.touch(exist_ok=True)
    except OSError:
        pass

    plan = torch_install_plan(target_id=site_packages.name, runtime=runtime)
    staging_dir = _create_sidecar_staging_dir(site_packages.parent)

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

    def _run_pip_install(
        package_names: tuple[str, ...],
        *,
        extra_args: tuple[str, ...] = (),
    ) -> int:
        package_label = " ".join(package_names)
        install_args = [
            "install",
            "--disable-pip-version-check",
            "--upgrade",
            "--no-cache-dir",
            "--log",
            str(pip_log_path),
            "--target",
            str(staging_dir),
            *package_names,
            *extra_args,
            *plan.index_args,
        ]
        if force_reinstall:
            install_args.insert(1, "--force-reinstall")

        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        exception_text: str | None = None

        try:
            with suppress_pip():
                with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
                    return_code = _coerce_pip_main_return_code(pip_main(install_args))
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
            package_name=package_label,
            install_args=install_args,
            return_code=return_code,
            stdout_text=stdout_buffer.getvalue(),
            stderr_text=stderr_buffer.getvalue(),
            exception_text=exception_text,
        )
        return return_code

    has_torch_func = path_has_torch
    try:
        with _sidecar_install_lock(site_packages), contextlib.ExitStack():
            # Another process may have completed the exact install while this
            # process waited for the lock.
            if not force_reinstall and torch_sidecar_is_valid(
                site_packages,
                target_id=plan.target_id,
                runtime=plan.runtime,
            ):
                return True

            torch_packages = plan.packages
            torch_install_extra_args: tuple[str, ...] = ()
            if (
                getattr(sys, "frozen", False)
                and plan.runtime == "rocm"
            ):
                selector_wheel_dir = site_packages.parent.parent / "wheels"
                selector_wheels = sorted(
                    selector_wheel_dir.glob("rocm-10.0.0-*.whl")
                )
                if len(selector_wheels) != 1:
                    output_func(
                        "The bundled ROCm 10 selector wheel is missing or ambiguous. "
                        "Reinstall the complete ShotSieve runtime pack before retrying."
                    )
                    _append_debug_log(
                        package_name="rocm==10.0.0",
                        install_args=[str(selector_wheel_dir)],
                        return_code=1,
                        stdout_text="",
                        stderr_text="Expected exactly one bundled rocm-10.0.0-*.whl.",
                        exception_text=None,
                    )
                    return False
                selector_wheel = selector_wheels[0]
                torch_packages = tuple(
                    str(selector_wheel)
                    if package == ROCM_SELECTOR_REQUIREMENT
                    else package
                    for package in torch_packages
                )
                torch_install_extra_args = (
                    "--find-links",
                    str(selector_wheel_dir),
                    "--only-binary=rocm",
                )

            torch_return_code = _run_pip_install(
                torch_packages,
                extra_args=torch_install_extra_args,
            )
            if torch_return_code != 0:
                output_func(
                    "PyTorch runtime installation failed with exit code "
                    f"{torch_return_code}. Check {pip_log_path} for details."
                )
                output_func("PyTorch installation failed. The app will continue without learned models.")
                return False

            if not has_torch_func(staging_dir):
                output_func(
                    "PyTorch installation returned success but the staged sidecar is incomplete. "
                    f"Check {pip_log_path} for details."
                )
                return False

            # Keep the durable log inside the published target directory even
            # though pip itself writes while the target is staged.
            try:
                shutil.copy2(pip_log_path, staging_dir / "pip-install.log")
            except OSError:
                pass
            _write_sidecar_state(staging_dir, plan)
            _commit_staged_sidecar(staging_dir, site_packages)
            state = _read_sidecar_state(site_packages) or {}
            stored_plan = state.get("plan")
            return bool(
                has_torch_func(site_packages)
                and state.get("schema") == SIDECAR_STATE_VERSION
                and state.get("complete") is True
                and isinstance(stored_plan, dict)
                and stored_plan.get("source_fingerprint") == plan.source_fingerprint
            )
    except TimeoutError as exc:
        output_func(f"PyTorch runtime installation is already in progress: {exc}")
        return False
    except Exception:
        output_func(
            "PyTorch runtime installation failed unexpectedly. "
            f"Check {pip_log_path} for details."
        )
        try:
            with pip_log_path.open("a", encoding="utf-8", errors="replace") as log_file:
                log_file.write("--- sidecar bootstrap exception ---\n")
                log_file.write(traceback.format_exc())
        except OSError:
            pass
        return False
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)


def install_torch_sidecar(
    *,
    runtime: str,
    site_packages: Path,
    output_func=print,
    force_reinstall: bool = False,
) -> bool:
    return _run_sidecar_install_subprocess(
        operation="torch",
        runtime=runtime,
        site_packages=site_packages,
        force_reinstall=force_reinstall,
        output_func=output_func,
    )


def _learned_iqa_packages_for_runtime(runtime: str) -> list[str]:
    packages = [
        *COMMON_MODEL_REQUIREMENTS,
        "opencv-python-headless",
        "pyyaml",
        "sympy",
        "requests",
        "tqdm",
        "scipy",
        "huggingface-hub",
        "pandas",
        # Direct dependencies of the torch-dependent packages above.  They
        # are installed explicitly so those packages can use --no-deps.
        "safetensors",
        "psutil",
        "ftfy",
        "regex",
    ]
    return packages


_LEARNED_IQA_TOP_LEVEL_ALIASES = {
    "opencv-python-headless": ("cv2",),
    "openai-clip": ("clip",),
    "pyyaml": ("yaml",),
    "huggingface-hub": ("huggingface_hub",),
}


def _normalized_distribution_name(value: str) -> str:
    return re.sub(r"[-_.]+", "_", value).casefold()


def _sidecar_distribution_name(metadata_dir: Path) -> str | None:
    metadata_path = metadata_dir / "METADATA"
    try:
        for line in metadata_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.casefold().startswith("name:"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass

    for suffix in (".dist-info", ".egg-info"):
        if metadata_dir.name.endswith(suffix):
            return metadata_dir.name[: -len(suffix)].rsplit("-", 1)[0]
    return None


def _safe_sidecar_path(site_packages: Path, relative_name: str) -> Path | None:
    relative_path = PurePosixPath(relative_name)
    if relative_path.is_absolute() or any(part in {"", ".", ".."} for part in relative_path.parts):
        return None
    candidate = site_packages.joinpath(*relative_path.parts)
    try:
        candidate.resolve().relative_to(site_packages.resolve())
    except ValueError:
        return None
    return candidate


def _purge_sidecar_distribution(site_packages: Path, requirement: str) -> None:
    """Remove one learned distribution before a clean staged reinstall.

    The Torch runtime and learned-IQA packages share one target directory. A
    staged learned repair must retain Torch, but it must not retain old
    Transformers or other learned package files that a ``--target`` install
    would otherwise leave behind.
    """
    distribution_name = _distribution_name(requirement)
    normalized_name = _normalized_distribution_name(distribution_name)
    metadata_dirs = [
        *site_packages.glob("*.dist-info"),
        *site_packages.glob("*.egg-info"),
    ]
    top_level_names: set[str] = set(
        _LEARNED_IQA_TOP_LEVEL_ALIASES.get(
            distribution_name,
            (distribution_name.replace("-", "_"),),
        )
    )

    for metadata_dir in metadata_dirs:
        metadata_name = _sidecar_distribution_name(metadata_dir) or ""
        if not metadata_dir.is_dir() or _normalized_distribution_name(metadata_name) != normalized_name:
            continue

        top_level_path = metadata_dir / "top_level.txt"
        try:
            top_level_names.update(
                line.strip()
                for line in top_level_path.read_text(encoding="utf-8", errors="replace").splitlines()
                if line.strip()
            )
        except OSError:
            pass

        record_path = metadata_dir / "RECORD"
        try:
            with record_path.open(newline="", encoding="utf-8", errors="replace") as record_file:
                for row in csv.reader(record_file):
                    if row and (recorded_path := _safe_sidecar_path(site_packages, row[0])) is not None:
                        try:
                            if recorded_path.is_file() or recorded_path.is_symlink():
                                recorded_path.unlink()
                        except OSError:
                            pass
        except OSError:
            pass

        shutil.rmtree(metadata_dir, ignore_errors=True)

    for top_level_name in top_level_names:
        top_level_path = _safe_sidecar_path(site_packages, top_level_name)
        if top_level_path is None:
            continue
        if top_level_path.is_dir():
            shutil.rmtree(top_level_path, ignore_errors=True)
        else:
            try:
                top_level_path.unlink()
            except OSError:
                pass


def _prepare_learned_iqa_staging(*, source_dir: Path, staging_dir: Path, runtime: str) -> None:
    """Copy the Torch base and remove learned packages before reinstalling."""
    if source_dir.exists():
        for source_path in source_dir.iterdir():
            if _is_python_bytecode_artifact(source_path.name):
                continue
            destination_path = staging_dir / source_path.name
            if source_path.is_dir() and not source_path.is_symlink():
                shutil.copytree(
                    source_path,
                    destination_path,
                    dirs_exist_ok=True,
                    ignore=_ignore_python_bytecode_artifacts,
                )
            else:
                shutil.copy2(source_path, destination_path)

    for package_name in _learned_iqa_packages_for_runtime(runtime):
        _purge_sidecar_distribution(staging_dir, package_name)


def _is_python_bytecode_artifact(name: str) -> bool:
    normalized_name = name.casefold()
    return normalized_name == "__pycache__" or normalized_name.endswith((".pyc", ".pyo"))


def _ignore_python_bytecode_artifacts(_directory: str, names: list[str]) -> list[str]:
    return [name for name in names if _is_python_bytecode_artifact(name)]


def _install_learned_iqa_sidecar_with_embedded_pip(
    *,
    runtime: str,
    site_packages: Path,
    force_reinstall: bool = False,
    output_func=print,
) -> bool | None:
    suppress_pip = _suppress_embedded_pip_warnings
    with suppress_pip():
        p_distlib = _patch_distlib_finder_for_frozen
        p_script = _patch_pip_scriptmaker_for_embedded_install
        p_main = _load_embedded_pip_main
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
        install_args.extend(_learned_iqa_install_options(package_name))

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

        if getattr(sys, "frozen", False) and _distribution_name(package_name) == "openai-clip":
            # Do not send this source-only package through pip in a frozen
            # process: PEP 517 would launch the ShotSieve EXE as its Python
            # interpreter.  The direct installer keeps this one package
            # compatible with the embedded runtime while preserving the
            # normal pip path for development/source installations.
            install_args = ["install-source-archive", package_name]
            try:
                _install_openai_clip_source(site_packages)
                return_code = 0
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

        try:
            with suppress_pip():
                with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
                    return_code = _coerce_pip_main_return_code(pip_main(install_args))
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
    for package_name in _learned_iqa_packages_for_runtime(runtime):
        package_results[package_name] = _run_pip_install(package_name)

    has_pyiqa_func = path_has_pyiqa
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
    site_packages = Path(site_packages)
    site_packages.parent.mkdir(parents=True, exist_ok=True)
    embedded_install_result: bool | None = False
    staging_dir = _create_sidecar_staging_dir(site_packages.parent)
    try:
        with _sidecar_install_lock(site_packages):
            _prepare_learned_iqa_staging(
                source_dir=site_packages,
                staging_dir=staging_dir,
                runtime=runtime,
            )
            embedded_install_result = _run_sidecar_install_subprocess(
                operation="learned-iqa",
                runtime=runtime,
                # A --target reinstall does not remove files from an older
                # package version. Build in a clean tree so Transformers and
                # its lazy-import modules cannot be mixed across versions.
                site_packages=staging_dir,
                force_reinstall=force_reinstall,
                output_func=output_func,
            )
            if embedded_install_result and path_has_pyiqa(staging_dir):
                _write_sidecar_state(
                    staging_dir,
                    torch_install_plan(target_id=site_packages.name, runtime=runtime),
                    learned_iqa=True,
                )
                _commit_staged_sidecar(staging_dir, site_packages)
    except TimeoutError as exc:
        output_func(f"Learned IQA installation is already in progress: {exc}")
        return False
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
    return embedded_install_result


def maybe_prepare_torch_runtime(
    asset,
    *,
    install_dir: Path,
    runtime_root: Path,
    input_func=input,
    output_func=print,
) -> dict[str, str]:
    if runtime_bundle_contains_torch(install_dir):
        return {}

    has_torch_func = path_has_torch
    site_packages = sidecar_site_packages_dir(runtime_root, asset.id)
    if torch_sidecar_is_valid(site_packages, target_id=asset.id, runtime=asset.runtime) and has_torch_func(site_packages):
        return _sidecar_environment(site_packages)

    auto_install = parse_env_bool(os.environ.get(DEFAULT_TORCH_AUTO_INSTALL_ENV))
    if auto_install is None:
        if not is_interactive_console():
            return {}
        auto_install = confirm(
            "PyTorch was not detected for this runtime. Download and install it now? [y/N]: ",
            input_func=input_func,
        )

    if not auto_install:
        output_func("Continuing without runtime PyTorch installation.")
        return {}

    output_func("Installing PyTorch runtime dependencies. This may take a few minutes...")
    installed = install_torch_sidecar(runtime=asset.runtime, site_packages=site_packages)
    if not installed:
        return {}

    return _sidecar_environment(site_packages)


def _sidecar_environment(site_packages: Path) -> dict[str, str]:
    """Build child-process environment updates for an installed sidecar."""
    updates = {
        "PYTHONPATH": compose_pythonpath(existing=os.environ.get("PYTHONPATH"), prepend_path=site_packages),
    }
    existing_path = os.environ.get("PATH")
    runtime_path = compose_runtime_dll_path(existing=existing_path, sidecar_path=site_packages)
    if runtime_path and runtime_path != (existing_path or ""):
        updates["PATH"] = runtime_path
    return updates


if __name__ == "__main__":
    helper_exit_code = dispatch_sidecar_install_command()
    raise SystemExit(2 if helper_exit_code is None else helper_exit_code)
