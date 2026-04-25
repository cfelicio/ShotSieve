from __future__ import annotations

import argparse
from collections.abc import Callable
import contextlib
import hashlib
import importlib
import io
import json
import os
import platform
import pkgutil
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import traceback
import urllib.error
import urllib.parse
import urllib.request
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shotsieve.release_targets import runtime_pack_release_targets
from shotsieve import runtime_support


APP_DIRNAME = "ShotSieve"
PORTABLE_RUNTIME_DIRNAME = "runtime"
DEFAULT_RELEASE_REPO = "cfelicio/ShotSieve"
DEFAULT_MANIFEST_URL = "https://github.com/cfelicio/ShotSieve/releases/latest/download/bootstrap-manifest.json"
DEFAULT_DOWNLOAD_TIMEOUT_SECONDS = 60
DEFAULT_TORCH_AUTO_INSTALL_ENV = "SHOTSIEVE_BOOTSTRAP_AUTO_INSTALL_TORCH"
DEFAULT_TORCH_SITE_PACKAGES_DIRNAME = "site-packages"
DISTUTILS_REPLACEMENT_WARNING_PATTERN = r"Setuptools is replacing distutils\..*"


@contextlib.contextmanager
def _suppress_distutils_replacement_warning():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=DISTUTILS_REPLACEMENT_WARNING_PATTERN,
            category=UserWarning,
        )
        yield


@dataclass(slots=True, frozen=True)
class RuntimeAsset:
    id: str
    platform: str
    runtime: str
    url: str
    archive_name: str
    executable_name: str
    variant_folder_name: str
    sha256: str | None = None


def default_runtime_root() -> Path:
    if getattr(sys, "frozen", False):
        executable_dir = Path(sys.executable).resolve().parent
        return (executable_dir / PORTABLE_RUNTIME_DIRNAME).resolve()

    source_checkout_root = runtime_support.source_checkout_root(__file__, package_name="shotsieve")
    if source_checkout_root is not None:
        return (source_checkout_root / "data" / PORTABLE_RUNTIME_DIRNAME).resolve()

    local_app_data = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
    if local_app_data:
        return Path(local_app_data).expanduser().resolve() / APP_DIRNAME / "runtime"
    return (Path.home() / f".{APP_DIRNAME.casefold()}" / "runtime").resolve()


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
    for candidate in candidates:
        if _path_has_torch(candidate):
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

    try:
        with _suppress_distutils_replacement_warning():
            distlib_resources = importlib.import_module("pip._vendor.distlib.resources")
            distlib_package = importlib.import_module("pip._vendor.distlib")
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

    get_loader = getattr(pkgutil, "get_loader", None)
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
        pyi_importers = importlib.import_module("pyimod02_importers")
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
    try:
        with _suppress_distutils_replacement_warning():
            wheel_module = importlib.import_module("pip._internal.operations.install.wheel")
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
    try:
        with _suppress_distutils_replacement_warning():
            pip_module = importlib.import_module("pip._internal.cli.main")
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
    with _suppress_distutils_replacement_warning():
        _patch_distlib_finder_for_frozen()
        _patch_pip_scriptmaker_for_embedded_install()
        pip_main = _load_embedded_pip_main()
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
            "--no-deps",
            "--log",
            str(pip_log_path),
            "--target",
            str(site_packages),
            package_name,
            *_torch_install_index_args(runtime),
        ]
        if force_reinstall:
            install_args.insert(1, "--force-reinstall")

        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        exception_text: str | None = None

        try:
            with _suppress_distutils_replacement_warning():
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

    torch_return_code = _run_pip_install("torch")
    if torch_return_code != 0:
        if _path_has_torch(site_packages):
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

    return _path_has_torch(site_packages)


def install_torch_sidecar(
    *,
    runtime: str,
    site_packages: Path,
    output_func=print,
    force_reinstall: bool = False,
) -> bool:
    embedded_install_result = _install_torch_sidecar_with_embedded_pip(
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
    with _suppress_distutils_replacement_warning():
        _patch_distlib_finder_for_frozen()
        _patch_pip_scriptmaker_for_embedded_install()
        pip_main = _load_embedded_pip_main()
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

        try:
            with _suppress_distutils_replacement_warning():
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

    pyiqa_return_code = package_results.get("pyiqa", 1)
    if pyiqa_return_code != 0 and not _path_has_pyiqa(site_packages):
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

    return _path_has_pyiqa(site_packages)


def install_learned_iqa_sidecar(
    *,
    runtime: str,
    site_packages: Path,
    output_func=print,
    force_reinstall: bool = False,
) -> bool:
    embedded_install_result = _install_learned_iqa_sidecar_with_embedded_pip(
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
    asset: RuntimeAsset,
    *,
    install_dir: Path,
    runtime_root: Path,
    input_func=input,
    output_func=print,
) -> dict[str, str]:
    if runtime_bundle_contains_torch(install_dir):
        return {}

    site_packages = sidecar_site_packages_dir(runtime_root, asset.id)
    if _path_has_torch(site_packages):
        return {"PYTHONPATH": _compose_pythonpath(existing=os.environ.get("PYTHONPATH"), prepend_path=site_packages)}

    auto_install = _parse_env_bool(os.environ.get(DEFAULT_TORCH_AUTO_INSTALL_ENV))
    if auto_install is None:
        if not _is_interactive_console():
            return {}
        auto_install = _confirm(
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

    return {"PYTHONPATH": _compose_pythonpath(existing=os.environ.get("PYTHONPATH"), prepend_path=site_packages)}


def detect_nvidia_runtime() -> bool:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return False

    return completed.returncode == 0 and bool(completed.stdout.strip())


def select_runtime_target(*, system_name: str, machine_name: str, has_nvidia: bool) -> str:
    system = system_name.casefold()
    machine = machine_name.casefold()

    if system == "windows":
        return "windows-nvidia" if has_nvidia else "windows-cpu"

    if system == "linux":
        return "linux-nvidia" if has_nvidia else "linux-cpu"

    if system == "darwin":
        if machine in {"arm64", "aarch64"}:
            return "macos-mps"
        return "macos-cpu"

    raise SystemExit(f"Unsupported platform '{system_name}' for bootstrap launcher")


def fetch_manifest(manifest_url: str) -> dict[str, Any]:
    try:
        with open_url(manifest_url) as response:
            payload = response.read().decode("utf-8")
        return json.loads(payload)
    except (urllib.error.HTTPError, urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        if manifest_url != DEFAULT_MANIFEST_URL:
            raise SystemExit(_manifest_fetch_error_message(manifest_url, exc)) from exc

        if isinstance(exc, urllib.error.HTTPError):
            github_fallback = _try_manifest_from_latest_release_api(manifest_url, status_code=exc.code)
            if github_fallback is not None:
                return github_fallback

        return _build_default_latest_manifest(DEFAULT_RELEASE_REPO)


def select_manifest_asset(manifest: dict[str, Any], target_id: str) -> dict[str, Any]:
    raw_assets = manifest.get("assets")
    if not isinstance(raw_assets, list):
        raise SystemExit("Bootstrap manifest is missing an 'assets' list")

    for entry in raw_assets:
        if isinstance(entry, dict) and entry.get("id") == target_id:
            return entry

    known_ids: list[str] = []
    for entry in raw_assets:
        if not isinstance(entry, dict):
            continue
        asset_id = entry.get("id")
        if isinstance(asset_id, str):
            known_ids.append(asset_id)
    known_ids.sort()
    raise SystemExit(f"Bootstrap manifest does not include target '{target_id}'. Known targets: {', '.join(known_ids)}")


def parse_runtime_asset(entry: dict[str, Any]) -> RuntimeAsset:
    required_keys = (
        "id",
        "platform",
        "runtime",
        "url",
        "archive_name",
        "executable_name",
        "variant_folder_name",
    )
    missing = [key for key in required_keys if key not in entry]
    if missing:
        raise SystemExit(f"Bootstrap manifest entry is missing keys: {', '.join(missing)}")

    return RuntimeAsset(
        id=str(entry["id"]),
        platform=str(entry["platform"]),
        runtime=str(entry["runtime"]),
        url=str(entry["url"]),
        archive_name=str(entry["archive_name"]),
        executable_name=str(entry["executable_name"]),
        variant_folder_name=str(entry["variant_folder_name"]),
        sha256=str(entry["sha256"]) if entry.get("sha256") else None,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _safe_join(base_dir: Path, member_name: str) -> Path:
    candidate = (base_dir / member_name).resolve()
    if base_dir.resolve() not in [candidate, *candidate.parents]:
        raise SystemExit(f"Archive member escapes target directory: {member_name}")
    return candidate


def extract_archive(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)

    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as archive:
            members = archive.infolist()

            for member in members:
                _safe_join(destination, member.filename)
                mode_bits = (member.external_attr >> 16) & 0o170000
                if mode_bits == 0o120000:
                    raise SystemExit(f"Unsupported archive member type: {member.filename}")

            for member in members:
                target = _safe_join(destination, member.filename)
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue

                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member, "r") as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
        return

    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()

        for member in members:
            _safe_join(destination, member.name)
            if member.issym() or member.islnk() or member.isdev():
                raise SystemExit(f"Unsupported archive member type: {member.name}")

        for member in members:
            target = _safe_join(destination, member.name)

            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue

            if not member.isfile():
                raise SystemExit(f"Unsupported archive member type: {member.name}")

            extracted = archive.extractfile(member)
            if extracted is None:
                raise SystemExit(f"Unsupported archive member type: {member.name}")

            target.parent.mkdir(parents=True, exist_ok=True)
            with extracted, target.open("wb") as output:
                shutil.copyfileobj(extracted, output)


def resolve_manifest_url(cli_manifest_url: str | None) -> str:
    if cli_manifest_url:
        return _normalize_manifest_location(cli_manifest_url)

    env_manifest = os.environ.get("SHOTSIEVE_BOOTSTRAP_MANIFEST_URL")
    if env_manifest:
        return _normalize_manifest_location(env_manifest)

    return DEFAULT_MANIFEST_URL


def _local_search_roots() -> list[Path]:
    roots: list[Path] = []

    if getattr(sys, "frozen", False):
        exe_path = Path(sys.executable).resolve()
        current_root = exe_path.parent
        for _ in range(3):
            roots.append(current_root)
            parent = current_root.parent
            if parent == current_root:
                break
            current_root = parent

    roots.append(Path.cwd().resolve())

    unique: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if root in seen:
            continue
        seen.add(root)
        unique.append(root)
    return unique


def local_runtime_archive_candidates(archive_name: str) -> list[Path]:
    candidates: list[Path] = []
    for root in _local_search_roots():
        candidates.append(root / archive_name)
        candidates.append(root / "dist" / archive_name)

    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique.append(candidate)
    return unique


def find_local_runtime_archive(archive_name: str) -> Path | None:
    for candidate in local_runtime_archive_candidates(archive_name):
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _normalize_manifest_location(raw_location: str) -> str:
    parsed = urllib.parse.urlparse(raw_location)
    if parsed.scheme in {"http", "https", "file"}:
        return raw_location

    path_candidate = Path(raw_location).expanduser().resolve()
    if path_candidate.exists():
        return path_candidate.as_uri()

    return raw_location


def _manifest_fetch_error_message(manifest_url: str, error: Exception) -> str:
    return (
        "Failed to load bootstrap manifest. "
        f"Source: {manifest_url}. "
        f"Error: {error}. "
        "Run with --manifest-url <url-or-file> (or SHOTSIEVE_BOOTSTRAP_MANIFEST_URL). "
        "For private repositories, set SHOTSIEVE_GITHUB_TOKEN or GITHUB_TOKEN with release-read access."
    )


def github_token() -> str | None:
    token = os.environ.get("SHOTSIEVE_GITHUB_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token:
        token = token.strip()
    return token or None


def open_url(url: str):
    headers = {
        "User-Agent": "ShotSieve-Bootstrap/0.1",
    }

    parsed = urllib.parse.urlparse(url)
    token = github_token()
    if token and parsed.netloc in {"github.com", "api.github.com", "objects.githubusercontent.com"}:
        headers["Authorization"] = f"Bearer {token}"

    request = urllib.request.Request(url, headers=headers)
    return urllib.request.urlopen(request, timeout=DEFAULT_DOWNLOAD_TIMEOUT_SECONDS)


def _try_manifest_from_latest_release_api(manifest_url: str, *, status_code: int) -> dict[str, Any] | None:
    if status_code != 404:
        return None

    match = re.match(
        r"^https://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/releases/latest/download/bootstrap-manifest\.json$",
        manifest_url,
    )
    if not match:
        return None

    owner = match.group("owner")
    repo = match.group("repo")
    api_url = f"https://api.github.com/repos/{owner}/{repo}/releases/latest"

    try:
        with open_url(api_url) as response:
            release_payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return None

    raw_assets = release_payload.get("assets")
    if not isinstance(raw_assets, list):
        return None

    download_url_by_name: dict[str, str] = {}
    for entry in raw_assets:
        if not isinstance(entry, dict):
            continue
        asset_name = entry.get("name")
        browser_download_url = entry.get("browser_download_url")
        if isinstance(asset_name, str) and isinstance(browser_download_url, str):
            download_url_by_name[asset_name] = browser_download_url

    manifest_assets: list[dict[str, Any]] = []
    for target in runtime_pack_release_targets():
        asset_url = download_url_by_name.get(target.archiveName)
        if not asset_url:
            continue
        manifest_assets.append(
            {
                "id": target.id,
                "platform": target.platform,
                "runtime": target.runtime,
                "archive_name": target.archiveName,
                "executable_name": target.executableName,
                "variant_folder_name": target.variantFolderName,
                "url": asset_url,
                "sha256": None,
            }
        )

    if not manifest_assets:
        return None

    return {
        "version": 1,
        "repo": f"{owner}/{repo}",
        "release_tag": release_payload.get("tag_name", "latest"),
        "assets": manifest_assets,
    }


def _build_default_latest_manifest(repo: str) -> dict[str, Any]:
    assets: list[dict[str, Any]] = []
    for target in runtime_pack_release_targets():
        assets.append(
            {
                "id": target.id,
                "platform": target.platform,
                "runtime": target.runtime,
                "archive_name": target.archiveName,
                "executable_name": target.executableName,
                "variant_folder_name": target.variantFolderName,
                "url": f"https://github.com/{repo}/releases/latest/download/{target.archiveName}",
                "sha256": None,
            }
        )

    return {
        "version": 1,
        "repo": repo,
        "release_tag": "latest",
        "assets": assets,
    }


def _find_runtime_executable(install_dir: Path, asset: RuntimeAsset) -> Path:
    expected = install_dir / asset.variant_folder_name / asset.executable_name
    if expected.exists():
        return expected

    matches = list(install_dir.rglob(asset.executable_name))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise SystemExit(
            f"Runtime for target '{asset.id}' was extracted but executable '{asset.executable_name}' was not found"
        )
    raise SystemExit(
        f"Runtime for target '{asset.id}' has multiple '{asset.executable_name}' executables; cannot disambiguate"
    )


def _frozen_colocated_runtime_executable(asset: RuntimeAsset) -> Path | None:
    if not getattr(sys, "frozen", False):
        return None

    launcher_dir = Path(sys.executable).resolve().parent
    candidate = launcher_dir / asset.executable_name
    if candidate.exists() and candidate.is_file():
        return candidate
    return None


def _download_archive_with_local_fallback(*, asset: RuntimeAsset, archive_path: Path) -> None:
    try:
        with open_url(asset.url) as response:
            with archive_path.open("wb") as handle:
                shutil.copyfileobj(response, handle)
        return
    except (urllib.error.HTTPError, urllib.error.URLError):
        local_archive = find_local_runtime_archive(asset.archive_name)
        if local_archive is not None:
            if local_archive.resolve() != archive_path.resolve():
                shutil.copy2(local_archive, archive_path)
            return

    local_candidates = ", ".join(str(path) for path in local_runtime_archive_candidates(asset.archive_name))
    raise SystemExit(
        f"Failed to download runtime archive '{asset.archive_name}' for target '{asset.id}'. "
        f"URL: {asset.url}. Also could not find a local fallback archive. "
        f"Local search paths: {local_candidates}. "
        "If this repository is private, set SHOTSIEVE_GITHUB_TOKEN or GITHUB_TOKEN with release-read access."
    )


def ensure_runtime_asset(asset: RuntimeAsset, *, runtime_root: Path, force_refresh: bool = False) -> Path:
    if not force_refresh:
        colocated = _frozen_colocated_runtime_executable(asset)
        if colocated is not None:
            return colocated

    downloads_dir = runtime_root / "downloads"
    installs_dir = runtime_root / "installs"
    install_dir = installs_dir / asset.id
    marker_path = install_dir / ".asset-sha256"

    if not force_refresh and install_dir.exists() and marker_path.exists():
        if asset.sha256:
            existing_hash = marker_path.read_text(encoding="utf-8").strip()
            if existing_hash == asset.sha256:
                return _find_runtime_executable(install_dir, asset)
        else:
            try:
                return _find_runtime_executable(install_dir, asset)
            except SystemExit:
                pass

    downloads_dir.mkdir(parents=True, exist_ok=True)
    installs_dir.mkdir(parents=True, exist_ok=True)

    archive_path = downloads_dir / asset.archive_name
    if force_refresh and archive_path.exists():
        archive_path.unlink()

    if not archive_path.exists():
        _download_archive_with_local_fallback(asset=asset, archive_path=archive_path)

    if asset.sha256:
        downloaded_hash = sha256_file(archive_path)
        if downloaded_hash != asset.sha256:
            raise SystemExit(
                f"Downloaded archive hash mismatch for target '{asset.id}'. Expected {asset.sha256}, got {downloaded_hash}."
            )

    with tempfile.TemporaryDirectory(prefix=f"shotsieve-bootstrap-{asset.id}-") as temp_dir:
        temp_path = Path(temp_dir)
        extract_archive(archive_path, temp_path)

        if install_dir.exists():
            shutil.rmtree(install_dir)
        shutil.move(str(temp_path), str(install_dir))

    marker_path.write_text(asset.sha256 or "", encoding="utf-8")

    try:
        archive_path.unlink(missing_ok=True)
    except OSError:
        pass

    return _find_runtime_executable(install_dir, asset)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="shotsieve-bootstrap", description="ShotSieve bootstrap launcher")
    parser.add_argument("--manifest-url", default=None, help="Override bootstrap manifest URL")
    parser.add_argument("--runtime-root", default=None, help="Directory used to cache downloaded runtime packs")
    parser.add_argument("--target", default=None, help="Override runtime target id (for example windows-nvidia)")
    parser.add_argument("--force-refresh", action="store_true", help="Redownload and reinstall the selected runtime pack")
    parser.add_argument("--print-plan", action="store_true", help="Print resolved bootstrap plan and exit")
    parser.add_argument("--no-browser", action="store_true", help="Pass --no-browser to the runtime application")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments forwarded to the runtime application")
    return parser


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    runtime_root = Path(args.runtime_root).expanduser().resolve() if args.runtime_root else default_runtime_root()
    manifest_url = resolve_manifest_url(args.manifest_url)
    has_nvidia = detect_nvidia_runtime()

    selected_target = args.target or select_runtime_target(
        system_name=platform.system(),
        machine_name=platform.machine(),
        has_nvidia=has_nvidia,
    )

    manifest = fetch_manifest(manifest_url)
    asset_entry = select_manifest_asset(manifest, selected_target)
    asset = parse_runtime_asset(asset_entry)

    return {
        "manifestUrl": manifest_url,
        "runtimeRoot": str(runtime_root),
        "detectedTarget": selected_target,
        "asset": {
            "id": asset.id,
            "platform": asset.platform,
            "runtime": asset.runtime,
            "url": asset.url,
            "archiveName": asset.archive_name,
            "executableName": asset.executable_name,
            "variantFolderName": asset.variant_folder_name,
            "sha256": asset.sha256,
        },
        "forwardArgs": [arg for arg in args.args if arg != "--"],
        "forceRefresh": bool(args.force_refresh),
        "noBrowser": bool(args.no_browser),
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    plan = build_plan(args)

    if args.print_plan:
        print(json.dumps(plan, indent=2))
        return

    runtime_asset = RuntimeAsset(
        id=plan["asset"]["id"],
        platform=plan["asset"]["platform"],
        runtime=plan["asset"]["runtime"],
        url=plan["asset"]["url"],
        archive_name=plan["asset"]["archiveName"],
        executable_name=plan["asset"]["executableName"],
        variant_folder_name=plan["asset"]["variantFolderName"],
        sha256=plan["asset"]["sha256"],
    )

    executable = ensure_runtime_asset(
        runtime_asset,
        runtime_root=Path(plan["runtimeRoot"]),
        force_refresh=bool(plan["forceRefresh"]),
    )

    runtime_root = Path(plan["runtimeRoot"])
    install_dir = runtime_root / "installs" / runtime_asset.id
    env_updates = maybe_prepare_torch_runtime(
        runtime_asset,
        install_dir=install_dir,
        runtime_root=runtime_root,
    )
    launch_env = os.environ.copy()
    launch_env.update(env_updates)

    forwarded_args = list(plan["forwardArgs"])
    if plan["noBrowser"] and "--no-browser" not in forwarded_args:
        forwarded_args.append("--no-browser")

    completed = subprocess.run([str(executable), *forwarded_args], env=launch_env, check=False)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
