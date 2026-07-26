from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.parse
import urllib.request
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


def _battr(name: str, fallback: Any) -> Any:
    mod = sys.modules.get("shotsieve.bootstrap")
    if mod is not None and hasattr(mod, name):
        return getattr(mod, name)
    return fallback


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

    bootstrap_file = _battr("__file__", __file__)
    source_checkout_root = runtime_support.source_checkout_root(bootstrap_file, package_name="shotsieve")
    if source_checkout_root is not None:
        return (source_checkout_root / "data" / PORTABLE_RUNTIME_DIRNAME).resolve()

    local_app_data = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
    if local_app_data:
        return Path(local_app_data).expanduser().resolve() / APP_DIRNAME / "runtime"
    return (Path.home() / f".{APP_DIRNAME.casefold()}" / "runtime").resolve()


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
        open_func = _battr("open_url", open_url)
        with open_func(manifest_url) as response:
            payload = response.read().decode("utf-8")
        return json.loads(payload)
    except (urllib.error.HTTPError, urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        if manifest_url != DEFAULT_MANIFEST_URL:
            err_func = _battr("_manifest_fetch_error_message", _manifest_fetch_error_message)
            raise SystemExit(err_func(manifest_url, exc)) from exc

        if isinstance(exc, urllib.error.HTTPError):
            fallback_func = _battr("_try_manifest_from_latest_release_api", _try_manifest_from_latest_release_api)
            github_fallback = fallback_func(manifest_url, status_code=exc.code)
            if github_fallback is not None:
                return github_fallback

        default_func = _battr("_build_default_latest_manifest", _build_default_latest_manifest)
        return default_func(DEFAULT_RELEASE_REPO)


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
                safe_join_func = _battr("_safe_join", _safe_join)
                safe_join_func(destination, member.filename)
                mode_bits = (member.external_attr >> 16) & 0o170000
                if mode_bits == 0o120000:
                    raise SystemExit(f"Unsupported archive member type: {member.filename}")

            for member in members:
                safe_join_func = _battr("_safe_join", _safe_join)
                target = safe_join_func(destination, member.filename)
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
            safe_join_func = _battr("_safe_join", _safe_join)
            safe_join_func(destination, member.name)
            if member.issym() or member.islnk() or member.isdev():
                raise SystemExit(f"Unsupported archive member type: {member.name}")

        for member in members:
            safe_join_func = _battr("_safe_join", _safe_join)
            target = safe_join_func(destination, member.name)

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
    norm_func = _battr("_normalize_manifest_location", _normalize_manifest_location)
    if cli_manifest_url:
        return norm_func(cli_manifest_url)

    env_manifest = os.environ.get("SHOTSIEVE_BOOTSTRAP_MANIFEST_URL")
    if env_manifest:
        return norm_func(env_manifest)

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
    search_roots_func = _battr("_local_search_roots", _local_search_roots)
    candidates: list[Path] = []
    for root in search_roots_func():
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
    candidates_func = _battr("local_runtime_archive_candidates", local_runtime_archive_candidates)
    for candidate in candidates_func(archive_name):
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
    gt_func = _battr("github_token", github_token)
    token = gt_func()
    if token and parsed.netloc in {"github.com", "api.github.com", "objects.githubusercontent.com"}:
        headers["Authorization"] = f"Bearer {token}"

    request = urllib.request.Request(url, headers=headers)
    mod = sys.modules.get("shotsieve.bootstrap")
    url_func = getattr(getattr(mod, "urllib", urllib), "request", urllib.request).urlopen if mod is not None else urllib.request.urlopen
    return url_func(request, timeout=DEFAULT_DOWNLOAD_TIMEOUT_SECONDS)


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

    open_func = _battr("open_url", open_url)
    try:
        with open_func(api_url) as response:
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
    open_func = _battr("open_url", open_url)
    try:
        with open_func(asset.url) as response:
            with archive_path.open("wb") as handle:
                shutil.copyfileobj(response, handle)
        return
    except (urllib.error.HTTPError, urllib.error.URLError):
        find_local_func = _battr("find_local_runtime_archive", find_local_runtime_archive)
        local_archive = find_local_func(asset.archive_name)
        if local_archive is not None:
            if local_archive.resolve() != archive_path.resolve():
                shutil.copy2(local_archive, archive_path)
            return

    candidates_func = _battr("local_runtime_archive_candidates", local_runtime_archive_candidates)
    local_candidates = ", ".join(str(path) for path in candidates_func(asset.archive_name))
    raise SystemExit(
        f"Failed to download runtime archive '{asset.archive_name}' for target '{asset.id}'. "
        f"URL: {asset.url}. Also could not find a local fallback archive. "
        f"Local search paths: {local_candidates}. "
        "If this repository is private, set SHOTSIEVE_GITHUB_TOKEN or GITHUB_TOKEN with release-read access."
    )


def ensure_runtime_asset(asset: RuntimeAsset, *, runtime_root: Path, force_refresh: bool = False) -> Path:
    if not force_refresh:
        colocated_func = _battr("_frozen_colocated_runtime_executable", _frozen_colocated_runtime_executable)
        colocated = colocated_func(asset)
        if colocated is not None:
            return colocated

    downloads_dir = runtime_root / "downloads"
    installs_dir = runtime_root / "installs"
    install_dir = installs_dir / asset.id
    marker_path = install_dir / ".asset-sha256"

    find_exe_func = _battr("_find_runtime_executable", _find_runtime_executable)
    if not force_refresh and install_dir.exists() and marker_path.exists():
        if asset.sha256:
            existing_hash = marker_path.read_text(encoding="utf-8").strip()
            if existing_hash == asset.sha256:
                return find_exe_func(install_dir, asset)
        else:
            try:
                return find_exe_func(install_dir, asset)
            except SystemExit:
                pass

    downloads_dir.mkdir(parents=True, exist_ok=True)
    installs_dir.mkdir(parents=True, exist_ok=True)

    archive_path = downloads_dir / asset.archive_name
    if force_refresh and archive_path.exists():
        archive_path.unlink()

    if not archive_path.exists():
        download_func = _battr("_download_archive_with_local_fallback", _download_archive_with_local_fallback)
        download_func(asset=asset, archive_path=archive_path)

    if asset.sha256:
        sha_func = _battr("sha256_file", sha256_file)
        downloaded_hash = sha_func(archive_path)
        if downloaded_hash != asset.sha256:
            raise SystemExit(
                f"Downloaded archive hash mismatch for target '{asset.id}'. Expected {asset.sha256}, got {downloaded_hash}."
            )

    extract_func = _battr("extract_archive", extract_archive)
    with tempfile.TemporaryDirectory(prefix=f"shotsieve-bootstrap-{asset.id}-") as temp_dir:
        temp_path = Path(temp_dir)
        extract_func(archive_path, temp_path)

        if install_dir.exists():
            shutil.rmtree(install_dir)
        shutil.move(str(temp_path), str(install_dir))

    marker_path.write_text(asset.sha256 or "", encoding="utf-8")

    try:
        archive_path.unlink(missing_ok=True)
    except OSError:
        pass

    return find_exe_func(install_dir, asset)
