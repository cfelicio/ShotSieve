from __future__ import annotations

import argparse
import importlib
import json
import os
import pkgutil
import platform
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from shotsieve.bootstrap_assets import (
    APP_DIRNAME,
    DEFAULT_DOWNLOAD_TIMEOUT_SECONDS,
    DEFAULT_MANIFEST_URL,
    DEFAULT_RELEASE_REPO,
    PORTABLE_RUNTIME_DIRNAME,
    RuntimeAsset,
    _build_default_latest_manifest,
    _download_archive_with_local_fallback,
    _find_runtime_executable,
    _frozen_colocated_runtime_executable,
    _local_search_roots,
    _manifest_fetch_error_message,
    _normalize_manifest_location,
    _safe_join,
    _try_manifest_from_latest_release_api,
    default_runtime_root,
    detect_nvidia_runtime,
    ensure_runtime_asset,
    extract_archive,
    fetch_manifest,
    find_local_runtime_archive,
    github_token,
    local_runtime_archive_candidates,
    open_url,
    parse_runtime_asset,
    resolve_manifest_url,
    select_manifest_asset,
    select_runtime_target,
    sha256_file,
)
from shotsieve.bootstrap_sidecar import (
    DEFAULT_TORCH_AUTO_INSTALL_ENV,
    DEFAULT_TORCH_SITE_PACKAGES_DIRNAME,
    DISTUTILS_REPLACEMENT_WARNING_PATTERN,
    PIP_UNEXPECTED_IMPORT_WARNING_PATTERN,
    _coerce_pip_main_return_code,
    _compose_pythonpath,
    _confirm,
    _install_learned_iqa_sidecar_with_embedded_pip,
    _install_torch_sidecar_with_embedded_pip,
    _is_interactive_console,
    _learned_iqa_packages_for_runtime,
    _load_embedded_pip_main,
    _parse_env_bool,
    _patch_distlib_finder_for_frozen,
    _patch_pip_scriptmaker_for_embedded_install,
    _path_has_pyiqa,
    _path_has_torch,
    _suppress_distutils_replacement_warning,
    _suppress_embedded_pip_warnings,
    _torch_install_index_args,
    install_learned_iqa_sidecar,
    install_torch_sidecar,
    maybe_prepare_torch_runtime,
    runtime_bundle_contains_torch,
    sidecar_site_packages_dir,
)

__all__ = [
    "APP_DIRNAME",
    "DEFAULT_DOWNLOAD_TIMEOUT_SECONDS",
    "DEFAULT_MANIFEST_URL",
    "DEFAULT_RELEASE_REPO",
    "DEFAULT_TORCH_AUTO_INSTALL_ENV",
    "DEFAULT_TORCH_SITE_PACKAGES_DIRNAME",
    "DISTUTILS_REPLACEMENT_WARNING_PATTERN",
    "PIP_UNEXPECTED_IMPORT_WARNING_PATTERN",
    "PORTABLE_RUNTIME_DIRNAME",
    "RuntimeAsset",
    "_build_default_latest_manifest",
    "_coerce_pip_main_return_code",
    "_compose_pythonpath",
    "_confirm",
    "_download_archive_with_local_fallback",
    "_find_runtime_executable",
    "_frozen_colocated_runtime_executable",
    "_install_learned_iqa_sidecar_with_embedded_pip",
    "_install_torch_sidecar_with_embedded_pip",
    "_is_interactive_console",
    "_learned_iqa_packages_for_runtime",
    "_load_embedded_pip_main",
    "_local_search_roots",
    "_manifest_fetch_error_message",
    "_normalize_manifest_location",
    "_parse_env_bool",
    "_patch_distlib_finder_for_frozen",
    "_patch_pip_scriptmaker_for_embedded_install",
    "_path_has_pyiqa",
    "_path_has_torch",
    "_safe_join",
    "_suppress_distutils_replacement_warning",
    "_suppress_embedded_pip_warnings",
    "_torch_install_index_args",
    "_try_manifest_from_latest_release_api",
    "build_parser",
    "build_plan",
    "default_runtime_root",
    "detect_nvidia_runtime",
    "ensure_runtime_asset",
    "extract_archive",
    "fetch_manifest",
    "find_local_runtime_archive",
    "github_token",
    "importlib",
    "install_learned_iqa_sidecar",
    "install_torch_sidecar",
    "local_runtime_archive_candidates",
    "main",
    "maybe_prepare_torch_runtime",
    "open_url",
    "parse_runtime_asset",
    "pkgutil",
    "resolve_manifest_url",
    "runtime_bundle_contains_torch",
    "select_manifest_asset",
    "select_runtime_target",
    "sha256_file",
    "sidecar_site_packages_dir",
    "urllib",
]


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
