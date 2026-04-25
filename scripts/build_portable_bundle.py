from __future__ import annotations

import argparse
import importlib
import json
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import TypedDict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

_release_targets = importlib.import_module("shotsieve.release_targets")
ReleaseTarget = _release_targets.ReleaseTarget
all_release_targets = _release_targets.all_release_targets


class BundlePlan(TypedDict):
    target: dict[str, object]
    projectRoot: str
    pyinstallerDistRoot: str
    pyinstallerWorkRoot: str
    distPath: str
    archivePath: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a portable ShotSieve bundle for a Tier 1 release target")
    parser.add_argument("--target", required=True, help="Tier 1 release target id, for example linux-nvidia")
    parser.add_argument("--dist-root", default="dist", help="Root directory for staged bundles and archives")
    parser.add_argument("--build-root", default="build/release-targets", help="Root directory for PyInstaller work output")
    parser.add_argument("--plan", action="store_true", help="Print the resolved build plan as JSON and exit")
    return parser


def target_by_id(target_id: str) -> ReleaseTarget:
    targets = {target.id: target for target in all_release_targets()}
    try:
        return targets[target_id]
    except KeyError as exc:
        known = ", ".join(sorted(targets))
        raise SystemExit(f"Unknown release target '{target_id}'. Known targets: {known}") from exc


def target_plan(target: ReleaseTarget, *, project_root: Path, dist_root: Path, build_root: Path) -> BundlePlan:
    target_build_root = build_root / target.id
    pyinstaller_dist_root = target_build_root / "dist"
    pyinstaller_work_root = target_build_root / "work"
    staged_bundle = dist_root / target.variantFolderName
    archive_path = dist_root / target.archiveName
    return {
        "target": target.to_json(),
        "projectRoot": str(project_root),
        "pyinstallerDistRoot": str(pyinstaller_dist_root),
        "pyinstallerWorkRoot": str(pyinstaller_work_root),
        "distPath": str(staged_bundle),
        "archivePath": str(archive_path),
    }


def default_launcher_name(target: ReleaseTarget) -> str:
    return "ShotSieve.exe" if target.platform == "windows" else "ShotSieve"


def build_bundle(target: ReleaseTarget, *, project_root: Path, dist_root: Path, build_root: Path) -> BundlePlan:
    plan = target_plan(target, project_root=project_root, dist_root=dist_root, build_root=build_root)
    pyinstaller_dist_root = Path(plan["pyinstallerDistRoot"])
    pyinstaller_work_root = Path(plan["pyinstallerWorkRoot"])
    staged_bundle = Path(plan["distPath"])
    archive_path = Path(plan["archivePath"])
    spec_path = (project_root / target.specPath).resolve()

    if not spec_path.exists():
        raise SystemExit(f"Expected PyInstaller spec file '{spec_path}' for target '{target.id}'")

    for cleanup_path in (pyinstaller_dist_root, pyinstaller_work_root):
        if cleanup_path.exists():
            shutil.rmtree(cleanup_path)
        cleanup_path.mkdir(parents=True, exist_ok=True)
    dist_root.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--clean",
            "--distpath",
            str(pyinstaller_dist_root),
            "--workpath",
            str(pyinstaller_work_root),
            str(spec_path),
        ],
        check=True,
        cwd=project_root,
    )

    source_bundle = pyinstaller_dist_root / "ShotSieve"
    if staged_bundle.exists():
        shutil.rmtree(staged_bundle)
    shutil.copytree(source_bundle, staged_bundle)

    launcher = staged_bundle / default_launcher_name(target)
    if not launcher.exists():
        raise SystemExit(f"Expected launcher '{launcher}' was not created by PyInstaller")
    launcher.rename(staged_bundle / target.executableName)

    if archive_path.exists():
        archive_path.unlink()

    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for file_path in staged_bundle.rglob("*"):
                archive.write(file_path, file_path.relative_to(staged_bundle))
    else:
        with tarfile.open(archive_path, "w:gz") as archive:
            for path in staged_bundle.iterdir():
                archive.add(path, arcname=path.name)

    return plan


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    target = target_by_id(args.target)
    dist_root = (PROJECT_ROOT / args.dist_root).resolve()
    build_root = (PROJECT_ROOT / args.build_root).resolve()

    plan = target_plan(target, project_root=PROJECT_ROOT, dist_root=dist_root, build_root=build_root)
    if args.plan:
        print(json.dumps(plan, indent=2))
        return

    built_plan = build_bundle(target, project_root=PROJECT_ROOT, dist_root=dist_root, build_root=build_root)
    print(json.dumps(built_plan, indent=2))


if __name__ == "__main__":
    main()