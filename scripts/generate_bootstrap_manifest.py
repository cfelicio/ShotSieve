from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from urllib.parse import quote


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from shotsieve.release_targets import ReleaseTarget, runtime_pack_release_targets  # noqa: E402


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
DEFAULT_REPOSITORY = "cfelicio/ShotSieve"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _archive_for_target(archive_root: Path, target: ReleaseTarget) -> Path:
    candidates = [path for path in archive_root.rglob(target.archiveName) if path.is_file()]
    if not candidates:
        raise SystemExit(
            f"Cannot generate bootstrap manifest: archive '{target.archiveName}' for target '{target.id}' was not found under '{archive_root}'."
        )
    if len(candidates) > 1:
        locations = ", ".join(str(path) for path in candidates)
        raise SystemExit(
            f"Cannot generate bootstrap manifest: archive '{target.archiveName}' for target '{target.id}' is ambiguous: {locations}"
        )
    return candidates[0]


def _release_archive_url(repository: str, release_tag: str, archive_name: str) -> str:
    encoded_tag = quote(release_tag, safe="")
    encoded_name = quote(archive_name, safe="")
    return f"https://github.com/{repository}/releases/download/{encoded_tag}/{encoded_name}"


def build_manifest(*, archive_root: Path, release_tag: str, repository: str = DEFAULT_REPOSITORY) -> dict[str, object]:
    normalized_tag = release_tag.strip()
    if not normalized_tag:
        raise SystemExit("Cannot generate bootstrap manifest without a release tag")

    assets: list[dict[str, object]] = []
    for target in runtime_pack_release_targets():
        archive_path = _archive_for_target(archive_root, target)
        digest = sha256_file(archive_path)
        if SHA256_PATTERN.fullmatch(digest) is None:
            raise SystemExit(f"Generated an invalid SHA-256 for archive '{archive_path}'")

        assets.append(
            {
                "id": target.id,
                "platform": target.platform,
                "runtime": target.runtime,
                "archive_name": target.archiveName,
                "executable_name": target.executableName,
                "variant_folder_name": target.variantFolderName,
                "url": _release_archive_url(repository, normalized_tag, target.archiveName),
                "sha256": digest,
            }
        )

    return {
        "version": 1,
        "repo": repository,
        "release_tag": normalized_tag,
        "assets": assets,
    }


def write_manifest(manifest: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate the checksummed ShotSieve bootstrap runtime manifest")
    parser.add_argument("--archive-root", default="dist", help="Directory containing built runtime archives")
    parser.add_argument("--output", default="dist/bootstrap-manifest.json", help="Manifest output path")
    parser.add_argument("--release-tag", required=True, help="GitHub release tag that owns the archives")
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY, help="GitHub owner/repository")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    archive_root = Path(args.archive_root).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    manifest = build_manifest(
        archive_root=archive_root,
        release_tag=args.release_tag,
        repository=args.repository,
    )
    write_manifest(manifest, output_path)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
