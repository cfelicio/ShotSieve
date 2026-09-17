from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from shotsieve.release_targets import runtime_pack_release_targets  # noqa: E402


# GitHub rejects release assets at or above 2 GiB. Keep a safety margin so the
# generated parts remain valid even if the platform's accounting is rounded.
GITHUB_RELEASE_ASSET_LIMIT_BYTES = 2 * 1024 * 1024 * 1024
RELEASE_PART_SIZE_BYTES = 1900 * 1024 * 1024
COPY_SUFFIXES = (".whl", ".zip", ".tar.gz")
READ_CHUNK_SIZE = 16 * 1024 * 1024


def _runtime_archive(source_root: Path, archive_name: str) -> Path:
    candidates = [path for path in source_root.rglob(archive_name) if path.is_file()]
    if not candidates:
        raise SystemExit(f"Runtime archive '{archive_name}' was not found under '{source_root}'.")
    if len(candidates) > 1:
        locations = ", ".join(str(path) for path in candidates)
        raise SystemExit(f"Runtime archive '{archive_name}' is ambiguous: {locations}")
    return candidates[0]


def _relative_output_path(source: Path, *, source_root: Path, output_root: Path) -> Path:
    return output_root / source.relative_to(source_root)


def _copy_release_file(source: Path, *, source_root: Path, output_root: Path) -> Path:
    destination = _relative_output_path(source, source_root=source_root, output_root=output_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def _split_runtime_archive(source: Path, *, source_root: Path, output_root: Path) -> list[Path]:
    destination_root = _relative_output_path(source, source_root=source_root, output_root=output_root).parent
    destination_root.mkdir(parents=True, exist_ok=True)

    part_paths: list[Path] = []
    part_index = 0
    with source.open("rb") as input_file:
        while True:
            part_path = destination_root / f"{source.name}.part-{part_index:03d}"
            bytes_written = 0
            with part_path.open("wb") as output_file:
                while bytes_written < RELEASE_PART_SIZE_BYTES:
                    chunk = input_file.read(min(READ_CHUNK_SIZE, RELEASE_PART_SIZE_BYTES - bytes_written))
                    if not chunk:
                        break
                    output_file.write(chunk)
                    bytes_written += len(chunk)

            if bytes_written == 0:
                part_path.unlink()
                break

            part_paths.append(part_path)
            part_index += 1

    if not part_paths:
        raise SystemExit(f"Runtime archive '{source}' was empty and could not be split.")

    # The original is a generated workflow artifact and is deliberately not
    # retained in the publish tree; otherwise the release glob could upload it
    # accidentally and the runner would need space for both copies.
    source.unlink()
    return part_paths


def prepare_release_assets(*, source_root: Path, output_root: Path) -> list[Path]:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    if source_root == output_root or source_root in output_root.parents or output_root in source_root.parents:
        raise SystemExit("The publish directory must be separate from the downloaded artifact directory.")

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    runtime_archive_names = {target.archiveName for target in runtime_pack_release_targets()}
    published: list[Path] = []
    for target in runtime_pack_release_targets():
        source = _runtime_archive(source_root, target.archiveName)
        source_size = source.stat().st_size
        if source_size < GITHUB_RELEASE_ASSET_LIMIT_BYTES:
            published.append(_copy_release_file(source, source_root=source_root, output_root=output_root))
            continue

        part_paths = _split_runtime_archive(source, source_root=source_root, output_root=output_root)
        published.extend(part_paths)
        print(
            f"Split {target.id} archive ({source_size} bytes before splitting) "
            f"into {len(part_paths)} GitHub release assets."
        )

    for source in source_root.rglob("*"):
        if not source.is_file() or output_root in source.parents or source.name in runtime_archive_names:
            continue
        if source.suffix == ".whl" or source.name.endswith(COPY_SUFFIXES[1:]):
            published.append(_copy_release_file(source, source_root=source_root, output_root=output_root))

    return published


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare GitHub release assets within the per-asset size limit")
    parser.add_argument("--source-root", required=True, help="Downloaded workflow artifacts root")
    parser.add_argument("--output-root", required=True, help="Staged publish asset root")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    published = prepare_release_assets(
        source_root=Path(args.source_root).expanduser(),
        output_root=Path(args.output_root).expanduser(),
    )
    print(f"Prepared {len(published)} release asset files.")


if __name__ == "__main__":
    main()
