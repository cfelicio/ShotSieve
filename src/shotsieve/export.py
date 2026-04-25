"""File export operations - copy or move selected files to a destination folder."""
from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from shotsieve.db import infer_preview_cache_roots, normalize_path_case
from shotsieve.preview import delete_managed_preview_file
from shotsieve.scanner import canonical_path_key


@dataclass(slots=True)
class ExportSummary:
    copied: int = 0
    moved: int = 0
    failed: list[dict] = field(default_factory=list)


def export_files(
    connection,
    *,
    file_ids: list[int],
    destination: str,
    mode: str = "copy",
    preview_cache_root: Path | None = None,
) -> ExportSummary:
    """Copy or move files to a destination directory.

    Args:
        connection: SQLite database connection.
        file_ids: List of file IDs to export.
        destination: Destination directory path.
        mode: "copy" or "move".
        preview_cache_root: Root directory of preview cache. Used to safely
            clean up old preview files after a move operation.

    Returns:
        ExportSummary with counts and any failures.

    Raises:
        ValueError: If mode is invalid, destination doesn't exist, or any
            file_ids are not found in the database.
    """
    if mode not in ("copy", "move"):
        raise ValueError(f"Export mode must be 'copy' or 'move', got '{mode}'")

    dest_path = Path(destination).expanduser().resolve()
    if not dest_path.is_absolute():
        dest_path = Path.cwd() / dest_path
    if not dest_path.is_dir():
        raise ValueError(f"Destination directory does not exist: {dest_path}")

    # Defense-in-depth: refuse system-critical directories.
    _reject_system_directory(dest_path)

    if not file_ids:
        return ExportSummary()

    # Deduplicate and normalize IDs.
    unique_ids = sorted(set(file_ids))

    placeholders = ",".join("?" for _ in unique_ids)
    rows = connection.execute(
        f"SELECT id, path, preview_path FROM files WHERE id IN ({placeholders}) ORDER BY id",
        unique_ids,
    ).fetchall()

    # Validate: all requested IDs must exist in the database.
    found_ids = {row["id"] for row in rows}
    missing_ids = [fid for fid in unique_ids if fid not in found_ids]
    if missing_ids:
        raise ValueError(f"File IDs not found in database: {missing_ids}")

    allow_preview_path_fallback = (
        preview_cache_root is not None
        and len(infer_preview_cache_roots(connection)) > 1
    )

    summary = ExportSummary()

    for row in rows:
        source = Path(row["path"])
        if not source.exists():
            summary.failed.append({"id": row["id"], "path": str(source), "error": "Source file not found"})
            continue

        target = _resolve_target(dest_path, source.name)

        if mode == "copy":
            try:
                shutil.copy2(str(source), str(target))
                summary.copied += 1
            except OSError as exc:
                summary.failed.append({"id": row["id"], "path": str(source), "error": str(exc)})
            continue

        try:
            shutil.move(str(source), str(target))
        except OSError as exc:
            summary.failed.append({"id": row["id"], "path": str(source), "error": str(exc)})
            continue

        try:
            # Update the cached path and clear preview so it gets regenerated.
            connection.execute(
                "UPDATE files SET path = ?, path_key = ?, preview_path = NULL, preview_status = 'missing' WHERE id = ?",
                (str(target), canonical_path_key(target), row["id"]),
            )

            # Persist successful move rows immediately so a later row's
            # database failure cannot roll back paths for files that have
            # already been moved on disk.
            connection.commit()
        except Exception:
            _restore_moved_source(source, target)
            raise

        summary.moved += 1

        try:
            # Clean up old preview file only after the database update succeeds.
            delete_managed_preview_file(
                row["preview_path"],
                source_path=row["path"],
                preview_cache_root=preview_cache_root,
                allow_path_parent_fallback=allow_preview_path_fallback,
            )
        except OSError as exc:
            summary.failed.append({"id": row["id"], "path": str(source), "error": str(exc)})

    return summary


def _resolve_target(dest_dir: Path, filename: str) -> Path:
    """Find a non-colliding target path, adding _2, _3, etc. if needed."""
    target = dest_dir / filename
    if not target.exists():
        return target

    stem = target.stem
    suffix = target.suffix
    max_attempts = 10_000
    for counter in range(2, 2 + max_attempts):
        candidate = dest_dir / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
    raise ValueError(f"Could not find a unique filename for {filename} after {max_attempts} attempts")


def _restore_moved_source(source: Path, target: Path) -> None:
    if not target.exists():
        return
    shutil.move(str(target), str(source))


def _reject_system_directory(dest: Path) -> None:
    """Raise ValueError if *dest* is inside a system-critical directory."""
    import platform

    system = platform.system()
    resolved = normalize_path_case(str(dest.resolve()))

    if system == "Windows":
        blocked = (
            normalize_path_case("c:\\windows"),
            normalize_path_case("c:\\program files"),
            normalize_path_case("c:\\program files (x86)"),
        )
    else:
        blocked = (
            normalize_path_case("/bin"),
            normalize_path_case("/sbin"),
            normalize_path_case("/usr/bin"),
            normalize_path_case("/usr/sbin"),
            normalize_path_case("/boot"),
            normalize_path_case("/proc"),
            normalize_path_case("/sys"),
        )

    for prefix in blocked:
        if resolved == prefix or resolved.startswith(prefix + ("/" if system != "Windows" else "\\")):
            raise ValueError(f"Cannot export to system directory: {dest}")
