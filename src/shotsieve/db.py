from __future__ import annotations

import platform
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from shotsieve.config import resolve_preview_dir
from shotsieve.schema import SCHEMA_MIGRATIONS, SCHEMA_SQL


PATH_KEY_NORMALIZATION_METADATA_KEY = "path_key_normalization_policy"
PREVIEW_CACHE_ROOT_METADATA_KEY = "preview_cache_root"
CASE_INSENSITIVE_PATH_PLATFORMS = {"Windows"}
TEXT_PREFIX_UPPER_BOUND = "\U0010FFFF"


def platform_uses_case_insensitive_paths() -> bool:
    return platform.system() in CASE_INSENSITIVE_PATH_PLATFORMS


def normalize_path_case(path_value: str) -> str:
    if platform_uses_case_insensitive_paths():
        return path_value.casefold()
    return path_value


def normalize_resolved_path(path: Path) -> str:
    return normalize_path_case(str(path.resolve()))


def current_path_key_normalization_policy() -> str:
    return "case-insensitive-v1" if platform_uses_case_insensitive_paths() else "case-sensitive-v1"


def connect(db_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(db_path, timeout=30.0)
    connection.row_factory = sqlite3.Row
    connection.create_function("unicode_casefold", 1, sqlite_unicode_casefold, deterministic=True)
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA busy_timeout=30000")
    return connection


def ensure_parent_dir(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)


def initialize_database(db_path: Path) -> None:
    ensure_parent_dir(db_path)

    connection = connect(db_path)
    try:
        connection.executescript(SCHEMA_SQL)
        apply_schema_migrations(connection)
        ensure_path_key_normalization(connection)
        connection.commit()
    finally:
        connection.close()


@contextmanager
def database(db_path: Path) -> Iterator[sqlite3.Connection]:
    ensure_parent_dir(db_path)
    connection = connect(db_path)

    try:
        yield connection
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()


def apply_schema_migrations(connection: sqlite3.Connection) -> None:
    for table_name, migrations in SCHEMA_MIGRATIONS.items():
        columns = {
            row["name"]
            for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()
        }
        for column_name, sql in migrations.items():
            if column_name not in columns:
                connection.execute(sql)


def escape_like(value: str) -> str:
    """Escape special characters for use in a LIKE clause with ESCAPE '\\'."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def sqlite_unicode_casefold(value: str | None) -> str | None:
    if value is None:
        return None
    return value.casefold()


def ensure_path_key_normalization(connection: sqlite3.Connection) -> None:
    current_policy = current_path_key_normalization_policy()
    existing_policy = get_metadata_value(connection, PATH_KEY_NORMALIZATION_METADATA_KEY)

    if existing_policy != current_policy:
        rebuild_path_keys(connection)
        set_metadata_value(connection, PATH_KEY_NORMALIZATION_METADATA_KEY, current_policy)


def get_metadata_value(connection: sqlite3.Connection, key: str) -> str | None:
    row = connection.execute(
        "SELECT value FROM app_metadata WHERE key = ?",
        (key,),
    ).fetchone()
    return row["value"] if row is not None else None


def set_metadata_value(connection: sqlite3.Connection, key: str, value: str) -> None:
    connection.execute(
        """
        INSERT INTO app_metadata(key, value)
        VALUES(?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (key, value),
    )


def set_preview_cache_root(connection: sqlite3.Connection, preview_dir: Path) -> Path:
    resolved_preview_dir = preview_dir.expanduser().resolve()
    _claim_preview_cache_root(resolved_preview_dir)
    set_metadata_value(connection, PREVIEW_CACHE_ROOT_METADATA_KEY, str(resolved_preview_dir))
    return resolved_preview_dir


def resolve_preview_cache_root(
    connection: sqlite3.Connection,
    *,
    db_path: Path | None = None,
) -> Path:
    stored_value = get_metadata_value(connection, PREVIEW_CACHE_ROOT_METADATA_KEY)
    if stored_value:
        return Path(stored_value).expanduser().resolve()

    inferred_root = infer_preview_cache_root(connection)
    if inferred_root is not None:
        return inferred_root

    effective_db_path = db_path or get_connection_db_path(connection)
    return resolve_preview_dir(None, db_path=effective_db_path)


def get_preview_cache_root(
    connection: sqlite3.Connection,
    *,
    db_path: Path | None = None,
    persist: bool = True,
) -> Path:
    resolved_root = resolve_preview_cache_root(connection, db_path=db_path)
    stored_value = get_metadata_value(connection, PREVIEW_CACHE_ROOT_METADATA_KEY)

    if stored_value is None and persist:
        inferred_root = infer_preview_cache_root(connection)
        if inferred_root is not None:
            try:
                set_preview_cache_root(connection, inferred_root)
            except ValueError:
                return inferred_root
            return inferred_root

        try:
            set_preview_cache_root(connection, resolved_root)
        except ValueError:
            return resolved_root

    return resolved_root


def infer_preview_cache_root(connection: sqlite3.Connection) -> Path | None:
    preview_roots = infer_preview_cache_roots(connection)
    if len(preview_roots) == 1:
        return preview_roots[0]
    return None


def infer_preview_cache_roots(connection: sqlite3.Connection) -> tuple[Path, ...]:
    rows = connection.execute(
        "SELECT preview_path FROM files WHERE preview_path IS NOT NULL"
    ).fetchall()
    preview_roots = sorted({
        Path(row["preview_path"]).expanduser().resolve().parent
        for row in rows
        if row["preview_path"]
    }, key=lambda path: str(path).casefold())
    return tuple(preview_roots)


def preview_cache_root_is_claimed(preview_dir: Path | None) -> bool:
    if preview_dir is None:
        return False

    try:
        resolved_preview_dir = preview_dir.expanduser().resolve()
    except OSError:
        return False

    return (resolved_preview_dir / ".shotsieve-preview-root").exists()


def get_connection_db_path(connection: sqlite3.Connection) -> Path:
    rows = connection.execute("PRAGMA database_list").fetchall()
    for row in rows:
        if row["name"] == "main":
            return Path(row["file"]).expanduser().resolve()
    raise ValueError("Unable to determine main database path for connection")


def _claim_preview_cache_root(preview_dir: Path) -> None:
    preview_dir.mkdir(parents=True, exist_ok=True)
    sentinel = preview_dir / ".shotsieve-preview-root"

    for child in preview_dir.iterdir():
        if child == sentinel:
            continue
        if child.is_dir():
            raise ValueError(
                f"Preview cache root must be dedicated to ShotSieve-managed files: {preview_dir}"
            )
        if not _looks_like_preview_cache_artifact(child):
            raise ValueError(
                f"Preview cache root contains unrelated files and cannot be claimed safely: {preview_dir}"
            )

    sentinel.touch(exist_ok=True)


def _looks_like_preview_cache_artifact(path: Path) -> bool:
    if path.name == ".shotsieve-preview-root":
        return True

    stem = path.stem.casefold()
    return path.suffix.casefold() == ".jpg" and len(stem) == 40 and all(char in "0123456789abcdef" for char in stem)


def rebuild_path_keys(connection: sqlite3.Connection) -> None:
    rows = connection.execute("SELECT id, path, path_key FROM files ORDER BY id").fetchall()
    if not rows:
        return

    updated_rows: list[tuple[str, int]] = []
    collision_sources: dict[str, list[str]] = {}

    for row in rows:
        rebuilt_key = normalize_resolved_path(Path(row["path"]))
        collision_sources.setdefault(rebuilt_key, []).append(row["path"])
        if row["path_key"] != rebuilt_key:
            updated_rows.append((rebuilt_key, row["id"]))

    collisions = [
        (rebuilt_key, paths)
        for rebuilt_key, paths in collision_sources.items()
        if len(paths) > 1
    ]
    if collisions:
        samples = "; ".join(
            f"{rebuilt_key}: {', '.join(paths)}"
            for rebuilt_key, paths in collisions[:3]
        )
        raise ValueError(
            "path_key normalization collision detected during rebuild; "
            f"resolve duplicate paths before continuing ({samples})"
        )

    if updated_rows:
        connection.executemany(
            "UPDATE files SET path_key = ? WHERE id = ?",
            updated_rows,
        )


def root_path_filter(column_name: str, root_path: Path) -> tuple[str, list[object]]:
    """Build a SQL predicate that matches a root path and its descendants only."""
    root_key = normalize_resolved_path(root_path)

    if root_key.endswith(("\\", "/")):
        return f"({column_name} >= ? AND {column_name} < ?)", [
            root_key,
            f"{root_key}{TEXT_PREFIX_UPPER_BOUND}",
        ]

    forward_prefix = f"{root_key}/"
    backward_prefix = f"{root_key}\\"

    clause = (
        f"({column_name} = ? OR "
        f"({column_name} >= ? AND {column_name} < ?) OR "
        f"({column_name} >= ? AND {column_name} < ?))"
    )
    return clause, [
        root_key,
        forward_prefix,
        f"{forward_prefix}{TEXT_PREFIX_UPPER_BOUND}",
        backward_prefix,
        f"{backward_prefix}{TEXT_PREFIX_UPPER_BOUND}",
    ]
