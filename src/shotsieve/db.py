from __future__ import annotations

import platform
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Sequence

from shotsieve.config import (
    RAW_CAMERA_EXTENSIONS,
    RENDERED_COMPANION_EXTENSIONS,
    resolve_preview_dir,
)
from shotsieve.models import ScanRunDiagnostic
from shotsieve.schema import SCHEMA_MIGRATIONS, SCHEMA_SQL


PATH_KEY_NORMALIZATION_METADATA_KEY = "path_key_normalization_policy"
PREVIEW_CACHE_ROOT_METADATA_KEY = "preview_cache_root"
CASE_INSENSITIVE_PATH_PLATFORMS = {"Windows"}
TEXT_PREFIX_UPPER_BOUND = "\U0010FFFF"
_SCAN_DIAGNOSTIC_ATTRIBUTE = "_shotsieve_scan_diagnostic"
_SCAN_DIAGNOSTIC_PERSISTED_ATTRIBUTE = "_shotsieve_scan_diagnostic_persisted"
_SQLITE_ID_BATCH_SIZE = 500


def iter_sqlite_id_batches(connection, file_ids: Sequence[int]) -> Iterator[Sequence[int]]:
    """Yield ID batches that stay within SQLite's bound-variable limit.

    The 500-ID default is safe for connection wrappers without ``getlimit``
    and matches the review-selection batch size. Native SQLite connections may
    be configured with a lower limit, so honor it when available.
    """
    batch_size = _SQLITE_ID_BATCH_SIZE
    getlimit = getattr(connection, "getlimit", None)
    if callable(getlimit):
        batch_size = min(
            batch_size,
            int(getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER)),
        )
    if batch_size < 1:
        raise ValueError("SQLite variable limit must allow at least one ID")

    for offset in range(0, len(file_ids), batch_size):
        yield file_ids[offset : offset + batch_size]


class _ShotSieveConnection(sqlite3.Connection):
    """SQLite connection that preserves scan diagnostics on rollback.

    Most callers use :func:`database`, but scanner tests and library helpers
    also use ``with connect(...)`` directly.  Keeping the rollback hook on the
    connection makes both transaction owners behave consistently; the direct
    context also closes the connection after the transaction ends.
    """

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            if exc_value is None:
                self.commit()
            else:
                self.rollback()
                _persist_scan_diagnostic_after_rollback(self, exc_value)
        except BaseException:
            self.rollback()
            raise
        finally:
            self.close()
        return False


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
    connection = sqlite3.connect(db_path, timeout=30.0, factory=_ShotSieveConnection)
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
    except BaseException as exc:
        connection.rollback()
        _persist_scan_diagnostic_after_rollback(connection, exc)
        raise
    finally:
        connection.close()


def attach_scan_run_diagnostic(exc: BaseException, diagnostic: ScanRunDiagnostic) -> None:
    """Attach rollback-safe scan details without changing the public error."""
    try:
        setattr(exc, _SCAN_DIAGNOSTIC_ATTRIBUTE, diagnostic)
    except Exception:
        # Exception implementations supplied by callers may not allow custom
        # attributes.  The ordinary in-transaction diagnostic remains intact.
        pass


def scan_run_diagnostic_from_exception(exc: BaseException) -> ScanRunDiagnostic | None:
    diagnostic = getattr(exc, _SCAN_DIAGNOSTIC_ATTRIBUTE, None)
    return diagnostic if isinstance(diagnostic, ScanRunDiagnostic) else None


def scan_run_diagnostic_was_persisted(exc: BaseException) -> bool:
    return bool(getattr(exc, _SCAN_DIAGNOSTIC_PERSISTED_ATTRIBUTE, False))


def mark_scan_run_diagnostic_persisted(exc: BaseException) -> None:
    try:
        setattr(exc, _SCAN_DIAGNOSTIC_PERSISTED_ATTRIBUTE, True)
    except (AttributeError, TypeError):
        return


def persist_scan_run_diagnostic(db_path: Path, diagnostic: ScanRunDiagnostic) -> None:
    """Insert a scan diagnostic in a short independent transaction."""
    diagnostic_connection = connect(db_path)
    try:
        existing_row = diagnostic_connection.execute(
            "SELECT 1 FROM scan_runs WHERE root_path = ? AND started_time = ? LIMIT 1",
            (diagnostic.root_path, diagnostic.started_time),
        ).fetchone()
        if existing_row is not None:
            return

        diagnostic_connection.execute(
            """
            INSERT INTO scan_runs(
                started_time, completed_time, root_path,
                files_seen, files_added, files_updated, files_unchanged,
                files_removed, status, error_text
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                diagnostic.started_time,
                diagnostic.completed_time,
                diagnostic.root_path,
                diagnostic.files_seen,
                diagnostic.files_added,
                diagnostic.files_updated,
                diagnostic.files_unchanged,
                diagnostic.files_removed,
                diagnostic.status,
                diagnostic.error_text,
            ),
        )
        diagnostic_connection.commit()
    finally:
        diagnostic_connection.close()


def _persist_scan_diagnostic_after_rollback(
    connection: sqlite3.Connection,
    exc: BaseException,
) -> None:
    diagnostic = scan_run_diagnostic_from_exception(exc)
    if diagnostic is None or scan_run_diagnostic_was_persisted(exc):
        return

    try:
        db_path = get_connection_db_path(connection)
        persist_scan_run_diagnostic(db_path, diagnostic)
    except Exception:
        # Never replace the original scan failure with a secondary diagnostic
        # persistence error.  The failed scan remains visible through the job
        # error even if the database is unavailable at this point.
        return

    mark_scan_run_diagnostic_persisted(exc)


def apply_schema_migrations(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS file_pairs (
            primary_file_id INTEGER NOT NULL,
            companion_file_id INTEGER NOT NULL,
            pair_type TEXT NOT NULL DEFAULT 'raw_pair',
            created_time TEXT NOT NULL,
            PRIMARY KEY (primary_file_id, companion_file_id),
            FOREIGN KEY(primary_file_id) REFERENCES files(id) ON DELETE CASCADE,
            FOREIGN KEY(companion_file_id) REFERENCES files(id) ON DELETE CASCADE
        )
        """
    )
    connection.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_file_pairs_companion ON file_pairs(companion_file_id)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_file_pairs_primary ON file_pairs(primary_file_id)"
    )
    for table_name, migrations in SCHEMA_MIGRATIONS.items():
        columns = {
            row[1]
            for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()
        }
        for column_name, sql in migrations.items():
            if column_name not in columns:
                connection.execute(sql)


def candidate_companion_path_keys(file_path: Path) -> list[str]:
    """Return normalized path_key candidates for companion RAW or rendered pairs."""
    suffix = file_path.suffix.casefold()
    target_extensions: tuple[str, ...]
    if suffix in RAW_CAMERA_EXTENSIONS:
        target_extensions = tuple(RENDERED_COMPANION_EXTENSIONS)
    elif suffix in RENDERED_COMPANION_EXTENSIONS:
        target_extensions = tuple(RAW_CAMERA_EXTENSIONS)
    else:
        return []

    candidates: list[str] = []
    seen: set[str] = set()
    for ext in target_extensions:
        for candidate_ext in (ext.lower(), ext.upper()):
            candidate_path = file_path.with_suffix(candidate_ext)
            key = normalize_resolved_path(candidate_path)
            if key not in seen:
                seen.add(key)
                candidates.append(key)
    return candidates


def link_file_pair(
    connection: sqlite3.Connection,
    primary_file_id: int,
    companion_file_id: int,
    *,
    pair_type: str = "raw_pair",
) -> None:
    """Link a primary rendered file (JPEG/HEIF) with its companion RAW file."""
    now_text = datetime.now(timezone.utc).isoformat()
    connection.execute(
        """
        INSERT INTO file_pairs(primary_file_id, companion_file_id, pair_type, created_time)
        VALUES(?, ?, ?, ?)
        ON CONFLICT(primary_file_id, companion_file_id) DO NOTHING
        """,
        (primary_file_id, companion_file_id, pair_type, now_text),
    )


def get_all_paired_file_ids(
    connection: sqlite3.Connection,
    file_ids: Sequence[int],
) -> list[int]:
    """Return the input file_ids plus any linked companion or primary file IDs."""
    if not file_ids:
        return []

    input_ids = list(dict.fromkeys(int(fid) for fid in file_ids if int(fid) > 0))
    if not input_ids:
        return []

    all_ids = set(input_ids)
    for id_batch in iter_sqlite_id_batches(connection, input_ids):
        placeholders = ",".join("?" for _ in id_batch)
        batch_tuple = tuple(id_batch)
        rows_comp = connection.execute(
            f"SELECT companion_file_id AS paired_id FROM file_pairs WHERE primary_file_id IN ({placeholders})",
            batch_tuple,
        ).fetchall()
        for row in rows_comp:
            all_ids.add(int(row["paired_id"]))

        rows_prim = connection.execute(
            f"SELECT primary_file_id AS paired_id FROM file_pairs WHERE companion_file_id IN ({placeholders})",
            batch_tuple,
        ).fetchall()
        for row in rows_prim:
            all_ids.add(int(row["paired_id"]))

    return sorted(all_ids, key=lambda fid: (fid not in input_ids, input_ids.index(fid) if fid in input_ids else fid))


def reconcile_catalog_pairs(
    connection: sqlite3.Connection,
    root_path: Path | None = None,
) -> int:
    """Detect and record RAW+JPEG/HEIF pairs for unlinked catalog files.

    Returns the number of newly linked pairs.
    """
    raw_placeholders = ",".join("?" for _ in RAW_CAMERA_EXTENSIONS)
    query_parts = [
        f"""
        SELECT files.id, files.path, files.path_key
        FROM files
        LEFT JOIN file_pairs ON file_pairs.companion_file_id = files.id
        WHERE file_pairs.companion_file_id IS NULL
          AND unicode_casefold(files.format) IN ({raw_placeholders})
        """
    ]
    raw_formats = [ext.lstrip(".").casefold() for ext in RAW_CAMERA_EXTENSIONS]
    params: list[object] = list(raw_formats)

    if root_path is not None:
        root_clause, root_params = root_path_filter("files.path_key", root_path)
        query_parts.append(f"AND {root_clause}")
        params.extend(root_params)

    query_parts.append("ORDER BY files.id ASC")
    raw_rows = connection.execute(" ".join(query_parts), tuple(params)).fetchall()
    if not raw_rows:
        return 0

    newly_linked = 0
    now_text = datetime.now(timezone.utc).isoformat()
    for row in raw_rows:
        raw_id = int(row["id"])
        raw_path = Path(row["path"])
        candidates = candidate_companion_path_keys(raw_path)
        if not candidates:
            continue

        placeholders = ",".join("?" for _ in candidates)
        companion_row = connection.execute(
            f"SELECT id, path FROM files WHERE path_key IN ({placeholders}) LIMIT 1",
            tuple(candidates),
        ).fetchone()

        if companion_row is not None:
            primary_id = int(companion_row["id"])
            connection.execute(
                """
                INSERT INTO file_pairs(primary_file_id, companion_file_id, pair_type, created_time)
                VALUES(?, ?, 'raw_pair', ?)
                ON CONFLICT(primary_file_id, companion_file_id) DO NOTHING
                """,
                (primary_id, raw_id, now_text),
            )
            newly_linked += 1

    return newly_linked


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


def roots_path_filter(column_name: str, roots: Sequence[Path]) -> tuple[str, list[object]]:
    """Build a SQL predicate that matches any of the given roots and their descendants."""
    if not roots:
        return "0", []
    if len(roots) == 1:
        return root_path_filter(column_name, roots[0])

    clauses = []
    params = []
    for root in roots:
        clause, root_params = root_path_filter(column_name, root)
        clauses.append(clause)
        params.extend(root_params)

    return f"({' OR '.join(clauses)})", params

