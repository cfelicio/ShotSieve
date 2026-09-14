"""Tests for database schema migration and path-policy compatibility."""
from __future__ import annotations

import platform
import sqlite3
from hashlib import sha1
from pathlib import Path

import pytest

from shotsieve.db import connect, initialize_database, root_path_filter
from shotsieve.preview import preview_output_paths, stable_preview_name
from shotsieve.scanner import canonical_path_key
from shotsieve.schema import SCHEMA_SQL

def test_schema_contains_core_tables() -> None:
    assert "CREATE TABLE IF NOT EXISTS files" in SCHEMA_SQL
    assert "CREATE TABLE IF NOT EXISTS scores" in SCHEMA_SQL
    assert "CREATE TABLE IF NOT EXISTS review_state" in SCHEMA_SQL
    assert "CREATE TABLE IF NOT EXISTS scan_runs" in SCHEMA_SQL


def test_initialize_database_creates_preview_path_index(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"

    initialize_database(db_path)

    with connect(db_path) as connection:
        indexes = {
            row["name"]
            for row in connection.execute("PRAGMA index_list(files)").fetchall()
        }

    assert "idx_files_preview_path" in indexes


def test_initialize_database_creates_review_score_sort_indexes(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"

    initialize_database(db_path)

    with connect(db_path) as connection:
        indexes = {
            row["name"]
            for row in connection.execute("PRAGMA index_list(scores)").fetchall()
        }

    assert {
        "idx_scores_review_overall_desc_file",
        "idx_scores_review_learned_asc_file",
    }.issubset(indexes)


def test_initialize_database_adds_analysis_diagnostic_columns_to_existing_files_table(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    db_path.parent.mkdir()
    legacy_schema = SCHEMA_SQL.replace(
        "    analysis_status TEXT,\n    analysis_error TEXT,\n    last_analysis_time TEXT\n",
        "",
    ).replace(
        "    scan_status TEXT NOT NULL DEFAULT 'new',\n",
        "    scan_status TEXT NOT NULL DEFAULT 'new'\n",
    )
    with sqlite3.connect(db_path) as connection:
        connection.executescript(legacy_schema)

    initialize_database(db_path)

    with connect(db_path) as connection:
        columns = {row["name"] for row in connection.execute("PRAGMA table_info(files)").fetchall()}

    assert {"analysis_status", "analysis_error", "last_analysis_time"}.issubset(columns)


def test_canonical_path_key_preserves_case_on_case_sensitive_platform(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample_path = tmp_path / "Photos" / "A.jpg"
    expected = str(sample_path.resolve())

    monkeypatch.setattr(platform, "system", lambda: "Linux")

    assert canonical_path_key(sample_path) == expected


def test_stable_preview_name_hashes_normalized_path_key_on_case_insensitive_platform(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample_path = tmp_path / "Photos" / "A.jpg"

    monkeypatch.setattr(platform, "system", lambda: "Windows")

    expected = sha1(canonical_path_key(sample_path).encode("utf-8")).hexdigest()

    assert stable_preview_name(sample_path) == expected


def test_preview_output_paths_do_not_cleanup_casefold_compatibility_name_on_case_sensitive_platform(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preview_dir = tmp_path / "previews"
    sample_path = tmp_path / "Photos" / "Sample.jpg"

    monkeypatch.setattr(platform, "system", lambda: "Linux")

    preview_path, stale_paths = preview_output_paths(sample_path, preview_dir)
    casefold_compatibility_path = preview_dir / (
        f"{sha1(str(sample_path.resolve()).casefold().encode('utf-8')).hexdigest()}.jpg"
    )

    assert preview_path.name == f"{stable_preview_name(sample_path)}.jpg"
    assert casefold_compatibility_path not in stale_paths


def test_root_path_filter_matches_case_sensitive_roots_without_lowercasing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Linux")

    root_path = tmp_path / "Photos"
    matching_path = str((root_path / "A.jpg").resolve())
    other_case_path = str(((tmp_path / "photos") / "A.jpg").resolve())

    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.execute("CREATE TABLE files(path_key TEXT NOT NULL)")
    connection.executemany(
        "INSERT INTO files(path_key) VALUES(?)",
        [(matching_path,), (other_case_path,)],
    )

    try:
        clause, params = root_path_filter("path_key", root_path)
        rows = connection.execute(
            f"SELECT path_key FROM files WHERE {clause} ORDER BY path_key",
            tuple(params),
        ).fetchall()
    finally:
        connection.close()

    assert [row["path_key"] for row in rows] == [matching_path]


def test_root_path_filter_matches_non_bmp_descendants(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Linux")

    root_path = tmp_path / "Photos"
    basic_path = str((root_path / "A.jpg").resolve())
    emoji_path = str((root_path / "😀.jpg").resolve())

    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.execute("CREATE TABLE files(path_key TEXT NOT NULL)")
    connection.executemany(
        "INSERT INTO files(path_key) VALUES(?)",
        [(basic_path,), (emoji_path,)],
    )

    try:
        clause, params = root_path_filter("path_key", root_path)
        rows = connection.execute(
            f"SELECT path_key FROM files WHERE {clause} ORDER BY path_key",
            tuple(params),
        ).fetchall()
    finally:
        connection.close()

    assert [row["path_key"] for row in rows] == [basic_path, emoji_path]


def test_initialize_database_rebuilds_legacy_path_keys_for_case_sensitive_platform(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    sample_path = tmp_path / "Photos" / "A.jpg"

    # Force the first init to store a case-insensitive policy so that the
    # subsequent switch to Linux triggers an actual rebuild.  Without this,
    # the test is a no-op on Linux CI where the policy is already
    # 'case-sensitive-v1'.
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    initialize_database(db_path)
    with connect(db_path) as connection:
        connection.execute(
            "INSERT INTO files(path, path_key) VALUES(?, ?)",
            (str(sample_path.resolve()), str(sample_path.resolve()).casefold()),
        )
        connection.commit()

    monkeypatch.setattr(platform, "system", lambda: "Linux")

    initialize_database(db_path)

    with connect(db_path) as connection:
        row = connection.execute("SELECT path_key FROM files").fetchone()

    assert row["path_key"] == str(sample_path.resolve())


def test_initialize_database_raises_for_path_key_collisions_after_policy_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    photo_dir = tmp_path / "Photos"
    upper_path = photo_dir / "A.jpg"
    lower_path = photo_dir / "a.jpg"

    monkeypatch.setattr(platform, "system", lambda: "Linux")
    initialize_database(db_path)
    with connect(db_path) as connection:
        connection.executemany(
            "INSERT INTO files(path, path_key) VALUES(?, ?)",
            [
                (str(upper_path.resolve()), str(upper_path.resolve())),
                (str(lower_path.resolve()), str(lower_path.resolve())),
            ],
        )
        connection.commit()

    monkeypatch.setattr(platform, "system", lambda: "Windows")

    with pytest.raises(ValueError, match="path_key normalization collision"):
        initialize_database(db_path)
