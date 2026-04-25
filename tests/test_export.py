"""Tests for the export module - copy, move, collision handling, and validation."""
from __future__ import annotations

from pathlib import Path
import sqlite3
from typing import cast

import pytest
from PIL import Image

from shotsieve.db import database, initialize_database
from shotsieve.export import _reject_system_directory, export_files
from shotsieve.scanner import scan_root


def create_image(path: Path) -> None:
    image = Image.new("RGB", (120, 80), color=(40, 90, 160))
    image.save(path, format="JPEG")


def setup_library(tmp_path: Path):
    """Create a library with 3 test images, scan them, return (db_path, photo_dir, ids_by_name)."""
    db_path = tmp_path / "data" / "shotsieve.db"
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()

    for name in ("alpha.jpg", "beta.jpg", "gamma.jpg"):
        create_image(photo_dir / name)

    initialize_database(db_path)
    with database(db_path) as connection:
        scan_root(connection, root=photo_dir, recursive=True, extensions=(".jpg",), preview_dir=tmp_path / "previews")
        ids_by_name = {
            Path(row["path"]).name: row["id"]
            for row in connection.execute("SELECT id, path FROM files").fetchall()
        }

    return db_path, photo_dir, ids_by_name


class TestCopyFiles:
    def test_copy_creates_files_at_destination(self, tmp_path: Path):
        db_path, photo_dir, ids_by_name = setup_library(tmp_path)
        dest = tmp_path / "export"
        dest.mkdir()

        with database(db_path) as connection:
            result = export_files(
                connection,
                file_ids=[ids_by_name["alpha.jpg"], ids_by_name["beta.jpg"]],
                destination=str(dest),
                mode="copy",
            )

        assert result.copied == 2
        assert result.moved == 0
        assert len(result.failed) == 0
        assert (dest / "alpha.jpg").exists()
        assert (dest / "beta.jpg").exists()
        # Originals still exist
        assert (photo_dir / "alpha.jpg").exists()
        assert (photo_dir / "beta.jpg").exists()

    def test_copy_preserves_original_database_paths(self, tmp_path: Path):
        db_path, photo_dir, ids_by_name = setup_library(tmp_path)
        dest = tmp_path / "export"
        dest.mkdir()

        with database(db_path) as connection:
            export_files(
                connection,
                file_ids=[ids_by_name["alpha.jpg"]],
                destination=str(dest),
                mode="copy",
            )
            row = connection.execute("SELECT path FROM files WHERE id = ?", (ids_by_name["alpha.jpg"],)).fetchone()

        assert str(photo_dir) in row["path"]


class TestMoveFiles:
    def test_move_removes_originals_and_updates_cache(self, tmp_path: Path):
        db_path, photo_dir, ids_by_name = setup_library(tmp_path)
        dest = tmp_path / "export"
        dest.mkdir()

        with database(db_path) as connection:
            result = export_files(
                connection,
                file_ids=[ids_by_name["alpha.jpg"]],
                destination=str(dest),
                mode="move",
            )
            row = connection.execute("SELECT path FROM files WHERE id = ?", (ids_by_name["alpha.jpg"],)).fetchone()

        assert result.moved == 1
        assert result.copied == 0
        assert not (photo_dir / "alpha.jpg").exists()
        assert (dest / "alpha.jpg").exists()
        assert str(dest) in row["path"]


class TestCollisionHandling:
    def test_collision_appends_suffix(self, tmp_path: Path):
        db_path, photo_dir, ids_by_name = setup_library(tmp_path)
        dest = tmp_path / "export"
        dest.mkdir()
        # Pre-place a file with the same name
        create_image(dest / "alpha.jpg")

        with database(db_path) as connection:
            result = export_files(
                connection,
                file_ids=[ids_by_name["alpha.jpg"]],
                destination=str(dest),
                mode="copy",
            )

        assert result.copied == 1
        assert (dest / "alpha_2.jpg").exists()


class TestValidation:
    def test_rejects_invalid_mode(self, tmp_path: Path):
        db_path, _, ids_by_name = setup_library(tmp_path)

        with database(db_path) as connection:
            with pytest.raises(ValueError, match="mode"):
                export_files(connection, file_ids=list(ids_by_name.values()), destination=str(tmp_path), mode="delete")

    def test_rejects_missing_destination(self, tmp_path: Path):
        db_path, _, ids_by_name = setup_library(tmp_path)

        with database(db_path) as connection:
            with pytest.raises(ValueError, match="Destination"):
                export_files(connection, file_ids=list(ids_by_name.values()), destination=str(tmp_path / "nonexistent"), mode="copy")

    def test_empty_file_ids_returns_zero_summary(self, tmp_path: Path):
        db_path, _, _ = setup_library(tmp_path)

        with database(db_path) as connection:
            result = export_files(connection, file_ids=[], destination=str(tmp_path), mode="copy")

        assert result.copied == 0
        assert result.moved == 0
        assert len(result.failed) == 0

    def test_missing_source_file_reports_failure(self, tmp_path: Path):
        db_path, photo_dir, ids_by_name = setup_library(tmp_path)
        dest = tmp_path / "export"
        dest.mkdir()
        # Delete the source file
        (photo_dir / "alpha.jpg").unlink()

        with database(db_path) as connection:
            result = export_files(
                connection,
                file_ids=[ids_by_name["alpha.jpg"]],
                destination=str(dest),
                mode="copy",
            )

        assert result.copied == 0
        assert len(result.failed) == 1
        assert "not found" in result.failed[0]["error"]

    def test_export_raises_for_nonexistent_ids(self, tmp_path: Path):
        db_path, _, ids_by_name = setup_library(tmp_path)
        dest = tmp_path / "export"
        dest.mkdir()

        with database(db_path) as connection:
            with pytest.raises(ValueError, match="not found"):
                export_files(
                    connection,
                    file_ids=[999],
                    destination=str(dest),
                    mode="copy",
                )

    def test_export_raises_for_partially_invalid_ids(self, tmp_path: Path):
        db_path, _, ids_by_name = setup_library(tmp_path)
        dest = tmp_path / "export"
        dest.mkdir()

        with database(db_path) as connection:
            with pytest.raises(ValueError, match="not found"):
                export_files(
                    connection,
                    file_ids=[ids_by_name["alpha.jpg"], 999],
                    destination=str(dest),
                    mode="copy",
                )

    def test_case_sensitive_system_directory_check_preserves_distinct_case(self, monkeypatch: pytest.MonkeyPatch):
        class FakeResolvedPath:
            def __init__(self, resolved_path: str) -> None:
                self._resolved_path = resolved_path

            def resolve(self) -> FakeResolvedPath:
                return self

            def __str__(self) -> str:
                return self._resolved_path

        monkeypatch.setattr("platform.system", lambda: "Linux")

        _reject_system_directory(cast(Path, FakeResolvedPath("/USR/BIN")))


class TestMovePreviewCleanup:
    def test_move_deletes_old_preview_file(self, tmp_path: Path):
        db_path, photo_dir, ids_by_name = setup_library(tmp_path)
        preview_dir = tmp_path / "previews"
        dest = tmp_path / "export"
        dest.mkdir()

        with database(db_path) as connection:
            # Confirm preview exists
            row = connection.execute(
                "SELECT preview_path FROM files WHERE id = ?",
                (ids_by_name["alpha.jpg"],),
            ).fetchone()
            preview_path = Path(row["preview_path"])
            assert preview_path.exists()

            result = export_files(
                connection,
                file_ids=[ids_by_name["alpha.jpg"]],
                destination=str(dest),
                mode="move",
                preview_cache_root=preview_dir,
            )

        assert result.moved == 1
        assert not preview_path.exists(), "Old preview file should be deleted after move"

    def test_move_keeps_preview_outside_cache_root(self, tmp_path: Path):
        db_path, photo_dir, ids_by_name = setup_library(tmp_path)
        preview_dir = tmp_path / "previews"
        external_preview_dir = tmp_path / "external-previews"
        external_preview_dir.mkdir()
        external_preview = external_preview_dir / "alpha-preview.jpg"
        create_image(external_preview)
        dest = tmp_path / "export"
        dest.mkdir()

        with database(db_path) as connection:
            connection.execute(
                "UPDATE files SET preview_path = ? WHERE id = ?",
                (str(external_preview.resolve()), ids_by_name["alpha.jpg"]),
            )

            result = export_files(
                connection,
                file_ids=[ids_by_name["alpha.jpg"]],
                destination=str(dest),
                mode="move",
                preview_cache_root=preview_dir,
            )

        assert result.moved == 1
        assert external_preview.exists(), "Preview outside the cache root should be preserved"

    def test_move_without_cache_root_leaves_preview(self, tmp_path: Path):
        db_path, photo_dir, ids_by_name = setup_library(tmp_path)
        dest = tmp_path / "export"
        dest.mkdir()

        with database(db_path) as connection:
            row = connection.execute(
                "SELECT preview_path FROM files WHERE id = ?",
                (ids_by_name["alpha.jpg"],),
            ).fetchone()
            preview_path = Path(row["preview_path"])
            assert preview_path.exists()

            result = export_files(
                connection,
                file_ids=[ids_by_name["alpha.jpg"]],
                destination=str(dest),
                mode="move",
                # No preview_cache_root — should leave preview file alone
            )

        assert result.moved == 1
        assert preview_path.exists(), "Preview should remain when no cache root is provided"

    def test_move_restores_source_and_preview_when_database_update_fails(self, tmp_path: Path):
        db_path, photo_dir, ids_by_name = setup_library(tmp_path)
        preview_dir = tmp_path / "previews"
        dest = tmp_path / "export"
        dest.mkdir()

        class FailingUpdateConnection:
            def __init__(self, inner_connection):
                self._inner = inner_connection

            def execute(self, sql: str, params=()):
                if sql.startswith("UPDATE files SET path = ?, path_key = ?, preview_path = NULL, preview_status = 'missing'"):
                    raise sqlite3.OperationalError("simulated export update failure")
                return self._inner.execute(sql, params)

            def __getattr__(self, name: str):
                return getattr(self._inner, name)

        source_path = photo_dir / "alpha.jpg"
        target_path = dest / "alpha.jpg"

        with database(db_path) as connection:
            row = connection.execute(
                "SELECT path, preview_path FROM files WHERE id = ?",
                (ids_by_name["alpha.jpg"],),
            ).fetchone()
            original_db_path = row["path"]
            preview_path = Path(row["preview_path"])
            assert source_path.exists()
            assert preview_path.exists()

            failing_connection = FailingUpdateConnection(connection)

            with pytest.raises(sqlite3.OperationalError, match="simulated export update failure"):
                export_files(
                    failing_connection,
                    file_ids=[ids_by_name["alpha.jpg"]],
                    destination=str(dest),
                    mode="move",
                    preview_cache_root=preview_dir,
                )

        assert source_path.exists(), "Source file should be restored when the cache update fails"
        assert not target_path.exists(), "Destination file should be removed when rollback restores the source"
        assert preview_path.exists(), "Preview should not be deleted until the database update succeeds"

        with database(db_path) as connection:
            row = connection.execute(
                "SELECT path, preview_path FROM files WHERE id = ?",
                (ids_by_name["alpha.jpg"],),
            ).fetchone()

        assert row["path"] == original_db_path
        assert row["preview_path"] == str(preview_path.resolve())

    def test_move_keeps_earlier_rows_consistent_when_later_database_update_fails(self, tmp_path: Path):
        db_path, photo_dir, ids_by_name = setup_library(tmp_path)
        preview_dir = tmp_path / "previews"
        dest = tmp_path / "export"
        dest.mkdir()

        class FailingSecondUpdateConnection:
            def __init__(self, inner_connection):
                self._inner = inner_connection
                self._update_count = 0

            def execute(self, sql: str, params=()):
                if sql.startswith("UPDATE files SET path = ?, path_key = ?, preview_path = NULL, preview_status = 'missing'"):
                    self._update_count += 1
                    if self._update_count == 2:
                        raise sqlite3.OperationalError("simulated second export update failure")
                return self._inner.execute(sql, params)

            def __getattr__(self, name: str):
                return getattr(self._inner, name)

        # export_files processes rows ORDER BY id, so the file with the lower
        # database ID is the "first" row (succeeds) and the higher ID is the
        # "second" row (fails).  On Linux the scan insertion order is non-
        # deterministic, so we must resolve the mapping at runtime.
        ordered_ids = sorted(ids_by_name.items(), key=lambda kv: kv[1])
        first_name, _ = ordered_ids[0]
        second_name, _ = ordered_ids[1]

        first_source = photo_dir / first_name
        second_source = photo_dir / second_name
        first_target = dest / first_name
        second_target = dest / second_name

        with database(db_path) as connection:
            preview_rows = connection.execute(
                "SELECT id, path, preview_path FROM files WHERE id IN (?, ?) ORDER BY id",
                (ids_by_name[first_name], ids_by_name[second_name]),
            ).fetchall()
            original_paths = {Path(row["path"]).name: row["path"] for row in preview_rows}
            preview_paths = {Path(row["path"]).name: Path(row["preview_path"]) for row in preview_rows}
            assert preview_paths[first_name].exists()
            assert preview_paths[second_name].exists()

        with pytest.raises(sqlite3.OperationalError, match="simulated second export update failure"):
            with database(db_path) as connection:
                failing_connection = FailingSecondUpdateConnection(connection)
                export_files(
                    failing_connection,
                    file_ids=[ids_by_name[first_name], ids_by_name[second_name]],
                    destination=str(dest),
                    mode="move",
                    preview_cache_root=preview_dir,
                )

        assert first_target.exists(), "Earlier successful rows should remain moved on disk"
        assert not first_source.exists()
        assert not preview_paths[first_name].exists(), "Earlier committed rows should clean up old previews"

        assert second_source.exists(), "The failed row should be restored to its original location"
        assert not second_target.exists()
        assert preview_paths[second_name].exists(), "The failed row should keep its preview when the DB update fails"

        with database(db_path) as connection:
            rows = connection.execute(
                "SELECT id, path, preview_path FROM files WHERE id IN (?, ?)",
                (ids_by_name[first_name], ids_by_name[second_name]),
            ).fetchall()
            rows_by_name = {Path(row["path"]).name: row for row in rows}

        assert str(first_target.resolve()) == rows_by_name[first_name]["path"]
        assert rows_by_name[first_name]["preview_path"] is None
        assert rows_by_name[second_name]["path"] == original_paths[second_name]
        assert rows_by_name[second_name]["preview_path"] == str(preview_paths[second_name].resolve())

    def test_move_raises_if_rollback_restore_also_fails(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        db_path, photo_dir, ids_by_name = setup_library(tmp_path)
        preview_dir = tmp_path / "previews"
        dest = tmp_path / "export"
        dest.mkdir()

        class FailingUpdateConnection:
            def __init__(self, inner_connection):
                self._inner = inner_connection

            def execute(self, sql: str, params=()):
                if sql.startswith("UPDATE files SET path = ?, path_key = ?, preview_path = NULL, preview_status = 'missing'"):
                    raise sqlite3.OperationalError("simulated export update failure")
                return self._inner.execute(sql, params)

            def __getattr__(self, name: str):
                return getattr(self._inner, name)

        monkeypatch.setattr(
            "shotsieve.export._restore_moved_source",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(PermissionError("simulated rollback restore failure")),
        )

        with database(db_path) as connection:
            failing_connection = FailingUpdateConnection(connection)

            with pytest.raises(PermissionError, match="simulated rollback restore failure"):
                export_files(
                    failing_connection,
                    file_ids=[ids_by_name["alpha.jpg"]],
                    destination=str(dest),
                    mode="move",
                    preview_cache_root=preview_dir,
                )

