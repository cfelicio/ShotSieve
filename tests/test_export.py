"""Tests for the export module - copy, move, collision handling, and validation."""
from __future__ import annotations

import errno
from pathlib import Path
import sqlite3
from typing import cast

import pytest
from PIL import Image

from shotsieve.db import database, initialize_database
from shotsieve.export import _reject_system_directory, export_files
from shotsieve.scanner import scan_root


def test_move_does_not_require_hard_links(tmp_path, monkeypatch):
    from shotsieve.export import _move_without_overwrite

    source, target = tmp_path / "source.jpg", tmp_path / "target.jpg"
    source.write_bytes(b"original photo")

    def unsupported(*args):
        raise PermissionError("hard links unavailable")

    monkeypatch.setattr("shotsieve.export.os.link", unsupported)
    _move_without_overwrite(source, target)
    assert target.read_bytes() == b"original photo"
    assert not source.exists()


def test_move_recovery_never_overwrites_a_racing_source(tmp_path, monkeypatch):
    from shotsieve import export

    source, target = tmp_path / "source.jpg", tmp_path / "target.jpg"
    target.write_bytes(b"moved photo")
    real_open = export.os.open

    def racing_open(path, flags, *args):
        source.write_bytes(b"new photo")
        return real_open(path, flags, *args)

    monkeypatch.setattr(export.os, "open", racing_open)
    with pytest.raises(FileExistsError):
        export._restore_moved_source(source, target)
    assert source.read_bytes() == b"new photo"
    assert target.read_bytes() == b"moved photo"


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

    def test_inaccessible_source_observation_is_not_retry_safe(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        from shotsieve import export as export_module
        from shotsieve.models import FilesystemObservation

        db_path, photo_dir, ids_by_name = setup_library(tmp_path)
        destination = tmp_path / "export"
        destination.mkdir()
        source = photo_dir / "alpha.jpg"
        real_observe = export_module.observe_filesystem_path

        def observe(path: Path):
            if path == source:
                return FilesystemObservation("unknown", "simulated access denied")
            return real_observe(path)

        monkeypatch.setattr(export_module, "observe_filesystem_path", observe)

        with database(db_path) as connection:
            result = export_files(
                connection,
                file_ids=[ids_by_name["alpha.jpg"]],
                destination=str(destination),
                mode="copy",
            )

        item = result.items[0]
        assert item.outcome == "failed"
        assert item.source_state == "unknown"
        assert item.retry_safe is False
        assert "simulated access denied" in item.observation_errors[0]

    def test_post_mutation_move_failure_is_uncertain_and_retains_both_paths(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        from shotsieve import export as export_module

        db_path, photo_dir, ids_by_name = setup_library(tmp_path)
        destination = tmp_path / "export"
        destination.mkdir()
        source = photo_dir / "alpha.jpg"
        target = destination / "alpha.jpg"
        original_bytes = source.read_bytes()

        def move_then_fail(source_path: Path, target_path: Path) -> None:
            target_path.write_bytes(source_path.read_bytes())
            source_path.unlink()
            raise OSError(errno.EIO, "simulated post-mutation failure")

        monkeypatch.setattr(export_module, "_move_without_overwrite", move_then_fail)

        with database(db_path) as connection:
            result = export_files(
                connection,
                file_ids=[ids_by_name["alpha.jpg"]],
                destination=str(destination),
                mode="move",
            )

        item = result.items[0]
        assert item.outcome == "uncertain"
        assert item.retry_safe is False
        assert item.source == str(source.resolve())
        assert item.destination == str(target.resolve())
        assert item.source_state == "missing"
        assert item.destination_state == "present"
        assert target.read_bytes() == original_bytes

    def test_copy_cleanup_failure_is_uncertain_and_retains_destination(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        from shotsieve import export as export_module

        db_path, _photo_dir, ids_by_name = setup_library(tmp_path)
        destination = tmp_path / "export"
        destination.mkdir()
        target = destination / "alpha.jpg"
        real_copyfileobj = export_module.shutil.copyfileobj
        real_unlink = Path.unlink

        def copy_then_fail(source_stream, destination_stream, *args, **kwargs):
            real_copyfileobj(source_stream, destination_stream, *args, **kwargs)
            raise OSError(errno.EIO, "simulated copy failure")

        def cleanup_then_fail(path: Path, *args, **kwargs):
            if path == target:
                raise PermissionError("simulated cleanup denial")
            return real_unlink(path, *args, **kwargs)

        monkeypatch.setattr(export_module.shutil, "copyfileobj", copy_then_fail)
        monkeypatch.setattr(Path, "unlink", cleanup_then_fail)

        with database(db_path) as connection:
            result = export_files(
                connection,
                file_ids=[ids_by_name["alpha.jpg"]],
                destination=str(destination),
                mode="copy",
            )

        item = result.items[0]
        assert item.outcome == "uncertain"
        assert item.retry_safe is False
        assert item.destination_state == "present"
        assert "cleanup failed" in (item.error_text or "")
        assert target.exists()

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
    def test_move_succeeds_when_remote_metadata_copy_is_unsupported(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        from shotsieve import export as export_module

        db_path, photo_dir, ids_by_name = setup_library(tmp_path)
        destination = tmp_path / "export"
        destination.mkdir()

        def fail_metadata_copy(*_args, **_kwargs):
            raise OSError("simulated remote metadata limitation")

        monkeypatch.setattr(export_module.shutil, "copystat", fail_metadata_copy)

        with database(db_path) as connection:
            result = export_files(
                connection,
                file_ids=[ids_by_name["alpha.jpg"]],
                destination=str(destination),
                mode="move",
            )

        assert result.moved == 1
        assert not (photo_dir / "alpha.jpg").exists()
        assert (destination / "alpha.jpg").exists()
        assert result.warnings[0]["stage"] == "transfer_metadata"

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

    def test_move_restores_source_when_database_commit_fails(self, tmp_path: Path):
        db_path, photo_dir, ids_by_name = setup_library(tmp_path)
        destination = tmp_path / "export"
        destination.mkdir()

        class FailingCommitConnection:
            def __init__(self, inner_connection):
                self._inner = inner_connection

            def commit(self):
                raise sqlite3.OperationalError("simulated export commit failure")

            def __getattr__(self, name: str):
                return getattr(self._inner, name)

        source = photo_dir / "alpha.jpg"
        target = destination / "alpha.jpg"

        with database(db_path) as connection:
            with pytest.raises(sqlite3.OperationalError, match="simulated export commit failure") as exc_info:
                export_files(
                    FailingCommitConnection(connection),
                    file_ids=[ids_by_name["alpha.jpg"]],
                    destination=str(destination),
                    mode="move",
                )

        item = exc_info.value.file_operation_summary["items"][0]
        assert item["outcome"] == "failed"
        assert item["retry_safe"] is True
        assert source.exists()
        assert not target.exists()

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

        # export_files preserves caller order, so pass the rows in database-ID
        # order here to make the simulated second update failure deterministic.
        # On Linux the scan insertion order is non-deterministic, so resolve
        # the mapping at runtime.
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

        with pytest.raises(sqlite3.OperationalError, match="simulated second export update failure") as exc_info:
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

        partial_summary = exc_info.value.file_operation_summary
        assert partial_summary["completed_count"] == 1
        assert partial_summary["failed_count"] == 1
        assert partial_summary["outcome"] == "partial"

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

    def test_cleanup_warning_is_not_counted_as_a_failed_photo(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        db_path, _photo_dir, ids_by_name = setup_library(tmp_path)
        dest = tmp_path / "export"
        dest.mkdir()

        def fail_cleanup(*_args, **_kwargs):
            raise OSError("simulated preview cleanup failure")

        monkeypatch.setattr("shotsieve.export.delete_managed_preview_file", fail_cleanup)

        with database(db_path) as connection:
            result = export_files(
                connection,
                file_ids=[ids_by_name["alpha.jpg"]],
                destination=str(dest),
                mode="move",
                preview_cache_root=tmp_path / "previews",
            )

        assert result.outcome == "success"
        assert result.moved == 1
        assert result.failed == []
        assert result.warnings[0]["stage"] == "preview_cleanup"

    def test_cancellation_retains_completed_and_unprocessed_rows(self, tmp_path: Path):
        db_path, _photo_dir, ids_by_name = setup_library(tmp_path)
        dest = tmp_path / "export"
        dest.mkdir()
        calls = 0

        def cancel_after_first_file() -> None:
            nonlocal calls
            calls += 1
            if calls > 1:
                raise InterruptedError("simulated cancellation")

        with database(db_path) as connection:
            with pytest.raises(InterruptedError) as exc_info:
                export_files(
                    connection,
                    file_ids=[ids_by_name["alpha.jpg"], ids_by_name["beta.jpg"]],
                    destination=str(dest),
                    mode="copy",
                    cancel_check=cancel_after_first_file,
                )

        summary = exc_info.value.file_operation_summary
        assert summary["outcome"] == "cancelled"
        assert summary["completed_count"] == 1
        assert summary["unprocessed_count"] == 1
        assert summary["safe_retry_ids"] == [ids_by_name["beta.jpg"]]
        assert (dest / "alpha.jpg").exists()

