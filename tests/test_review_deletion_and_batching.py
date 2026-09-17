import errno
from pathlib import Path
import sqlite3
from typing import cast

import pytest
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFilter

from shotsieve.db import connect, initialize_database, normalize_resolved_path
from shotsieve.learned_iqa import LearnedScoreResult
from shotsieve.review import (
    delete_files,
    update_review_state_batch,
)
from shotsieve.scanner import scan_root
from shotsieve.scoring import score_files


def _dict_value(value: object) -> dict[str, object]:
    return cast(dict[str, object], value)


def _path_text(item: dict[str, object]) -> str:
    return str(item["path"])


def _failed_error(result: dict[str, object]) -> str:
    failed = cast(list[dict[str, object]], result["failed"])
    return str(failed[0]["error"])


class FakeLearnedBackend:
    name = "topiq_nr"
    model_version = "fake:topiq_nr"

    def score_paths(self, image_paths, *, batch_size: int = 4, resource_profile: str | None = None):
        return [LearnedScoreResult(raw_score=0.82, normalized_score=82.0, confidence=91.0) for _ in image_paths]


def score_with_fake_learned_backend(connection) -> None:
    score_files(connection, learned_backend_factory=lambda model_name: FakeLearnedBackend())


def test_batch_review_state_updates(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "previews"
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    create_image(photo_dir / "a.jpg")
    create_image(photo_dir / "b.jpg")

    initialize_database(db_path)

    with connect(db_path) as connection:
        scan_root(connection, root=photo_dir, recursive=True, extensions=(".jpg",), preview_dir=preview_dir)
        score_with_fake_learned_backend(connection)
        file_ids = [row["id"] for row in connection.execute("SELECT id FROM files ORDER BY id").fetchall()]

        updated = update_review_state_batch(
            connection,
            file_ids=file_ids,
            decision_state="export",
            delete_marked=False,
            export_marked=True,
            updated_time="2026-03-24T00:00:00+00:00",
        )
        rows = connection.execute(
            "SELECT decision_state, export_marked FROM review_state ORDER BY file_id"
        ).fetchall()

    assert updated == 2
    assert all(row["decision_state"] == "export" for row in rows)
    assert all(row["export_marked"] == 1 for row in rows)


def test_batch_review_state_updates_avoids_per_file_lookup_queries(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "previews"
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    for name in ("a.jpg", "b.jpg", "c.jpg", "d.jpg"):
        create_image(photo_dir / name)

    initialize_database(db_path)

    with connect(db_path) as connection:
        scan_root(connection, root=photo_dir, recursive=True, extensions=(".jpg",), preview_dir=preview_dir)
        file_ids = [row["id"] for row in connection.execute("SELECT id FROM files ORDER BY id").fetchall()]

        traced_sql: list[str] = []
        connection.set_trace_callback(traced_sql.append)
        try:
            updated = update_review_state_batch(
                connection,
                file_ids=file_ids,
                decision_state="export",
                delete_marked=False,
                export_marked=True,
                updated_time="2026-03-24T00:00:00+00:00",
            )
        finally:
            connection.set_trace_callback(None)

    per_file_review_selects = [
        sql for sql in traced_sql
        if "FROM review_state WHERE file_id =" in sql
    ]
    per_file_file_selects = [
        sql for sql in traced_sql
        if "FROM files WHERE id =" in sql
    ]

    assert updated == 4
    assert per_file_review_selects == []
    assert per_file_file_selects == []


def test_review_state_table_rejects_conflicting_flags_at_db_level(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "previews"
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    create_image(photo_dir / "sample.jpg")

    initialize_database(db_path)

    with connect(db_path) as connection:
        scan_root(connection, root=photo_dir, recursive=True, extensions=(".jpg",), preview_dir=preview_dir)
        file_id = connection.execute("SELECT id FROM files LIMIT 1").fetchone()["id"]

        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO review_state(file_id, decision_state, delete_marked, export_marked, updated_time)
                VALUES(?, 'pending', 1, 1, ?)
                """,
                (file_id, "2026-03-24T00:00:00+00:00"),
            )


def test_delete_files_removes_source_and_cache(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "previews"
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    source_path = photo_dir / "sample.jpg"
    create_image(source_path)

    initialize_database(db_path)

    with connect(db_path) as connection:
        scan_root(connection, root=photo_dir, recursive=True, extensions=(".jpg",), preview_dir=preview_dir)
        row = connection.execute("SELECT id, preview_path FROM files LIMIT 1").fetchone()
        file_id = row["id"]
        preview_path = Path(row["preview_path"])
        assert preview_path.exists()

        result = delete_files(
            connection,
            file_ids=[file_id],
            delete_from_disk=True,
            preview_cache_root=preview_dir,
        )
        count = connection.execute("SELECT COUNT(*) AS count FROM files").fetchone()["count"]

    assert result["deleted_count"] == 1
    assert result["failed_count"] == 0
    assert not source_path.exists()
    assert not preview_path.exists()
    assert count == 0


def test_delete_post_mutation_failure_is_uncertain_and_not_retry_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from shotsieve import review_cache as review_cache_module

    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "previews"
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    source_path = photo_dir / "sample.jpg"
    create_image(source_path)
    initialize_database(db_path)

    with connect(db_path) as connection:
        scan_root(connection, root=photo_dir, recursive=True, extensions=(".jpg",), preview_dir=preview_dir)
        file_id = connection.execute("SELECT id FROM files LIMIT 1").fetchone()["id"]

    real_unlink = Path.unlink

    def unlink_then_fail(path: Path, *args, **kwargs):
        if path == source_path:
            real_unlink(path, *args, **kwargs)
            raise OSError(errno.EIO, "simulated post-mutation delete failure")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(review_cache_module.Path, "unlink", unlink_then_fail)

    with connect(db_path) as connection:
        result = delete_files(
            connection,
            file_ids=[file_id],
            delete_from_disk=True,
            preview_cache_root=preview_dir,
        )

    item = result["items"][0]
    assert item["outcome"] == "uncertain"
    assert item["retry_safe"] is False
    assert item["source_state"] == "missing"
    assert not source_path.exists()


def test_delete_catalog_failure_after_source_removal_retains_uncertain_result(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "previews"
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    source_path = photo_dir / "sample.jpg"
    create_image(source_path)
    initialize_database(db_path)

    class FailingDeleteConnection:
        def __init__(self, inner_connection):
            self._inner = inner_connection

        def execute(self, sql: str, params=()):
            if sql.startswith("DELETE FROM files WHERE id = ?"):
                raise sqlite3.OperationalError("simulated delete catalog failure")
            return self._inner.execute(sql, params)

        def __getattr__(self, name: str):
            return getattr(self._inner, name)

    with connect(db_path) as connection:
        scan_root(connection, root=photo_dir, recursive=True, extensions=(".jpg",), preview_dir=preview_dir)
        file_id = connection.execute("SELECT id FROM files LIMIT 1").fetchone()["id"]
        with pytest.raises(sqlite3.OperationalError, match="simulated delete catalog failure") as exc_info:
            delete_files(
                FailingDeleteConnection(connection),
                file_ids=[file_id],
                delete_from_disk=True,
                preview_cache_root=preview_dir,
            )

    item = exc_info.value.file_operation_summary["items"][0]
    assert item["outcome"] == "uncertain"
    assert item["retry_safe"] is False
    assert item["source_state"] == "missing"
    assert not source_path.exists()


def test_delete_catalog_failure_retains_all_later_ids_as_unprocessed(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "previews"
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    for name in ("first.jpg", "second.jpg", "third.jpg"):
        create_image(photo_dir / name)
    initialize_database(db_path)

    class FailingSecondDeleteConnection:
        def __init__(self, inner_connection):
            self._inner = inner_connection
            self._delete_count = 0

        def execute(self, sql: str, params=()):
            if sql.startswith("DELETE FROM files WHERE id = ?"):
                self._delete_count += 1
                if self._delete_count == 2:
                    raise sqlite3.OperationalError("simulated second delete catalog failure")
            return self._inner.execute(sql, params)

        def __getattr__(self, name: str):
            return getattr(self._inner, name)

    with connect(db_path) as connection:
        scan_root(connection, root=photo_dir, recursive=True, extensions=(".jpg",), preview_dir=preview_dir)
        file_ids = [row["id"] for row in connection.execute("SELECT id FROM files ORDER BY id").fetchall()]
        with pytest.raises(sqlite3.OperationalError, match="simulated second delete catalog failure") as exc_info:
            delete_files(
                FailingSecondDeleteConnection(connection),
                file_ids=file_ids,
                delete_from_disk=True,
                preview_cache_root=preview_dir,
            )

    summary = exc_info.value.file_operation_summary
    assert {item["file_id"] for item in summary["items"]} == set(file_ids)
    assert summary["completed_count"] == 1
    assert summary["partial_count"] == 1
    assert summary["unprocessed_count"] == 1
    assert summary["safe_retry_ids"] == [file_ids[2]]


def test_delete_files_rejects_disk_delete_outside_scanned_roots(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "previews"
    photo_dir = tmp_path / "photos"
    sibling_dir = tmp_path / "photos-archive"
    photo_dir.mkdir()
    sibling_dir.mkdir()
    create_image(photo_dir / "sample.jpg")
    escaped_source = sibling_dir / "escaped.jpg"
    create_image(escaped_source)

    initialize_database(db_path)

    with connect(db_path) as connection:
        scan_root(connection, root=photo_dir, recursive=True, extensions=(".jpg",), preview_dir=preview_dir)
        row = connection.execute("SELECT id, preview_path FROM files LIMIT 1").fetchone()
        file_id = row["id"]
        preview_path = Path(row["preview_path"])
        assert preview_path.exists()

        connection.execute(
            "UPDATE files SET path = ?, path_key = ? WHERE id = ?",
            (
                str(escaped_source.resolve()),
                normalize_resolved_path(escaped_source.resolve()),
                file_id,
            ),
        )

        result = delete_files(
            connection,
            file_ids=[file_id],
            delete_from_disk=True,
            preview_cache_root=preview_dir,
        )
        count = connection.execute("SELECT COUNT(*) AS count FROM files").fetchone()["count"]

    assert result["deleted_count"] == 0
    assert result["failed_count"] == 1
    assert "outside tracked scan roots" in _failed_error(result)
    assert escaped_source.exists()
    assert preview_path.exists()
    assert count == 1


def test_delete_files_rejects_disk_delete_when_path_key_identity_mismatches(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "previews"
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    original_source = photo_dir / "sample.jpg"
    alternate_source = photo_dir / "alternate.jpg"
    create_image(original_source)
    create_image(alternate_source)

    initialize_database(db_path)

    with connect(db_path) as connection:
        scan_root(connection, root=photo_dir, recursive=True, extensions=(".jpg",), preview_dir=preview_dir)
        row = connection.execute("SELECT id, preview_path FROM files WHERE path LIKE ? LIMIT 1", ("%sample.jpg",)).fetchone()
        file_id = row["id"]
        preview_path = Path(row["preview_path"])
        assert preview_path.exists()

        connection.execute(
            "UPDATE files SET path = ? WHERE id = ?",
            (str(alternate_source.resolve()), file_id),
        )

        result = delete_files(
            connection,
            file_ids=[file_id],
            delete_from_disk=True,
            preview_cache_root=preview_dir,
        )
        count = connection.execute("SELECT COUNT(*) AS count FROM files").fetchone()["count"]

    assert result["deleted_count"] == 0
    assert result["failed_count"] == 1
    assert "path key" in _failed_error(result)
    assert original_source.exists()
    assert alternate_source.exists()
    assert preview_path.exists()
    assert count == 2


def test_delete_files_preserves_preview_outside_configured_root(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    custom_preview_dir = tmp_path / "custom-previews"
    default_preview_dir = db_path.parent / "previews"
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    source_path = photo_dir / "sample.jpg"
    create_image(source_path)

    initialize_database(db_path)

    with connect(db_path) as connection:
        scan_root(
            connection,
            root=photo_dir,
            recursive=True,
            extensions=(".jpg",),
            preview_dir=custom_preview_dir,
        )
        row = connection.execute("SELECT id, preview_path FROM files LIMIT 1").fetchone()
        file_id = row["id"]
        preview_path = Path(row["preview_path"])
        assert preview_path.exists()
        assert preview_path.is_relative_to(custom_preview_dir.resolve())

        result = delete_files(
            connection,
            file_ids=[file_id],
            delete_from_disk=True,
            preview_cache_root=default_preview_dir,
        )

    assert result["deleted_count"] == 1
    assert preview_path.exists()


def test_delete_files_preserves_non_preview_sidecar_inside_root(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "previews"
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    source_path = photo_dir / "sample.jpg"
    create_image(source_path)

    initialize_database(db_path)

    with connect(db_path) as connection:
        scan_root(connection, root=photo_dir, recursive=True, extensions=(".jpg",), preview_dir=preview_dir)
        row = connection.execute("SELECT id FROM files LIMIT 1").fetchone()
        file_id = row["id"]
        sidecar_path = preview_dir / "keep-me.txt"
        sidecar_path.write_text("keep", encoding="utf-8")
        connection.execute(
            "UPDATE files SET preview_path = ? WHERE id = ?",
            (str(sidecar_path.resolve()), file_id),
        )

        result = delete_files(
            connection,
            file_ids=[file_id],
            delete_from_disk=True,
            preview_cache_root=preview_dir,
        )

    assert result["deleted_count"] == 1
    assert sidecar_path.exists()


def test_delete_cancellation_commits_completed_rows_and_retains_unprocessed_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    create_image(photo_dir / "first.jpg")
    create_image(photo_dir / "second.jpg")
    initialize_database(db_path)

    with connect(db_path) as connection:
        scan_root(connection, root=photo_dir, recursive=True, extensions=(".jpg",), preview_dir=tmp_path / "previews")
        file_ids = [row["id"] for row in connection.execute("SELECT id FROM files ORDER BY id").fetchall()]
        requested_ids = [file_ids[1], file_ids[0]]
        calls = 0

        def cancel_after_first_file() -> None:
            nonlocal calls
            calls += 1
            if calls > 1:
                raise InterruptedError("simulated cancellation")

        with pytest.raises(InterruptedError) as exc_info:
            delete_files(
                connection,
                file_ids=requested_ids,
                delete_from_disk=False,
                cancel_check=cancel_after_first_file,
            )

    summary = exc_info.value.file_operation_summary
    assert summary["outcome"] == "cancelled"
    assert summary["completed_count"] == 1
    assert summary["unprocessed_count"] == 1
    assert summary["safe_retry_ids"] == [requested_ids[1]]

    with connect(db_path) as connection:
        remaining_ids = [row["id"] for row in connection.execute("SELECT id FROM files ORDER BY id").fetchall()]
    assert remaining_ids == [requested_ids[1]]






def create_image(path: Path) -> None:
    image = Image.new("RGB", (120, 80), color=(40, 90, 160))
    image.save(path, format="JPEG")


def create_tiff_image(path: Path) -> None:
    image = Image.new("RGB", (128, 96), color=(60, 110, 170))
    image.save(path, format="TIFF")


def create_pattern_image(path: Path, *, blur_radius: int) -> None:
    image = Image.new("RGB", (240, 160), color=(245, 245, 245))
    draw = ImageDraw.Draw(image)

    for x in range(0, 240, 24):
        draw.rectangle((x, 0, x + 11, 159), fill=(20, 30, 40))

    for y in range(0, 160, 24):
        draw.line((0, y, 239, y), fill=(220, 60, 60), width=3)

    if blur_radius:
        image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    image.save(path, format="JPEG", quality=92)
