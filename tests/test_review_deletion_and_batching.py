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