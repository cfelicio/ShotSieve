import platform
from pathlib import Path
import sqlite3
from typing import cast

import pytest
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFilter

from shotsieve.db import connect, get_preview_cache_root, initialize_database, normalize_resolved_path, resolve_preview_cache_root
from shotsieve.learned_iqa import LearnedScoreResult
from shotsieve.review import (
    clear_cache_scope,
    count_review_files,
    delete_files,
    get_review_file_detail,
    list_review_files,
    list_review_state_file_ids,
    media_path_for_file,
    prune_missing_cache_entries,
    remove_files_from_cache,
    review_overview,
    update_review_state,
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


def test_review_listing_and_state_updates(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "previews"
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    create_image(photo_dir / "a.jpg")
    create_image(photo_dir / "b.jpg")

    initialize_database(db_path)

    with connect(db_path) as connection:
        scan_root(
            connection,
            root=photo_dir,
            recursive=True,
            extensions=(".jpg",),
            preview_dir=preview_dir,
        )
        score_with_fake_learned_backend(connection)

        first_file_id = connection.execute("SELECT id FROM files ORDER BY id ASC LIMIT 1").fetchone()["id"]
        update_review_state(
            connection,
            file_id=first_file_id,
            decision_state="delete",
            delete_marked=True,
            export_marked=False,
            updated_time="2026-03-24T00:00:00+00:00",
        )

        delete_items = list_review_files(connection, marked="delete")
        unmarked_items = list_review_files(connection, marked="none")
        detail = get_review_file_detail(connection, first_file_id)
        overview = review_overview(connection)

    assert len(delete_items) == 1
    assert len(unmarked_items) == 1
    assert detail is not None
    assert detail["decision_state"] == "delete"
    summary = _dict_value(overview["summary"])
    assert summary["delete_marked"] == 1
    assert summary["scored_files"] == 2


def test_review_state_queries_do_not_require_scores(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "previews"
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    create_image(photo_dir / "a.jpg")
    create_image(photo_dir / "b.jpg")

    initialize_database(db_path)

    with connect(db_path) as connection:
        scan_root(
            connection,
            root=photo_dir,
            recursive=True,
            extensions=(".jpg",),
            preview_dir=preview_dir,
        )
        rows = connection.execute("SELECT id FROM files ORDER BY id ASC").fetchall()
        delete_id = int(rows[0]["id"])
        export_id = int(rows[1]["id"])

        update_review_state(
            connection,
            file_id=delete_id,
            decision_state="delete",
            delete_marked=True,
            export_marked=False,
            updated_time="2026-04-20T00:00:00+00:00",
        )
        update_review_state(
            connection,
            file_id=export_id,
            decision_state="export",
            delete_marked=False,
            export_marked=True,
            updated_time="2026-04-20T00:01:00+00:00",
        )

        assert list_review_files(connection) == []
        assert count_review_files(connection) == 0

        assert list_review_state_file_ids(connection, marked="delete") == [delete_id]
        assert list_review_state_file_ids(connection, marked="export") == [export_id]


def test_review_state_file_ids_support_pagination(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "previews"
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        create_image(photo_dir / name)

    initialize_database(db_path)

    with connect(db_path) as connection:
        scan_root(
            connection,
            root=photo_dir,
            recursive=True,
            extensions=(".jpg",),
            preview_dir=preview_dir,
        )
        file_ids = [int(row["id"]) for row in connection.execute("SELECT id FROM files ORDER BY id ASC").fetchall()]
        for file_id in file_ids:
            update_review_state(
                connection,
                file_id=file_id,
                decision_state="delete",
                delete_marked=True,
                export_marked=False,
                updated_time=f"2026-04-20T00:00:0{file_id}+00:00",
            )

        first_page = list_review_state_file_ids(connection, marked="delete", limit=2, offset=0)
        second_page = list_review_state_file_ids(connection, marked="delete", limit=2, offset=2)

    assert first_page == file_ids[:2]
    assert second_page == file_ids[2:]


def test_update_review_state_rejects_invalid_inputs(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "previews"
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    create_image(photo_dir / "sample.jpg")

    initialize_database(db_path)

    with connect(db_path) as connection:
        scan_root(
            connection,
            root=photo_dir,
            recursive=True,
            extensions=(".jpg",),
            preview_dir=preview_dir,
        )
        file_id = connection.execute("SELECT id FROM files LIMIT 1").fetchone()["id"]

        with pytest.raises(ValueError, match="cannot both be true"):
            update_review_state(
                connection,
                file_id=file_id,
                decision_state="delete",
                delete_marked=True,
                export_marked=True,
                updated_time="2026-03-24T00:00:00+00:00",
            )

        with pytest.raises(ValueError, match="decision_state"):
            update_review_state(
                connection,
                file_id=file_id,
                decision_state="archive",
                delete_marked=False,
                export_marked=False,
                updated_time="2026-03-24T00:00:00+00:00",
            )

        with pytest.raises(ValueError, match="does not exist"):
            update_review_state(
                connection,
                file_id=file_id + 999,
                decision_state="pending",
                delete_marked=False,
                export_marked=False,
                updated_time="2026-03-24T00:00:00+00:00",
            )


def test_update_review_state_rejects_conflicting_merged_flags(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "previews"
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    create_image(photo_dir / "sample.jpg")

    initialize_database(db_path)

    with connect(db_path) as connection:
        scan_root(
            connection,
            root=photo_dir,
            recursive=True,
            extensions=(".jpg",),
            preview_dir=preview_dir,
        )
        file_id = connection.execute("SELECT id FROM files LIMIT 1").fetchone()["id"]

        update_review_state(
            connection,
            file_id=file_id,
            decision_state="delete",
            delete_marked=True,
            export_marked=False,
            updated_time="2026-03-24T00:00:00+00:00",
        )

        with pytest.raises(ValueError, match="cannot both be true"):
            update_review_state(
                connection,
                file_id=file_id,
                decision_state=None,
                delete_marked=None,
                export_marked=True,
                updated_time="2026-03-24T01:00:00+00:00",
            )


def test_review_listing_filters_by_score_band_and_root(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "previews"
    root_a = tmp_path / "set-a"
    root_b = tmp_path / "set-b"
    root_a.mkdir()
    root_b.mkdir()
    create_pattern_image(root_a / "sharp.jpg", blur_radius=0)
    create_pattern_image(root_b / "soft.jpg", blur_radius=5)

    initialize_database(db_path)

    with connect(db_path) as connection:
        scan_root(
            connection,
            root=root_a,
            recursive=True,
            extensions=(".jpg",),
            preview_dir=preview_dir,
        )
        scan_root(
            connection,
            root=root_b,
            recursive=True,
            extensions=(".jpg",),
            preview_dir=preview_dir,
        )
        score_with_fake_learned_backend(connection)

        connection.execute(
            """
            UPDATE scores
               SET overall_score = 80.0
             WHERE file_id = (SELECT id FROM files WHERE path LIKE ? LIMIT 1)
            """,
            ("%sharp.jpg",),
        )
        connection.execute(
            """
            UPDATE scores
               SET overall_score = 40.0
             WHERE file_id = (SELECT id FROM files WHERE path LIKE ? LIMIT 1)
            """,
            ("%soft.jpg",),
        )

        low_score_items = list_review_files(connection, max_score=60, sort="score_asc")
        root_a_items = list_review_files(connection, root=str(root_a))

    assert len(low_score_items) == 1
    assert _path_text(low_score_items[0]).endswith("soft.jpg")
    assert len(root_a_items) == 1
    assert _path_text(root_a_items[0]).endswith("sharp.jpg")


def test_review_listing_filters_by_query(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "previews"
    first_root = tmp_path / "set-a"
    second_root = tmp_path / "set-b"
    first_root.mkdir()
    second_root.mkdir()
    create_image(first_root / "keep-me.jpg")
    create_image(second_root / "discard-me.jpg")

    initialize_database(db_path)

    with connect(db_path) as connection:
        scan_root(connection, root=first_root, recursive=True, extensions=(".jpg",), preview_dir=preview_dir)
        scan_root(connection, root=second_root, recursive=True, extensions=(".jpg",), preview_dir=preview_dir)
        score_with_fake_learned_backend(connection)

        items = list_review_files(connection, query="discard")

    assert len(items) == 1
    assert _path_text(items[0]).endswith("discard-me.jpg")


def test_review_listing_filters_to_files_with_issues_only(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "previews"
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    create_image(photo_dir / "clean.jpg")
    create_image(photo_dir / "problem.jpg")

    initialize_database(db_path)

    with connect(db_path) as connection:
        scan_root(connection, root=photo_dir, recursive=True, extensions=(".jpg",), preview_dir=preview_dir)
        score_with_fake_learned_backend(connection)
        connection.execute(
            """
            UPDATE files
               SET last_error = ?
             WHERE path LIKE ?
            """,
            ("data corruption detected", "%problem.jpg"),
        )

        issue_items = list_review_files(connection, issues="issues")
        issue_total = count_review_files(connection, issues="issues")

    assert issue_total == 1
    assert len(issue_items) == 1
    assert _path_text(issue_items[0]).endswith("problem.jpg")


def test_review_listing_filters_by_query_with_unicode_casefold(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "previews"
    photo_root = tmp_path / "Åland"
    photo_root.mkdir()
    create_image(photo_root / "harbor.jpg")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(platform, "system", lambda: "Linux")

    try:
        initialize_database(db_path)

        with connect(db_path) as connection:
            scan_root(connection, root=photo_root, recursive=True, extensions=(".jpg",), preview_dir=preview_dir)
            score_with_fake_learned_backend(connection)

            items = list_review_files(connection, query="åland")
    finally:
        monkeypatch.undo()

    assert len(items) == 1
    assert _path_text(items[0]).endswith("harbor.jpg")


def test_review_listing_root_filter_excludes_sibling_prefixes(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "previews"
    root_main = tmp_path / "photos"
    root_sibling = tmp_path / "photos-archive"
    root_main.mkdir()
    root_sibling.mkdir()
    create_image(root_main / "keep.jpg")
    create_image(root_sibling / "exclude.jpg")

    initialize_database(db_path)

    with connect(db_path) as connection:
        scan_root(connection, root=root_main, recursive=True, extensions=(".jpg",), preview_dir=preview_dir)
        scan_root(connection, root=root_sibling, recursive=True, extensions=(".jpg",), preview_dir=preview_dir)
        score_with_fake_learned_backend(connection)

        filtered = list_review_files(connection, root=str(root_main))

    assert len(filtered) == 1
    assert _path_text(filtered[0]).endswith("keep.jpg")


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


def test_prune_missing_cache_entries_removes_managed_preview(tmp_path: Path) -> None:
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
        preview_path = Path(row["preview_path"])
        assert preview_path.exists()

        source_path.unlink()
        removed = prune_missing_cache_entries(connection, preview_cache_root=preview_dir)
        count = connection.execute("SELECT COUNT(*) AS count FROM files").fetchone()["count"]

    assert removed == 1
    assert not preview_path.exists()
    assert count == 0


def test_prune_missing_cache_entries_processes_rows_in_batches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "previews"
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()

    for name in ("a.jpg", "b.jpg", "c.jpg"):
        create_image(photo_dir / name)

    initialize_database(db_path)

    batch_sizes: list[int] = []

    class RecordingExecutor:
        def __init__(self, *, max_workers: int) -> None:
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def map(self, fn, items):
            captured_items = list(items)
            batch_sizes.append(len(captured_items))
            return [fn(item) for item in captured_items]

    monkeypatch.setattr("shotsieve.review._PRUNE_MISSING_CACHE_BATCH_SIZE", 2, raising=False)
    monkeypatch.setattr("shotsieve.review.ThreadPoolExecutor", RecordingExecutor)

    with connect(db_path) as connection:
        scan_root(connection, root=photo_dir, recursive=True, extensions=(".jpg",), preview_dir=preview_dir)

        missing_source = photo_dir / "b.jpg"
        missing_source.unlink()

        removed = prune_missing_cache_entries(connection, preview_cache_root=preview_dir)
        remaining_paths = [
            Path(row["path"]).name
            for row in connection.execute("SELECT path FROM files ORDER BY path ASC").fetchall()
        ]

    assert removed == 1
    assert batch_sizes == [2, 1]
    assert remaining_paths == ["a.jpg", "c.jpg"]


def test_remove_files_from_cache_removes_managed_preview(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "custom-previews"
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

        preview_cache_root = get_preview_cache_root(connection, db_path=db_path)
        removed = remove_files_from_cache(
            connection,
            file_ids=[file_id],
            preview_cache_root=preview_cache_root,
        )
        count = connection.execute("SELECT COUNT(*) AS count FROM files").fetchone()["count"]

    assert removed == 1
    assert source_path.exists()
    assert not preview_path.exists()
    assert count == 0


def test_clear_cache_scope_all_wipes_preview_cache(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "custom-previews"
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    create_image(photo_dir / "sample.jpg")

    initialize_database(db_path)

    with connect(db_path) as connection:
        scan_root(connection, root=photo_dir, recursive=True, extensions=(".jpg",), preview_dir=preview_dir)
        row = connection.execute("SELECT preview_path FROM files LIMIT 1").fetchone()
        preview_path = Path(row["preview_path"])
        assert preview_path.exists()

        preview_cache_root = get_preview_cache_root(connection, db_path=db_path)
        result = clear_cache_scope(connection, scope="all", preview_cache_root=preview_cache_root)
        count = connection.execute("SELECT COUNT(*) AS count FROM files").fetchone()["count"]

    assert result["files"] == 1
    assert not preview_path.exists()
    assert count == 0


def test_get_preview_cache_root_recovers_custom_preview_dir(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "custom-previews"
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    create_image(photo_dir / "sample.jpg")

    initialize_database(db_path)

    with connect(db_path) as connection:
        scan_root(connection, root=photo_dir, recursive=True, extensions=(".jpg",), preview_dir=preview_dir)
        recovered_preview_root = get_preview_cache_root(connection, db_path=db_path)

    assert recovered_preview_root == preview_dir.resolve()


def test_scan_without_preview_generation_keeps_existing_preview_root(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    first_preview_dir = tmp_path / "preview-a"
    second_preview_dir = tmp_path / "preview-b"
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    create_image(photo_dir / "sample.jpg")

    initialize_database(db_path)

    with connect(db_path) as connection:
        scan_root(
            connection,
            root=photo_dir,
            recursive=True,
            extensions=(".jpg",),
            preview_dir=first_preview_dir,
        )
        initial_preview_root = get_preview_cache_root(connection, db_path=db_path)

        scan_root(
            connection,
            root=photo_dir,
            recursive=True,
            extensions=(".jpg",),
            preview_dir=second_preview_dir,
            generate_previews=False,
        )
        preserved_preview_root = get_preview_cache_root(connection, db_path=db_path)

    assert initial_preview_root == first_preview_dir.resolve()
    assert preserved_preview_root == first_preview_dir.resolve()


def test_scan_excludes_stored_preview_root_when_requested_root_changes(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    first_preview_dir = tmp_path / "preview-a"
    second_preview_dir = tmp_path / "preview-b"
    photo_root = tmp_path / "library"
    photo_root.mkdir()
    create_image(photo_root / "sample.jpg")

    initialize_database(db_path)

    with connect(db_path) as connection:
        scan_root(
            connection,
            root=photo_root,
            recursive=True,
            extensions=(".jpg",),
            preview_dir=first_preview_dir,
        )

        preview_path = Path(
            connection.execute("SELECT preview_path FROM files LIMIT 1").fetchone()["preview_path"]
        )
        assert preview_path.exists()

        scan_root(
            connection,
            root=tmp_path,
            recursive=True,
            extensions=(".jpg",),
            preview_dir=second_preview_dir,
            generate_previews=False,
        )
        rows = connection.execute("SELECT path FROM files ORDER BY path ASC").fetchall()

    paths = [row["path"] for row in rows]
    assert str(preview_path.resolve()) not in paths


def test_scan_excludes_inferred_legacy_preview_root_when_metadata_missing(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    legacy_preview_dir = tmp_path / "legacy-previews"
    new_preview_dir = tmp_path / "new-previews"
    photo_root = tmp_path / "library"
    photo_root.mkdir()
    create_image(photo_root / "sample.jpg")

    initialize_database(db_path)

    with connect(db_path) as connection:
        scan_root(
            connection,
            root=photo_root,
            recursive=True,
            extensions=(".jpg",),
            preview_dir=legacy_preview_dir,
        )
        legacy_preview_path = Path(
            connection.execute("SELECT preview_path FROM files LIMIT 1").fetchone()["preview_path"]
        )
        connection.execute("DELETE FROM app_metadata WHERE key = 'preview_cache_root'")

        scan_root(
            connection,
            root=tmp_path,
            recursive=True,
            extensions=(".jpg",),
            preview_dir=new_preview_dir,
            generate_previews=False,
        )
        rows = connection.execute("SELECT path FROM files ORDER BY path ASC").fetchall()
        resolved_preview_root = resolve_preview_cache_root(connection, db_path=db_path)
        stored_preview_root = connection.execute(
            "SELECT value FROM app_metadata WHERE key = 'preview_cache_root'"
        ).fetchone()

    paths = [row["path"] for row in rows]
    assert str(legacy_preview_path.resolve()) not in paths
    assert resolved_preview_root == legacy_preview_dir.resolve()
    assert stored_preview_root is None


def test_unchanged_rescan_does_not_rewrite_preview_root_metadata(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    original_preview_dir = tmp_path / "original-previews"
    new_preview_dir = tmp_path / "new-previews"
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    create_image(photo_dir / "sample.jpg")

    initialize_database(db_path)

    with connect(db_path) as connection:
        scan_root(
            connection,
            root=photo_dir,
            recursive=True,
            extensions=(".jpg",),
            preview_dir=original_preview_dir,
        )

        scan_root(
            connection,
            root=photo_dir,
            recursive=True,
            extensions=(".jpg",),
            preview_dir=new_preview_dir,
        )

        preview_root = get_preview_cache_root(connection, db_path=db_path, persist=False)

    assert preview_root == original_preview_dir.resolve()
    assert not new_preview_dir.exists()


def test_unchanged_legacy_rescan_with_sidecar_root_stays_read_only(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "legacy-previews"
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    create_image(photo_dir / "sample.jpg")

    initialize_database(db_path)

    with connect(db_path) as connection:
        scan_root(
            connection,
            root=photo_dir,
            recursive=True,
            extensions=(".jpg",),
            preview_dir=preview_dir,
        )
        connection.execute("DELETE FROM app_metadata WHERE key = 'preview_cache_root'")
        (preview_dir / "keep-me.txt").write_text("legacy", encoding="utf-8")

        scan_root(
            connection,
            root=photo_dir,
            recursive=True,
            extensions=(".jpg",),
            preview_dir=preview_dir,
        )

        preview_root = resolve_preview_cache_root(connection, db_path=db_path)
        stored_preview_root = connection.execute(
            "SELECT value FROM app_metadata WHERE key = 'preview_cache_root'"
        ).fetchone()

    assert preview_root == preview_dir.resolve()
    assert stored_preview_root is None


def test_scan_excludes_all_legacy_preview_roots_when_metadata_missing(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    first_preview_dir = tmp_path / "legacy-previews-a"
    second_preview_dir = tmp_path / "legacy-previews-b"
    scan_preview_dir = tmp_path / "new-previews"
    first_root = tmp_path / "set-a"
    second_root = tmp_path / "set-b"
    first_root.mkdir()
    second_root.mkdir()
    create_image(first_root / "a.jpg")
    create_image(second_root / "b.jpg")

    initialize_database(db_path)

    with connect(db_path) as connection:
        scan_root(connection, root=first_root, recursive=True, extensions=(".jpg",), preview_dir=first_preview_dir)
        scan_root(connection, root=second_root, recursive=True, extensions=(".jpg",), preview_dir=second_preview_dir)
        connection.execute("DELETE FROM app_metadata WHERE key = 'preview_cache_root'")

        scan_root(
            connection,
            root=tmp_path,
            recursive=True,
            extensions=(".jpg",),
            preview_dir=scan_preview_dir,
            generate_previews=False,
        )

        rows = connection.execute("SELECT path FROM files ORDER BY path ASC").fetchall()

    paths = [row["path"] for row in rows]
    assert not any(path.startswith(str(first_preview_dir.resolve())) for path in paths)
    assert not any(path.startswith(str(second_preview_dir.resolve())) for path in paths)


def test_scan_excludes_stored_and_legacy_preview_roots_together(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    first_preview_dir = tmp_path / "preview-a"
    second_preview_dir = tmp_path / "preview-b"
    first_root = tmp_path / "set-a"
    second_root = tmp_path / "set-b"
    first_root.mkdir()
    second_root.mkdir()
    create_image(first_root / "a.jpg")
    create_image(second_root / "b.jpg")

    initialize_database(db_path)

    with connect(db_path) as connection:
        scan_root(connection, root=first_root, recursive=True, extensions=(".jpg",), preview_dir=first_preview_dir)
        scan_root(connection, root=second_root, recursive=True, extensions=(".jpg",), preview_dir=second_preview_dir)

        scan_root(
            connection,
            root=tmp_path,
            recursive=True,
            extensions=(".jpg",),
            preview_dir=second_preview_dir,
            generate_previews=False,
        )

        rows = connection.execute("SELECT path FROM files ORDER BY path ASC").fetchall()

    paths = [row["path"] for row in rows]
    assert not any(path.startswith(str(first_preview_dir.resolve())) for path in paths)
    assert not any(path.startswith(str(second_preview_dir.resolve())) for path in paths)


def test_clear_cache_all_does_not_sweep_unclaimed_hash_jpegs(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "legacy-previews"
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    source_path = photo_dir / "sample.jpg"
    create_image(source_path)

    initialize_database(db_path)

    with connect(db_path) as connection:
        scan_root(connection, root=photo_dir, recursive=True, extensions=(".jpg",), preview_dir=preview_dir)
        row = connection.execute("SELECT path, preview_path FROM files LIMIT 1").fetchone()
        db_backed_preview = Path(row["preview_path"])
        unrelated_hash_preview = preview_dir / ("a" * 40 + ".jpg")
        unrelated_hash_preview.write_bytes(b"not-a-managed-preview")
        connection.execute("DELETE FROM app_metadata WHERE key = 'preview_cache_root'")
        (preview_dir / ".shotsieve-preview-root").unlink()

        result = clear_cache_scope(connection, scope="all", preview_cache_root=preview_dir)

    assert result["files"] == 1
    assert not db_backed_preview.exists()
    assert unrelated_hash_preview.exists()


def test_clear_cache_all_removes_row_backed_previews_across_multiple_roots(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    first_preview_dir = tmp_path / "preview-a"
    second_preview_dir = tmp_path / "preview-b"
    first_root = tmp_path / "set-a"
    second_root = tmp_path / "set-b"
    first_root.mkdir()
    second_root.mkdir()
    create_image(first_root / "a.jpg")
    create_image(second_root / "b.jpg")

    initialize_database(db_path)

    with connect(db_path) as connection:
        scan_root(connection, root=first_root, recursive=True, extensions=(".jpg",), preview_dir=first_preview_dir)
        scan_root(connection, root=second_root, recursive=True, extensions=(".jpg",), preview_dir=second_preview_dir)
        preview_rows = connection.execute("SELECT preview_path FROM files ORDER BY id ASC").fetchall()
        first_preview = Path(preview_rows[0]["preview_path"])
        second_preview = Path(preview_rows[1]["preview_path"])

        result = clear_cache_scope(connection, scope="all", preview_cache_root=second_preview_dir)

    assert result["files"] == 2
    assert not first_preview.exists()
    assert not second_preview.exists()


def test_clear_cache_all_sweeps_claimed_orphans_across_multiple_roots(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    first_preview_dir = tmp_path / "preview-a"
    second_preview_dir = tmp_path / "preview-b"
    first_root = tmp_path / "set-a"
    second_root = tmp_path / "set-b"
    first_root.mkdir()
    second_root.mkdir()
    create_image(first_root / "a.jpg")
    create_image(second_root / "b.jpg")

    initialize_database(db_path)

    with connect(db_path) as connection:
        scan_root(connection, root=first_root, recursive=True, extensions=(".jpg",), preview_dir=first_preview_dir)
        scan_root(connection, root=second_root, recursive=True, extensions=(".jpg",), preview_dir=second_preview_dir)
        orphan_preview = first_preview_dir / ("b" * 40 + ".jpg")
        orphan_preview.write_bytes(b"orphan")

        result = clear_cache_scope(connection, scope="all", preview_cache_root=second_preview_dir)

    assert result["files"] == 2
    assert not orphan_preview.exists()


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


def test_rescan_with_changed_file_allows_preview_generation_when_claim_fails(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "legacy-previews"
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    source_path = photo_dir / "sample.jpg"
    create_image(source_path)

    initialize_database(db_path)

    with connect(db_path) as connection:
        scan_root(connection, root=photo_dir, recursive=True, extensions=(".jpg",), preview_dir=preview_dir)
        connection.execute("DELETE FROM app_metadata WHERE key = 'preview_cache_root'")
        (preview_dir / "keep-me.txt").write_text("legacy", encoding="utf-8")
        create_pattern_image(source_path, blur_radius=3)

        summary = scan_root(connection, root=photo_dir, recursive=True, extensions=(".jpg",), preview_dir=preview_dir)
        row = connection.execute("SELECT preview_path FROM files LIMIT 1").fetchone()

    assert summary.files_updated == 1
    assert row["preview_path"] is not None
    assert Path(row["preview_path"]).exists()


def test_scan_excludes_claimed_preview_root_without_metadata_or_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "previews"
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    create_image(photo_dir / "sample.jpg")

    initialize_database(db_path)

    with connect(db_path) as connection:
        scan_root(connection, root=photo_dir, recursive=True, extensions=(".jpg",), preview_dir=preview_dir)
        preview_path = Path(connection.execute("SELECT preview_path FROM files LIMIT 1").fetchone()["preview_path"])
        connection.execute("DELETE FROM files")
        connection.execute("DELETE FROM app_metadata WHERE key = 'preview_cache_root'")

        scan_root(
            connection,
            root=tmp_path,
            recursive=True,
            extensions=(".jpg",),
            preview_dir=preview_dir,
            generate_previews=False,
        )

        rows = connection.execute("SELECT path FROM files ORDER BY path ASC").fetchall()

    paths = [row["path"] for row in rows]
    assert str(preview_path.resolve()) not in paths


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


def test_clear_cache_scope_all_removes_orphan_preview_files(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "custom-previews"
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    create_image(photo_dir / "sample.jpg")

    initialize_database(db_path)

    with connect(db_path) as connection:
        scan_root(connection, root=photo_dir, recursive=True, extensions=(".jpg",), preview_dir=preview_dir)
        orphan_preview = preview_dir / ("f" * 40 + ".jpg")
        sidecar_file = preview_dir / "keep-me.txt"
        create_image(orphan_preview)
        sidecar_file.write_text("leave me alone", encoding="utf-8")
        assert orphan_preview.exists()
        assert sidecar_file.exists()

        preview_cache_root = get_preview_cache_root(connection, db_path=db_path)
        clear_cache_scope(connection, scope="all", preview_cache_root=preview_cache_root)

    assert not orphan_preview.exists()
    assert sidecar_file.exists()


def test_clear_cache_scope_scores_only(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "previews"
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    create_image(photo_dir / "sample.jpg")

    initialize_database(db_path)

    with connect(db_path) as connection:
        scan_root(connection, root=photo_dir, recursive=True, extensions=(".jpg",), preview_dir=preview_dir)
        score_with_fake_learned_backend(connection)

        result = clear_cache_scope(connection, scope="scores")
        score_count = connection.execute("SELECT COUNT(*) AS count FROM scores").fetchone()["count"]
        file_count = connection.execute("SELECT COUNT(*) AS count FROM files").fetchone()["count"]

    assert result["scores"] == 1
    assert score_count == 0
    assert file_count == 1


def test_media_path_for_tiff_prefers_preview_when_generated(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "previews"
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    tiff_path = photo_dir / "sample.tiff"
    create_tiff_image(tiff_path)

    initialize_database(db_path)

    with connect(db_path) as connection:
        scan_root(
            connection,
            root=photo_dir,
            recursive=True,
            extensions=(".tiff",),
            preview_dir=preview_dir,
            generate_previews=True,
        )
        row = connection.execute(
            "SELECT id, preview_path, preview_status FROM files LIMIT 1"
        ).fetchone()

        selected = media_path_for_file(connection, file_id=int(row["id"]), variant="preview")

    assert row["preview_status"] == "ready"
    assert row["preview_path"] is not None
    assert selected is not None
    assert selected.resolve() == Path(str(row["preview_path"])).resolve()


def test_media_path_for_tiff_fast_scan_falls_back_to_source(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "previews"
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    tiff_path = photo_dir / "sample.tiff"
    create_tiff_image(tiff_path)

    initialize_database(db_path)

    with connect(db_path) as connection:
        scan_root(
            connection,
            root=photo_dir,
            recursive=True,
            extensions=(".tiff",),
            preview_dir=preview_dir,
            generate_previews=False,
        )
        file_id = int(connection.execute("SELECT id FROM files LIMIT 1").fetchone()["id"])
        selected = media_path_for_file(connection, file_id=file_id, variant="preview")

    assert selected is not None
    assert selected.resolve() == tiff_path.resolve()


def test_media_path_source_variant_returns_original_for_raw_when_preview_ready(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "previews"
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    raw_path = photo_dir / "sample.cr2"
    raw_path.write_bytes(b"fake-raw-bytes")

    preview_dir.mkdir(parents=True, exist_ok=True)
    ready_preview = preview_dir / "sample-preview.jpg"
    create_image(ready_preview)

    initialize_database(db_path)

    with connect(db_path) as connection:
        scan_root(
            connection,
            root=photo_dir,
            recursive=True,
            extensions=(".cr2",),
            preview_dir=preview_dir,
            generate_previews=False,
        )
        file_id = int(connection.execute("SELECT id FROM files LIMIT 1").fetchone()["id"])
        connection.execute(
            "UPDATE files SET preview_path = ?, preview_status = 'ready' WHERE id = ?",
            (str(ready_preview.resolve()), file_id),
        )

        selected_source = media_path_for_file(connection, file_id=file_id, variant="source")
        selected_preview = media_path_for_file(connection, file_id=file_id, variant="preview")

    assert selected_source is not None
    assert selected_source.resolve() == raw_path.resolve()
    assert selected_preview is not None
    assert selected_preview.resolve() == ready_preview.resolve()


def test_media_path_heic_preview_fallback_is_source_when_no_generated_preview(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "previews"
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    heic_path = photo_dir / "sample.heic"
    heic_path.write_bytes(b"fake-heic-bytes")

    initialize_database(db_path)

    with connect(db_path) as connection:
        scan_root(
            connection,
            root=photo_dir,
            recursive=True,
            extensions=(".heic",),
            preview_dir=preview_dir,
            generate_previews=False,
        )
        file_id = int(connection.execute("SELECT id FROM files LIMIT 1").fetchone()["id"])
        selected = media_path_for_file(connection, file_id=file_id, variant="preview")

    assert selected is not None
    assert selected.resolve() == heic_path.resolve()


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