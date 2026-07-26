import platform
from pathlib import Path
from typing import cast

import pytest
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFilter

from shotsieve.db import connect, initialize_database
from shotsieve.learned_iqa import LearnedScoreResult
from shotsieve.review import (
    count_review_files,
    get_review_file_detail,
    list_analysis_diagnostics,
    list_review_files,
    list_review_state_file_ids,
    review_selection_revision,
    review_overview,
    update_review_state,
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


def test_analysis_diagnostics_explain_preview_and_model_failures(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "previews"
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    preview_failed = photo_dir / "preview-failed.jpg"
    model_failed = photo_dir / "model-failed.jpg"
    create_image(preview_failed)
    create_image(model_failed)
    initialize_database(db_path)

    class MixedBackend:
        name = "topiq_nr"
        model_version = "fake:mixed"

        def score_paths(self, image_paths, *, batch_size: int = 4, resource_profile: str | None = None):
            return [
                LearnedScoreResult(
                    raw_score=None if path.name == "model-failed.jpg" else 0.82,
                    normalized_score=None if path.name == "model-failed.jpg" else 82.0,
                    confidence=None if path.name == "model-failed.jpg" else 91.0,
                    error="model unavailable" if path.name == "model-failed.jpg" else None,
                )
                for path in image_paths
            ]

    with connect(db_path) as connection:
        scan_root(
            connection,
            root=photo_dir,
            recursive=True,
            extensions=(".jpg",),
            preview_dir=preview_dir,
            generate_previews=False,
        )
        score_files(connection, learned_backend_factory=lambda _model_name: MixedBackend())
        connection.execute(
            """
            UPDATE files
            SET preview_status = 'failed', last_error = 'decoder could not read image',
                analysis_status = NULL, analysis_error = NULL
            WHERE path = ?
            """,
            (str(preview_failed),),
        )
        connection.execute("DELETE FROM scores WHERE file_id = (SELECT id FROM files WHERE path = ?)", (str(preview_failed),))
        diagnostics = list_analysis_diagnostics(connection, root=str(photo_dir))

    assert diagnostics["total"] == 2
    by_name = {Path(str(item["path"])).name: item for item in diagnostics["items"]}
    assert by_name["preview-failed.jpg"]["status"] == "failed"
    assert "decoder could not read image" in str(by_name["preview-failed.jpg"]["error"])
    assert by_name["model-failed.jpg"]["status"] == "failed"
    assert by_name["model-failed.jpg"]["error"] == "model unavailable"


def test_unchanged_scan_preserves_prior_analysis_diagnostic(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "previews"
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    create_image(photo_dir / "sample.jpg")
    initialize_database(db_path)

    with connect(db_path) as connection:
        scan_root(connection, root=photo_dir, recursive=True, extensions=(".jpg",), preview_dir=preview_dir)
        connection.execute(
            """
            UPDATE files
            SET analysis_status = 'failed', analysis_error = 'model process stopped',
                last_analysis_time = '2026-03-24T00:00:00+00:00'
            """
        )
        scan_root(connection, root=photo_dir, recursive=True, extensions=(".jpg",), preview_dir=preview_dir)
        diagnostic = list_analysis_diagnostics(connection, root=str(photo_dir))["items"][0]

    assert diagnostic["status"] == "failed"
    assert diagnostic["error"] == "model process stopped"


def test_review_overview_separates_active_library_from_catalog(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "previews"
    root_a = tmp_path / "library-a"
    root_b = tmp_path / "library-b"
    root_a.mkdir()
    root_b.mkdir()
    create_image(root_a / "a-1.jpg")
    create_image(root_a / "a-2.jpg")
    create_image(root_b / "b-1.jpg")

    initialize_database(db_path)

    with connect(db_path) as connection:
        scan_root(connection, root=root_a, recursive=True, extensions=(".jpg",), preview_dir=preview_dir)
        scan_root(connection, root=root_b, recursive=True, extensions=(".jpg",), preview_dir=preview_dir)
        score_with_fake_learned_backend(connection)
        b_id = int(connection.execute("SELECT id FROM files WHERE path LIKE ?", ("%b-1.jpg",)).fetchone()["id"])
        update_review_state(
            connection,
            file_id=b_id,
            decision_state="delete",
            delete_marked=True,
            export_marked=False,
            updated_time="2026-07-20T00:00:00+00:00",
        )

        overview = review_overview(connection, root=str(root_b))
        root_a_revision = review_selection_revision(connection, scope="review-browser", root=str(root_a))
        root_b_revision = review_selection_revision(connection, scope="review-browser", root=str(root_b))
        overview_none = review_overview(connection, root=None)
        overview_empty = review_overview(connection, root=" ")

    active = _dict_value(overview["active_library"])
    catalog = _dict_value(overview["catalog"])
    summary = _dict_value(overview["summary"])
    assert active == {
        "root": str(root_b.resolve()),
        "total_files": 1,
        "scored_files": 1,
        "delete_marked": 1,
        "export_marked": 0,
    }
    assert summary == {
        "total_files": 1,
        "scored_files": 1,
        "delete_marked": 1,
        "export_marked": 0,
    }
    assert catalog == {
        "total_files": 3,
        "scored_files": 3,
        "delete_marked": 1,
        "export_marked": 0,
    }
    assert root_a_revision != root_b_revision

    active_none = _dict_value(overview_none["active_library"])
    assert active_none == {
        "root": None,
        "total_files": 0,
        "scored_files": 0,
        "delete_marked": 0,
        "export_marked": 0,
    }

    active_empty = _dict_value(overview_empty["active_library"])
    assert active_empty == {
        "root": None,
        "total_files": 0,
        "scored_files": 0,
        "delete_marked": 0,
        "export_marked": 0,
    }


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