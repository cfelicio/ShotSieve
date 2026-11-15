from __future__ import annotations

from pathlib import Path

from shotsieve.db import (
    candidate_companion_path_keys,
    connect,
    get_all_paired_file_ids,
    initialize_database,
    normalize_resolved_path,
    reconcile_catalog_pairs,
)
from shotsieve.export import export_files
from shotsieve.review import (
    count_review_files,
    get_review_file_detail,
    list_review_files,
    review_selection_revision,
    update_review_state,
    update_review_state_batch,
)
from shotsieve.review_cache import delete_files
from shotsieve.scoring import LearnedScoreResult, fetch_score_rows, upsert_score_row
from shotsieve.web import build_options_payload


def _create_sample_pair(
    db_path: Path,
    folder: Path,
    stem: str = "DSC_0001",
    raw_ext: str = ".arw",
    rendered_ext: str = ".jpg",
) -> tuple[int, int, Path, Path]:
    folder.mkdir(parents=True, exist_ok=True)
    raw_file = folder / f"{stem}{raw_ext}"
    rendered_file = folder / f"{stem}{rendered_ext}"
    raw_file.write_bytes(b"RAW_DATA_12345")
    rendered_file.write_bytes(b"JPEG_DATA_67890")

    connection = connect(db_path)
    try:
        cur1 = connection.execute(
            """
            INSERT INTO files(path, path_key, format, size_bytes, modified_time, width, height, scan_status)
            VALUES(?, ?, ?, ?, 1000.0, 4000, 3000, 'ready')
            """,
            (str(rendered_file), normalize_resolved_path(rendered_file), rendered_ext.lstrip("."), len(rendered_file.read_bytes())),
        )
        rendered_id = cur1.lastrowid

        cur2 = connection.execute(
            """
            INSERT INTO files(path, path_key, format, size_bytes, modified_time, width, height, scan_status)
            VALUES(?, ?, ?, ?, 1000.0, 4000, 3000, 'ready')
            """,
            (str(raw_file), normalize_resolved_path(raw_file), raw_ext.lstrip("."), len(raw_file.read_bytes())),
        )
        raw_id = cur2.lastrowid

        connection.execute(
            """
            INSERT INTO scan_runs(started_time, completed_time, root_path, status)
            VALUES('2026-01-01T00:00:00Z', '2026-01-01T00:01:00Z', ?, 'completed')
            """,
            (str(folder),),
        )
        connection.commit()
    finally:
        connection.close()

    return rendered_id, raw_id, rendered_file, raw_file


def test_candidate_companion_path_keys(tmp_path: Path) -> None:
    raw_path = tmp_path / "DSC_0042.ARW"
    candidates = candidate_companion_path_keys(raw_path)
    assert len(candidates) > 0
    jpg_key = normalize_resolved_path(tmp_path / "DSC_0042.jpg")
    heic_key = normalize_resolved_path(tmp_path / "DSC_0042.heic")
    assert jpg_key in candidates
    assert heic_key in candidates

    jpg_path = tmp_path / "DSC_0042.JPG"
    raw_candidates = candidate_companion_path_keys(jpg_path)
    assert len(raw_candidates) > 0
    arw_key = normalize_resolved_path(tmp_path / "DSC_0042.arw")
    cr3_key = normalize_resolved_path(tmp_path / "DSC_0042.cr3")
    assert arw_key in raw_candidates
    assert cr3_key in raw_candidates


def test_reconcile_and_link_file_pairs(tmp_path: Path) -> None:
    db_path = tmp_path / "catalog.db"
    initialize_database(db_path)

    rendered_id, raw_id, _, _ = _create_sample_pair(db_path, tmp_path / "photos")

    connection = connect(db_path)
    try:
        newly_linked = reconcile_catalog_pairs(connection)
        connection.commit()
        assert newly_linked == 1

        pairs = connection.execute("SELECT primary_file_id, companion_file_id FROM file_pairs").fetchall()
        assert len(pairs) == 1
        assert int(pairs[0]["primary_file_id"]) == rendered_id
        assert int(pairs[0]["companion_file_id"]) == raw_id

        # Idempotent: running again links 0 additional
        assert reconcile_catalog_pairs(connection) == 0

        paired_ids = get_all_paired_file_ids(connection, [rendered_id])
        assert set(paired_ids) == {rendered_id, raw_id}

        paired_ids_from_raw = get_all_paired_file_ids(connection, [raw_id])
        assert set(paired_ids_from_raw) == {rendered_id, raw_id}
    finally:
        connection.close()


def test_review_queries_with_grouping(tmp_path: Path) -> None:
    db_path = tmp_path / "catalog.db"
    initialize_database(db_path)

    rendered_id, raw_id, _, _ = _create_sample_pair(db_path, tmp_path / "photos")
    connection = connect(db_path)
    try:
        reconcile_catalog_pairs(connection)
        connection.execute(
            "INSERT INTO scores(file_id, overall_score, learned_score_normalized) VALUES(?, 85.0, 85.0)",
            (rendered_id,),
        )
        connection.execute(
            "INSERT INTO scores(file_id, overall_score, learned_score_normalized) VALUES(?, 85.0, 85.0)",
            (raw_id,),
        )
        connection.commit()

        # Grouping enabled (default): 1 tile returned
        total = count_review_files(connection, group_pairs=True)
        assert total == 1

        files = list_review_files(connection, group_pairs=True)
        assert len(files) == 1
        assert files[0]["id"] == rendered_id
        assert files[0]["is_pair"] is True
        assert files[0]["companion"] is not None
        assert files[0]["companion"]["id"] == raw_id
        assert files[0]["companion"]["format"] == "arw"

        # Grouping disabled: 2 tiles returned
        total_unpaired = count_review_files(connection, group_pairs=False)
        assert total_unpaired == 2

        files_unpaired = list_review_files(connection, group_pairs=False)
        assert len(files_unpaired) == 2

        detail = get_review_file_detail(connection, rendered_id)
        assert detail is not None
        assert detail["is_pair"] is True
        assert detail["companion"]["id"] == raw_id
    finally:
        connection.close()


def test_update_review_state_synchronizes_pairs(tmp_path: Path) -> None:
    db_path = tmp_path / "catalog.db"
    initialize_database(db_path)

    rendered_id, raw_id, _, _ = _create_sample_pair(db_path, tmp_path / "photos")
    connection = connect(db_path)
    try:
        reconcile_catalog_pairs(connection)
        connection.commit()

        update_review_state(connection, file_id=rendered_id, decision_state="export", export_marked=True, updated_time="2026-11-14T00:00:00Z")
        connection.commit()

        rendered_state = connection.execute("SELECT decision_state, export_marked FROM review_state WHERE file_id = ?", (rendered_id,)).fetchone()
        raw_state = connection.execute("SELECT decision_state, export_marked FROM review_state WHERE file_id = ?", (raw_id,)).fetchone()

        assert rendered_state["decision_state"] == "export"
        assert rendered_state["export_marked"] == 1
        assert raw_state["decision_state"] == "export"
        assert raw_state["export_marked"] == 1

        # Test batch update sync
        update_review_state_batch(connection, file_ids=[rendered_id], decision_state="delete", delete_marked=True, export_marked=False, updated_time="2026-11-14T01:00:00Z")
        connection.commit()

        rendered_del = connection.execute("SELECT decision_state, delete_marked FROM review_state WHERE file_id = ?", (rendered_id,)).fetchone()
        raw_del = connection.execute("SELECT decision_state, delete_marked FROM review_state WHERE file_id = ?", (raw_id,)).fetchone()

        assert rendered_del["decision_state"] == "delete"
        assert rendered_del["delete_marked"] == 1
        assert raw_del["decision_state"] == "delete"
        assert raw_del["delete_marked"] == 1
    finally:
        connection.close()


def test_export_pair_modes(tmp_path: Path) -> None:
    db_path = tmp_path / "catalog.db"
    initialize_database(db_path)

    rendered_id, raw_id, rendered_file, raw_file = _create_sample_pair(db_path, tmp_path / "photos")
    connection = connect(db_path)
    try:
        reconcile_catalog_pairs(connection)
        connection.commit()

        dest_both = tmp_path / "dest_both"
        dest_both.mkdir()
        summary_both = export_files(connection, file_ids=[rendered_id], destination=str(dest_both), mode="copy", pair_mode="both")
        assert summary_both.copied == 2
        assert (dest_both / rendered_file.name).exists()
        assert (dest_both / raw_file.name).exists()

        dest_raw = tmp_path / "dest_raw"
        dest_raw.mkdir()
        summary_raw = export_files(connection, file_ids=[rendered_id], destination=str(dest_raw), mode="copy", pair_mode="raw_only")
        assert summary_raw.copied == 1
        assert not (dest_raw / rendered_file.name).exists()
        assert (dest_raw / raw_file.name).exists()

        dest_rendered = tmp_path / "dest_rendered"
        dest_rendered.mkdir()
        summary_rendered = export_files(connection, file_ids=[rendered_id], destination=str(dest_rendered), mode="copy", pair_mode="rendered_only")
        assert summary_rendered.copied == 1
        assert (dest_rendered / rendered_file.name).exists()
        assert not (dest_rendered / raw_file.name).exists()
    finally:
        connection.close()


def test_delete_pair_removes_both_files(tmp_path: Path) -> None:
    db_path = tmp_path / "catalog.db"
    initialize_database(db_path)

    rendered_id, raw_id, rendered_file, raw_file = _create_sample_pair(db_path, tmp_path / "photos")
    connection = connect(db_path)
    try:
        reconcile_catalog_pairs(connection)
        connection.commit()

        delete_files(connection, file_ids=[rendered_id], delete_from_disk=True)
        connection.commit()

        assert not rendered_file.exists()
        assert not raw_file.exists()

        count = connection.execute("SELECT COUNT(*) AS c FROM files WHERE id IN (?, ?)", (rendered_id, raw_id)).fetchone()["c"]
        assert count == 0

        pairs_count = connection.execute("SELECT COUNT(*) AS c FROM file_pairs WHERE primary_file_id = ? OR companion_file_id = ?", (rendered_id, raw_id)).fetchone()["c"]
        assert pairs_count == 0
    finally:
        connection.close()


def test_fetch_score_rows_with_grouping(tmp_path: Path) -> None:
    db_path = tmp_path / "catalog.db"
    initialize_database(db_path)

    photo_dir = tmp_path / "photos"
    rendered_id, raw_id, _, _ = _create_sample_pair(db_path, photo_dir)
    connection = connect(db_path)
    try:
        reconcile_catalog_pairs(connection)
        connection.commit()

        # Grouping enabled (default): only primary (rendered) file is queued
        rows_grouped = fetch_score_rows(connection, limit=10, raw_root=photo_dir, group_pairs=True)
        assert len(rows_grouped) == 1
        assert rows_grouped[0]["id"] == rendered_id

        # Grouping disabled: both files are queued
        rows_all = fetch_score_rows(connection, limit=10, raw_root=photo_dir, group_pairs=False)
        assert len(rows_all) == 2
    finally:
        connection.close()


def test_scoring_mirrors_to_companion_raw(tmp_path: Path) -> None:
    db_path = tmp_path / "catalog.db"
    initialize_database(db_path)

    rendered_id, raw_id, _, _ = _create_sample_pair(db_path, tmp_path / "photos")
    connection = connect(db_path)
    try:
        reconcile_catalog_pairs(connection)
        connection.commit()

        rendered_row = connection.execute("SELECT id, modified_time, size_bytes FROM files WHERE id = ?", (rendered_id,)).fetchone()
        learned = LearnedScoreResult(raw_score=88.5, normalized_score=88.5, confidence=1.0)
        upsert_score_row(
            connection,
            rendered_row,
            learned=learned,
            learned_backend="musiq",
            learned_model_version="1.0",
        )
        connection.commit()

        raw_score = connection.execute(
            "SELECT overall_score, learned_backend, model_version FROM scores WHERE file_id = ?",
            (raw_id,),
        ).fetchone()
        assert raw_score is not None
        assert raw_score["overall_score"] == 88.5
        assert raw_score["learned_backend"] == "musiq"
        assert raw_score["model_version"] == "learned:1.0"
    finally:
        connection.close()


def test_build_options_payload_includes_pair_settings(tmp_path: Path) -> None:
    db_path = tmp_path / "catalog.db"
    initialize_database(db_path)
    options = build_options_payload(db_path=db_path)
    assert options["default_group_pairs"] is True
    assert options["default_pair_export_mode"] == "both"
    assert options["pair_export_modes"] == ["both", "raw_only", "rendered_only"]


def test_selection_revision_with_grouping(tmp_path: Path) -> None:
    db_path = tmp_path / "catalog.db"
    initialize_database(db_path)

    rendered_id, raw_id, _, _ = _create_sample_pair(db_path, tmp_path / "photos")
    connection = connect(db_path)
    try:
        reconcile_catalog_pairs(connection)
        connection.execute(
            "INSERT INTO scores(file_id, overall_score, learned_score_normalized) VALUES(?, 85.0, 85.0)",
            (rendered_id,),
        )
        connection.execute(
            "INSERT INTO scores(file_id, overall_score, learned_score_normalized) VALUES(?, 85.0, 85.0)",
            (raw_id,),
        )
        connection.commit()

        rev_grouped = review_selection_revision(connection, scope="review-browser", group_pairs=True)
        rev_ungrouped = review_selection_revision(connection, scope="review-browser", group_pairs=False)
        assert "|1|" in rev_grouped
        assert "|2|" in rev_ungrouped
        assert rev_grouped != rev_ungrouped

        rev_marked_before = review_selection_revision(connection, scope="review-browser", marked="export", group_pairs=True)
        assert "|0|" in rev_marked_before

        update_review_state(connection, file_id=rendered_id, decision_state="export", export_marked=True, updated_time="2026-11-14T00:00:00Z")
        connection.commit()

        rev_marked_after = review_selection_revision(connection, scope="review-browser", marked="export", group_pairs=True)
        assert "|1|" in rev_marked_after
        assert rev_marked_after != rev_marked_before
    finally:
        connection.close()

