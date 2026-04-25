import os
from pathlib import Path
from typing import cast

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFilter

from shotsieve import preview as preview_module
from shotsieve import scoring as scoring_module
from shotsieve.db import connect, initialize_database
from shotsieve.learned_iqa import LearnedScoreResult
from shotsieve.scanner import scan_root
from shotsieve.scoring import AnalysisProgress, compare_learned_models, score_files, select_analysis_path


def _row_value(row: object, key: str) -> object:
    return cast(dict[str, object], row)[key]


def _coerce_int(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        return int(value)
    raise TypeError(f"Expected int-like test value, got {type(value).__name__}")


def _row_id(row: object) -> int:
    return _coerce_int(_row_value(row, "id"))


def test_preview_legacy_extension_aliases_are_removed_from_preview_module() -> None:
    assert not hasattr(preview_module, "RAW_PREVIEWABLE_EXTENSIONS")
    assert not hasattr(preview_module, "HEIF_PREVIEWABLE_EXTENSIONS")
    assert not hasattr(preview_module, "PIL_PREVIEWABLE_EXTENSIONS")
    assert not hasattr(preview_module, "PREVIEWABLE_EXTENSIONS")


def test_scoring_compare_progress_alias_is_removed_from_scoring_module() -> None:
    assert not hasattr(scoring_module, "CompareProgress")


def test_preview_legacy_extension_aliases_are_removed_from_star_imports() -> None:
    namespace: dict[str, object] = {}
    exec("from shotsieve.preview import *", namespace)

    assert "RAW_PREVIEWABLE_EXTENSIONS" not in namespace
    assert "HEIF_PREVIEWABLE_EXTENSIONS" not in namespace
    assert "PIL_PREVIEWABLE_EXTENSIONS" not in namespace
    assert "PREVIEWABLE_EXTENSIONS" not in namespace


def test_scoring_compare_progress_alias_is_removed_from_star_imports() -> None:
    namespace: dict[str, object] = {}
    exec("from shotsieve.scoring import *", namespace)

    assert "CompareProgress" not in namespace


def test_prepare_analysis_candidates_prefers_generated_preview_and_returns_persistence_results(tmp_path: Path, monkeypatch) -> None:
    preview_dir = tmp_path / "previews"
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()

    tiff_path = photo_dir / "sample.tiff"
    Image.new("RGB", (120, 80), color=(35, 80, 145)).save(tiff_path, format="TIFF")

    ready_preview = preview_dir / "sample-tiff-preview.jpg"
    preview_dir.mkdir(parents=True, exist_ok=True)
    create_image(ready_preview)

    def fake_generate_previews_parallel(source_paths, generated_preview_dir: Path, *, max_workers=None, progress_callback=None):
        assert source_paths == [tiff_path]
        assert generated_preview_dir == preview_dir
        assert max_workers == 7
        return [
            preview_module.PreviewResult(
                path=str(ready_preview),
                status="ready",
                width=120,
                height=80,
                capture_time=None,
                error_text=None,
            )
        ]

    monkeypatch.setattr(scoring_module, "generate_previews_parallel", fake_generate_previews_parallel)

    prepared = scoring_module._prepare_analysis_candidates(
        [
            {
                "id": 1,
                "path": str(tiff_path),
                "preview_path": None,
                "preview_status": None,
            }
        ],
        preview_dir=preview_dir,
        preview_workers=7,
        resource_profile="normal",
        preview_progress_callback=None,
    )

    assert len(prepared.analysis_candidates) == 1
    assert _row_id(prepared.analysis_candidates[0].row) == 1
    assert prepared.analysis_candidates[0].analysis_path.resolve() == ready_preview.resolve()
    assert len(prepared.generated_preview_results) == 1
    assert _row_id(prepared.generated_preview_results[0].row) == 1
    assert prepared.generated_preview_results[0].preview_result.status == "ready"
    assert prepared.has_ready_generated_preview is True
    assert prepared.unresolved_preview_results == []
    assert prepared.unavailable_rows == []


def test_prepare_analysis_candidates_returns_fallback_paths_and_unresolved_failures(tmp_path: Path, monkeypatch) -> None:
    preview_dir = tmp_path / "previews"
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()

    tiff_path = photo_dir / "sample.tiff"
    Image.new("RGB", (120, 80), color=(35, 80, 145)).save(tiff_path, format="TIFF")
    raw_path = photo_dir / "sample.cr2"
    raw_path.write_bytes(b"fake-raw")

    def fake_generate_previews_parallel(source_paths, generated_preview_dir: Path, *, max_workers=None, progress_callback=None):
        assert source_paths == [tiff_path, raw_path]
        assert generated_preview_dir == preview_dir
        return [
            preview_module.PreviewResult(
                path=None,
                status="failed",
                width=None,
                height=None,
                capture_time=None,
                error_text="tiff preview failed",
            ),
            preview_module.PreviewResult(
                path=None,
                status="failed",
                width=None,
                height=None,
                capture_time=None,
                error_text="raw preview failed",
            ),
        ]

    monkeypatch.setattr(scoring_module, "generate_previews_parallel", fake_generate_previews_parallel)

    prepared = scoring_module._prepare_analysis_candidates(
        [
            {
                "id": 1,
                "path": str(tiff_path),
                "preview_path": None,
                "preview_status": None,
            },
            {
                "id": 2,
                "path": str(raw_path),
                "preview_path": None,
                "preview_status": None,
            },
        ],
        preview_dir=preview_dir,
        preview_workers=3,
        resource_profile=None,
        preview_progress_callback=None,
    )

    assert len(prepared.analysis_candidates) == 1
    assert _row_id(prepared.analysis_candidates[0].row) == 1
    assert prepared.analysis_candidates[0].analysis_path.resolve() == tiff_path.resolve()
    assert len(prepared.generated_preview_results) == 2
    assert [_row_id(result.row) for result in prepared.generated_preview_results] == [1, 2]
    assert len(prepared.unresolved_preview_results) == 1
    assert _row_id(prepared.unresolved_preview_results[0].row) == 2
    assert prepared.unresolved_preview_results[0].preview_result.status == "failed"
    assert prepared.unresolved_preview_results[0].preview_result.error_text == "raw preview failed"
    assert prepared.has_ready_generated_preview is False
    assert prepared.unavailable_rows == []


def test_score_defaults_to_ai_only_when_backend_unspecified(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "previews"
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    create_image(photo_dir / "sample.jpg")

    selected_models: list[str] = []

    class FakeLearnedBackend:
        name = "topiq_nr"
        model_version = "fake:topiq_nr"

        def score_paths(self, image_paths, *, batch_size: int = 4, resource_profile: str | None = None):
            return [LearnedScoreResult(raw_score=0.82, normalized_score=82.0, confidence=91.0) for _ in image_paths]

    initialize_database(db_path)

    with connect(db_path) as connection:
        scan_root(
            connection,
            root=photo_dir,
            recursive=True,
            extensions=(".jpg",),
            preview_dir=preview_dir,
        )
        summary = score_files(
            connection,
            learned_backend_factory=lambda model_name: selected_models.append(model_name) or FakeLearnedBackend(),
        )
        row = connection.execute(
            """
            SELECT files.path, scores.overall_score,
                   scores.learned_backend, scores.preset_name
            FROM scores
            JOIN files ON files.id = scores.file_id
            """
        ).fetchone()

    assert selected_models == ["topiq_nr"]
    assert summary.files_considered == 1
    assert summary.files_scored == 1
    assert summary.learned_scored == 1
    assert summary.files_failed == 0
    assert row["learned_backend"] == "topiq_nr"
    assert row["preset_name"] == "learned-only"
    assert row["overall_score"] == 82.0


def test_score_populates_learned_columns_with_backend(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "previews"
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    create_image(photo_dir / "sample.jpg")

    class FakeLearnedBackend:
        name = "topiq_nr"
        model_version = "fake:topiq_nr"

        def score_paths(self, image_paths, *, batch_size: int = 4, resource_profile: str | None = None):
            return [LearnedScoreResult(raw_score=0.82, normalized_score=82.0, confidence=91.0) for _ in image_paths]

    initialize_database(db_path)

    with connect(db_path) as connection:
        scan_root(
            connection,
            root=photo_dir,
            recursive=True,
            extensions=(".jpg",),
            preview_dir=preview_dir,
        )
        summary = score_files(
            connection,
            learned_backend_name="topiq_nr",
            learned_backend_factory=lambda model_name: FakeLearnedBackend(),
        )
        row = connection.execute(
            """
            SELECT learned_backend, learned_raw_score, learned_score_normalized, learned_confidence,
                   overall_score, preset_name, model_version
            FROM scores
            """
        ).fetchone()

    assert summary.files_scored == 1
    assert summary.learned_scored == 1
    assert row["learned_backend"] == "topiq_nr"
    assert row["learned_raw_score"] == 0.82
    assert row["learned_score_normalized"] == 82.0
    assert row["learned_confidence"] == 91.0
    assert row["preset_name"] == "learned-only"
    assert row["overall_score"] == 82.0
    assert "learned:fake:topiq_nr" in row["model_version"]
    assert "technical:" not in row["model_version"]
    assert "fusion:" not in row["model_version"]


def test_score_refreshes_when_switching_backends(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "previews"
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    create_image(photo_dir / "sample.jpg")

    class FakeLearnedBackend:
        name = "topiq_nr"
        model_version = "fake:topiq_nr"

        def score_paths(self, image_paths, *, batch_size: int = 4, resource_profile: str | None = None):
            return [LearnedScoreResult(raw_score=0.9, normalized_score=90.0, confidence=88.0) for _ in image_paths]

    initialize_database(db_path)

    with connect(db_path) as connection:
        scan_root(
            connection,
            root=photo_dir,
            recursive=True,
            extensions=(".jpg",),
            preview_dir=preview_dir,
        )
        score_files(
            connection,
            learned_backend_name="topiq_nr",
            learned_backend_factory=lambda model_name: FakeLearnedBackend(),
        )
        first = connection.execute(
            "SELECT overall_score, learned_backend, model_version FROM scores"
        ).fetchone()

        class SecondBackend:
            name = "arniqa"
            model_version = "fake:arniqa"

            def score_paths(self, image_paths, *, batch_size: int = 4, resource_profile: str | None = None):
                return [LearnedScoreResult(raw_score=0.9, normalized_score=90.0, confidence=88.0) for _ in image_paths]

        summary = score_files(
            connection,
            learned_backend_name="arniqa",
            learned_backend_factory=lambda model_name: SecondBackend(),
        )
        second = connection.execute(
            "SELECT overall_score, preset_name, learned_score_normalized, learned_backend, model_version FROM scores"
        ).fetchone()

    assert summary.learned_scored == 1
    assert first["learned_backend"] == "topiq_nr"
    assert second["learned_backend"] == "arniqa"
    assert second["preset_name"] == "learned-only"
    assert second["overall_score"] == second["learned_score_normalized"] == 90.0
    assert "learned:fake:arniqa" in second["model_version"]
    assert "topiq_nr" not in second["model_version"]


def test_score_refreshes_when_model_version_changes(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "previews"
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    create_image(photo_dir / "sample.jpg")

    class FirstBackend:
        name = "topiq_nr"
        model_version = "fake:v1"

        def score_paths(self, image_paths, *, batch_size: int = 4, resource_profile: str | None = None):
            return [LearnedScoreResult(raw_score=0.82, normalized_score=82.0, confidence=91.0) for _ in image_paths]

    class SecondBackend:
        name = "topiq_nr"
        model_version = "fake:v2"

        def score_paths(self, image_paths, *, batch_size: int = 4, resource_profile: str | None = None):
            return [LearnedScoreResult(raw_score=0.63, normalized_score=63.0, confidence=88.0) for _ in image_paths]

    initialize_database(db_path)

    with connect(db_path) as connection:
        scan_root(
            connection,
            root=photo_dir,
            recursive=True,
            extensions=(".jpg",),
            preview_dir=preview_dir,
        )
        first_summary = score_files(
            connection,
            learned_backend_name="topiq_nr",
            learned_backend_factory=lambda model_name: FirstBackend(),
        )
        first_row = connection.execute(
            "SELECT overall_score, model_version FROM scores"
        ).fetchone()

        second_summary = score_files(
            connection,
            learned_backend_name="topiq_nr",
            learned_backend_factory=lambda model_name: SecondBackend(),
            learned_model_version_resolver=lambda model_name: "learned:fake:v2",
        )
        second_row = connection.execute(
            "SELECT overall_score, learned_score_normalized, model_version FROM scores"
        ).fetchone()

    assert first_summary.files_scored == 1
    assert first_row["overall_score"] == 82.0
    assert first_row["model_version"] == "learned:fake:v1"
    assert second_summary.files_scored == 1
    assert second_row["overall_score"] == 63.0
    assert second_row["overall_score"] == second_row["learned_score_normalized"]
    assert second_row["model_version"] == "learned:fake:v2"


def test_score_skips_unchanged_rows_without_backend_reinitialization(tmp_path: Path, caplog) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "previews"
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    create_image(photo_dir / "sample.jpg")

    class FirstBackend:
        name = "topiq_nr"
        model_version = "fake:stable"

        def score_paths(self, image_paths, *, batch_size: int = 4, resource_profile: str | None = None):
            return [LearnedScoreResult(raw_score=0.82, normalized_score=82.0, confidence=91.0) for _ in image_paths]

    def fail_if_reinitialized(model_name: str):
        raise AssertionError("backend factory should not be called for unchanged rows")

    initialize_database(db_path)
    caplog.set_level("WARNING")

    with connect(db_path) as connection:
        scan_root(
            connection,
            root=photo_dir,
            recursive=True,
            extensions=(".jpg",),
            preview_dir=preview_dir,
        )
        first_summary = score_files(
            connection,
            learned_backend_name="topiq_nr",
            learned_backend_factory=lambda model_name: FirstBackend(),
        )

        second_summary = score_files(
            connection,
            learned_backend_name="topiq_nr",
            learned_backend_factory=fail_if_reinitialized,
        )

    assert first_summary.files_scored == 1
    assert second_summary.files_considered == 0
    assert second_summary.files_scored == 0
    assert second_summary.files_skipped == 0
    assert "same-backend model-version invalidation is skipped" in caplog.text


def test_score_rescores_unchanged_rows_when_version_resolver_fails(tmp_path: Path, caplog) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "previews"
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    create_image(photo_dir / "sample.jpg")

    class FirstBackend:
        name = "topiq_nr"
        model_version = "fake:stable"

        def score_paths(self, image_paths, *, batch_size: int = 4, resource_profile: str | None = None):
            return [LearnedScoreResult(raw_score=0.82, normalized_score=82.0, confidence=91.0) for _ in image_paths]

    class RefreshBackend:
        name = "topiq_nr"
        model_version = "fake:stable"

        def score_paths(self, image_paths, *, batch_size: int = 4, resource_profile: str | None = None):
            return [LearnedScoreResult(raw_score=0.61, normalized_score=61.0, confidence=90.0) for _ in image_paths]

    def fail_version_probe(model_name: str):
        raise RuntimeError("probe failed")

    initialize_database(db_path)
    caplog.set_level("WARNING")

    with connect(db_path) as connection:
        scan_root(
            connection,
            root=photo_dir,
            recursive=True,
            extensions=(".jpg",),
            preview_dir=preview_dir,
        )
        first_summary = score_files(
            connection,
            learned_backend_name="topiq_nr",
            learned_backend_factory=lambda model_name: FirstBackend(),
        )

        second_summary = score_files(
            connection,
            learned_backend_name="topiq_nr",
            learned_backend_factory=lambda model_name: RefreshBackend(),
            learned_model_version_resolver=fail_version_probe,
        )
        row = connection.execute(
            "SELECT overall_score FROM scores"
        ).fetchone()

    assert first_summary.files_scored == 1
    assert second_summary.files_considered == 1
    assert second_summary.files_scored == 1
    assert second_summary.files_skipped == 0
    assert row["overall_score"] == 61.0
    assert "Falling back to rescoring same-backend cached rows" in caplog.text


def test_score_rescores_unchanged_rows_when_default_version_probe_fails(tmp_path: Path, monkeypatch, caplog) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "previews"
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    create_image(photo_dir / "sample.jpg")

    class FirstBackend:
        name = "topiq_nr"
        model_version = "fake:stable"

        def score_paths(self, image_paths, *, batch_size: int = 4, resource_profile: str | None = None):
            return [LearnedScoreResult(raw_score=0.82, normalized_score=82.0, confidence=91.0) for _ in image_paths]

    class RefreshBackend:
        name = "topiq_nr"
        model_version = "fake:stable"

        def score_paths(self, image_paths, *, batch_size: int = 4, resource_profile: str | None = None):
            return [LearnedScoreResult(raw_score=0.57, normalized_score=57.0, confidence=87.0) for _ in image_paths]

    def fail_default_probe(model_name: str, device: str | None = None):
        raise RuntimeError("probe failed")

    monkeypatch.setattr(scoring_module, "resolve_learned_model_version", fail_default_probe)
    monkeypatch.setattr(scoring_module, "build_learned_backend", lambda model_name, device=None: RefreshBackend())

    initialize_database(db_path)
    caplog.set_level("WARNING")

    with connect(db_path) as connection:
        scan_root(
            connection,
            root=photo_dir,
            recursive=True,
            extensions=(".jpg",),
            preview_dir=preview_dir,
        )
        first_summary = score_files(
            connection,
            learned_backend_name="topiq_nr",
            learned_backend_factory=lambda model_name: FirstBackend(),
        )

        second_summary = score_files(
            connection,
            learned_backend_name="topiq_nr",
        )
        row = connection.execute(
            "SELECT overall_score FROM scores"
        ).fetchone()

    assert first_summary.files_scored == 1
    assert second_summary.files_considered == 1
    assert second_summary.files_scored == 1
    assert row["overall_score"] == 57.0
    assert "Falling back to rescoring same-backend cached rows" in caplog.text


def test_score_rewrites_legacy_rows_missing_source_fingerprints(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "previews"
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    create_image(photo_dir / "sample.jpg")

    class FirstBackend:
        name = "topiq_nr"
        model_version = "fake:stable"

        def score_paths(self, image_paths, *, batch_size: int = 4, resource_profile: str | None = None):
            return [LearnedScoreResult(raw_score=0.82, normalized_score=82.0, confidence=91.0) for _ in image_paths]

    class RefreshBackend:
        name = "topiq_nr"
        model_version = "fake:stable"

        def score_paths(self, image_paths, *, batch_size: int = 4, resource_profile: str | None = None):
            return [LearnedScoreResult(raw_score=0.58, normalized_score=58.0, confidence=89.0) for _ in image_paths]

    initialize_database(db_path)

    with connect(db_path) as connection:
        scan_root(
            connection,
            root=photo_dir,
            recursive=True,
            extensions=(".jpg",),
            preview_dir=preview_dir,
        )
        first_summary = score_files(
            connection,
            learned_backend_name="topiq_nr",
            learned_backend_factory=lambda model_name: FirstBackend(),
            learned_model_version_resolver=lambda model_name: "fake:stable",
        )

        connection.execute(
            "UPDATE scores SET source_modified_time = NULL, source_size_bytes = NULL"
        )

        second_summary = score_files(
            connection,
            learned_backend_name="topiq_nr",
            learned_backend_factory=lambda model_name: RefreshBackend(),
            learned_model_version_resolver=lambda model_name: "fake:stable",
        )
        row = connection.execute(
            "SELECT overall_score, source_modified_time, source_size_bytes FROM scores"
        ).fetchone()

    assert first_summary.files_scored == 1
    assert second_summary.files_scored == 1
    assert row["overall_score"] == 58.0
    assert row["source_modified_time"] is not None
    assert row["source_size_bytes"] is not None


def test_score_refreshes_when_source_file_changes(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "previews"
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    sample_path = photo_dir / "sample.jpg"
    create_image(sample_path)

    score_values = iter([82.0, 47.0])

    class FakeLearnedBackend:
        name = "topiq_nr"
        model_version = "fake:topiq_nr"

        def score_paths(self, image_paths, *, batch_size: int = 4, resource_profile: str | None = None):
            score = next(score_values)
            return [LearnedScoreResult(raw_score=score / 100.0, normalized_score=score, confidence=91.0) for _ in image_paths]

    initialize_database(db_path)

    with connect(db_path) as connection:
        scan_root(
            connection,
            root=photo_dir,
            recursive=True,
            extensions=(".jpg",),
            preview_dir=preview_dir,
        )
        first_summary = score_files(
            connection,
            learned_backend_name="topiq_nr",
            learned_backend_factory=lambda model_name: FakeLearnedBackend(),
        )
        first_row = connection.execute(
            "SELECT overall_score, model_version FROM scores"
        ).fetchone()

        initial_stat = sample_path.stat()
        Image.new("RGB", (180, 120), color=(180, 70, 40)).save(sample_path, format="JPEG", quality=95)
        os.utime(sample_path, (initial_stat.st_atime + 5, initial_stat.st_mtime + 5))

        rescan_summary = scan_root(
            connection,
            root=photo_dir,
            recursive=True,
            extensions=(".jpg",),
            preview_dir=preview_dir,
        )
        second_summary = score_files(
            connection,
            learned_backend_name="topiq_nr",
            learned_backend_factory=lambda model_name: FakeLearnedBackend(),
        )
        second_row = connection.execute(
            "SELECT overall_score, learned_score_normalized, model_version FROM scores"
        ).fetchone()

    assert first_summary.files_scored == 1
    assert first_row["overall_score"] == 82.0
    assert rescan_summary.files_updated == 1
    assert second_summary.files_scored == 1
    assert second_row["overall_score"] == 47.0
    assert second_row["overall_score"] == second_row["learned_score_normalized"]
    assert second_row["model_version"] == first_row["model_version"]


def test_score_removes_stale_score_row_when_rescore_fails(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "previews"
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    create_image(photo_dir / "sample.jpg")

    class SuccessfulBackend:
        name = "topiq_nr"
        model_version = "fake:v1"

        def score_paths(self, image_paths, *, batch_size: int = 4, resource_profile: str | None = None):
            return [LearnedScoreResult(raw_score=0.82, normalized_score=82.0, confidence=91.0) for _ in image_paths]

    class FailingBackend:
        name = "topiq_nr"
        model_version = "fake:v2"

        def score_paths(self, image_paths, *, batch_size: int = 4, resource_profile: str | None = None):
            return [
                LearnedScoreResult(
                    raw_score=None,
                    normalized_score=None,
                    confidence=None,
                    error="forced backend failure",
                )
                for _ in image_paths
            ]

    initialize_database(db_path)

    with connect(db_path) as connection:
        scan_root(
            connection,
            root=photo_dir,
            recursive=True,
            extensions=(".jpg",),
            preview_dir=preview_dir,
        )
        first_summary = score_files(
            connection,
            learned_backend_name="topiq_nr",
            learned_backend_factory=lambda model_name: SuccessfulBackend(),
        )

        second_summary = score_files(
            connection,
            learned_backend_name="topiq_nr",
            learned_backend_factory=lambda model_name: FailingBackend(),
            learned_model_version_resolver=lambda model_name: "learned:fake:v2",
        )
        score_count = connection.execute(
            "SELECT COUNT(*) AS count FROM scores"
        ).fetchone()["count"]

    assert first_summary.files_scored == 1
    assert second_summary.files_considered == 1
    assert second_summary.files_scored == 0
    assert second_summary.learned_scored == 0
    assert second_summary.files_failed == 1
    assert score_count == 0


def test_score_removes_stale_score_row_when_preview_refresh_fails_without_fallback(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "previews"
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    raw_path = photo_dir / "sample.cr2"
    raw_path.write_bytes(b"fake-raw")

    def fake_generate_previews_parallel(source_paths, generated_preview_dir: Path, *, max_workers=None, progress_callback=None):
        assert source_paths == [raw_path]
        assert generated_preview_dir == preview_dir
        return [
            preview_module.PreviewResult(
                path=None,
                status="failed",
                width=None,
                height=None,
                capture_time=None,
                error_text="forced preview failure",
            )
        ]

    monkeypatch.setattr(scoring_module, "generate_previews_parallel", fake_generate_previews_parallel)

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
        file_row = connection.execute("SELECT id FROM files").fetchone()
        connection.execute(
            """
            INSERT INTO scores(
                file_id,
                overall_score,
                learned_backend,
                learned_raw_score,
                learned_score_normalized,
                learned_confidence,
                source_modified_time,
                source_size_bytes,
                preset_name,
                model_version,
                computed_time
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                file_row["id"],
                82.0,
                "topiq_nr",
                0.82,
                82.0,
                91.0,
                0.0,
                raw_path.stat().st_size - 1,
                "learned-only",
                "learned:fake:v1",
                "2026-04-19T00:00:00+00:00",
            ),
        )

        summary = score_files(
            connection,
            learned_backend_name="topiq_nr",
            preview_dir=preview_dir,
        )
        score_count = connection.execute(
            "SELECT COUNT(*) AS count FROM scores"
        ).fetchone()["count"]

    assert summary.files_considered == 1
    assert summary.files_scored == 0
    assert summary.files_failed == 1
    assert score_count == 0


def test_compare_learned_models_returns_side_by_side_rows(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "previews"
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    create_image(photo_dir / "sample.jpg")

    perf_values = iter([10.0, 10.0, 11.5, 11.5, 13.0, 13.0])
    monkeypatch.setattr(scoring_module.time, "perf_counter", lambda: next(perf_values))

    class FakeLearnedBackend:
        def __init__(self, name: str, score: float, confidence: float) -> None:
            self.name = name
            self.model_version = f"fake:{name}"
            self._score = score
            self._confidence = confidence

        def score_paths(self, image_paths, *, batch_size: int = 4, resource_profile: str | None = None):
            return [LearnedScoreResult(raw_score=self._score / 100.0, normalized_score=self._score, confidence=self._confidence) for _ in image_paths]

    initialize_database(db_path)

    with connect(db_path) as connection:
        scan_root(
            connection,
            root=photo_dir,
            recursive=True,
            extensions=(".jpg",),
            preview_dir=preview_dir,
        )

        comparison = compare_learned_models(
            connection,
            model_names=["topiq_nr", "arniqa"],
            learned_backend_factory=lambda model_name: {
                "topiq_nr": FakeLearnedBackend("topiq_nr", 82.0, 91.0),
                "arniqa": FakeLearnedBackend("arniqa", 74.0, 85.0),
            }[model_name],
        )

    assert comparison.files_considered == 1
    assert comparison.files_compared == 1
    assert comparison.files_skipped == 0
    assert comparison.model_names == ["topiq_nr", "arniqa"]
    assert comparison.elapsed_seconds == 3.0
    assert comparison.model_timings_seconds == {"topiq_nr": 1.5, "arniqa": 1.5}
    assert len(comparison.rows) == 1
    row = comparison.rows[0]
    assert isinstance(row["file_id"], int)
    assert isinstance(row["path"], str)
    assert row["path"].endswith("sample.jpg")
    assert row["topiq_nr_score"] == 82.0
    assert row["topiq_nr_confidence"] == 91.0
    assert row["arniqa_score"] == 74.0
    assert row["arniqa_confidence"] == 85.0


def test_compare_learned_models_reports_failed_results_without_fake_scores(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "previews"
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    create_image(photo_dir / "sample.jpg")

    class FailingBackend:
        name = "topiq_nr"
        model_version = "fake:topiq_nr"

        def score_paths(self, image_paths, *, batch_size: int = 4, resource_profile: str | None = None):
            return [
                LearnedScoreResult(
                    raw_score=None,
                    normalized_score=None,
                    confidence=None,
                    error="forced comparison failure",
                )
                for _ in image_paths
            ]

    initialize_database(db_path)

    with connect(db_path) as connection:
        scan_root(
            connection,
            root=photo_dir,
            recursive=True,
            extensions=(".jpg",),
            preview_dir=preview_dir,
        )

        comparison = compare_learned_models(
            connection,
            model_names=["topiq_nr"],
            learned_backend_factory=lambda model_name: FailingBackend(),
        )

    assert comparison.files_compared == 1
    assert comparison.files_failed == 1
    assert len(comparison.rows) == 1
    row = comparison.rows[0]
    assert row["topiq_nr_score"] is None
    assert row["topiq_nr_confidence"] is None
    assert row["topiq_nr_raw"] is None
    assert row["topiq_nr_error"] == "forced comparison failure"


def test_compare_learned_models_counts_failed_files_once_when_multiple_models_fail(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "previews"
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    create_image(photo_dir / "sample.jpg")

    class FailingBackend:
        def __init__(self, model_name: str) -> None:
            self.name = model_name
            self.model_version = f"fake:{model_name}"

        def score_paths(self, image_paths, *, batch_size: int = 4, resource_profile: str | None = None):
            return [
                LearnedScoreResult(
                    raw_score=None,
                    normalized_score=None,
                    confidence=None,
                    error=f"forced {self.name} failure",
                )
                for _ in image_paths
            ]

    initialize_database(db_path)

    with connect(db_path) as connection:
        scan_root(
            connection,
            root=photo_dir,
            recursive=True,
            extensions=(".jpg",),
            preview_dir=preview_dir,
        )

        comparison = compare_learned_models(
            connection,
            model_names=["topiq_nr", "arniqa"],
            learned_backend_factory=lambda model_name: FailingBackend(model_name),
        )

    assert comparison.files_compared == 1
    assert comparison.files_failed == 1


def test_compare_learned_models_releases_backend_before_loading_next(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "previews"
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    create_image(photo_dir / "sample.jpg")

    active_backends = 0

    class ExclusiveBackend:
        def __init__(self, model_name: str) -> None:
            nonlocal active_backends
            active_backends += 1
            if active_backends > 1:
                raise RuntimeError("multiple learned backends active at once")
            self.name = model_name
            self.model_version = f"fake:{model_name}"

        def score_paths(self, image_paths, *, batch_size: int = 4, resource_profile: str | None = None):
            return [LearnedScoreResult(raw_score=0.75, normalized_score=75.0, confidence=80.0) for _ in image_paths]

        def close(self) -> None:
            nonlocal active_backends
            active_backends = max(0, active_backends - 1)

    initialize_database(db_path)

    with connect(db_path) as connection:
        scan_root(
            connection,
            root=photo_dir,
            recursive=True,
            extensions=(".jpg",),
            preview_dir=preview_dir,
        )

        comparison = compare_learned_models(
            connection,
            model_names=["topiq_nr", "arniqa"],
            learned_backend_factory=lambda model_name: ExclusiveBackend(model_name),
        )

    assert comparison.files_compared == 1
    assert comparison.model_names == ["topiq_nr", "arniqa"]
    assert active_backends == 0


def test_compare_learned_models_reports_progress_per_model_and_chunk(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "previews"
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    for index in range(5):
        create_image(photo_dir / f"sample-{index}.jpg")

    class FakeLearnedBackend:
        def __init__(self, model_name: str, score: float) -> None:
            self.name = model_name
            self.model_version = f"fake:{model_name}"
            self._score = score

        def score_paths(self, image_paths, *, batch_size: int = 4, resource_profile: str | None = None):
            return [LearnedScoreResult(raw_score=self._score / 100.0, normalized_score=self._score, confidence=88.0) for _ in image_paths]

    progress_updates: list[AnalysisProgress] = []

    initialize_database(db_path)

    with connect(db_path) as connection:
        scan_root(
            connection,
            root=photo_dir,
            recursive=True,
            extensions=(".jpg",),
            preview_dir=preview_dir,
        )

        comparison = compare_learned_models(
            connection,
            model_names=["topiq_nr", "arniqa"],
            learned_batch_size=2,
            compare_chunk_size=2,
            progress_callback=progress_updates.append,
            learned_backend_factory=lambda model_name: {
                "topiq_nr": FakeLearnedBackend("topiq_nr", 82.0),
                "arniqa": FakeLearnedBackend("arniqa", 74.0),
            }[model_name],
        )

    assert comparison.files_compared == 5
    assert comparison.model_names == ["topiq_nr", "arniqa"]
    assert len(progress_updates) == 10

    topiq_updates = [update for update in progress_updates if update.model_name == "topiq_nr"]
    arniqa_updates = [update for update in progress_updates if update.model_name == "arniqa"]

    assert [update.phase for update in topiq_updates] == ["loading", "scoring", "scoring", "scoring", "scoring"]
    assert [update.phase for update in arniqa_updates] == ["loading", "scoring", "scoring", "scoring", "scoring"]
    assert [update.files_processed for update in topiq_updates] == [0, 0, 2, 4, 5]
    assert [update.files_processed for update in arniqa_updates] == [0, 0, 2, 4, 5]
    assert topiq_updates[0].model_index == 1
    assert topiq_updates[0].model_count == 2
    assert topiq_updates[-1].files_total == 5


def test_compare_learned_models_reports_truncation_contract_when_max_rows_caps_results(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "previews"
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    for index in range(3):
        create_image(photo_dir / f"sample-{index}.jpg")

    monkeypatch.setattr(scoring_module, "COMPARE_MAX_ROWS", 2, raising=False)

    class FakeLearnedBackend:
        name = "topiq_nr"
        model_version = "fake:topiq_nr"

        def score_paths(self, image_paths, *, batch_size: int = 4, resource_profile: str | None = None):
            return [
                LearnedScoreResult(raw_score=0.82, normalized_score=82.0, confidence=91.0)
                for _ in image_paths
            ]

    initialize_database(db_path)

    with connect(db_path) as connection:
        scan_root(
            connection,
            root=photo_dir,
            recursive=True,
            extensions=(".jpg",),
            preview_dir=preview_dir,
        )

        comparison = compare_learned_models(
            connection,
            model_names=["topiq_nr"],
            learned_backend_factory=lambda model_name: FakeLearnedBackend(),
        )

    assert comparison.requested_rows_total == 3
    assert comparison.processed_rows_total == 2
    assert comparison.truncated is True
    assert comparison.max_rows == 2
    assert comparison.files_considered == 2
    assert comparison.files_compared == 2


def test_compare_learned_models_can_keep_backends_loaded(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "previews"
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    create_image(photo_dir / "sample.jpg")

    close_calls = 0

    class ClosableBackend:
        def __init__(self, model_name: str) -> None:
            self.name = model_name
            self.model_version = f"fake:{model_name}"

        def score_paths(self, image_paths, *, batch_size: int = 4, resource_profile: str | None = None):
            return [LearnedScoreResult(raw_score=0.75, normalized_score=75.0, confidence=80.0) for _ in image_paths]

        def close(self) -> None:
            nonlocal close_calls
            close_calls += 1

    initialize_database(db_path)

    with connect(db_path) as connection:
        scan_root(
            connection,
            root=photo_dir,
            recursive=True,
            extensions=(".jpg",),
            preview_dir=preview_dir,
        )

        comparison = compare_learned_models(
            connection,
            model_names=["topiq_nr", "arniqa"],
            learned_backend_factory=lambda model_name: ClosableBackend(model_name),
            release_backends=False,
        )

    assert comparison.files_compared == 1
    assert close_calls == 0


def test_score_files_releases_backend_after_scoring(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "previews"
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    create_image(photo_dir / "sample.jpg")

    close_calls = 0

    class ClosableBackend:
        name = "topiq_nr"
        model_version = "fake:topiq_nr"

        def score_paths(self, image_paths, *, batch_size: int = 4, resource_profile: str | None = None):
            return [LearnedScoreResult(raw_score=0.82, normalized_score=82.0, confidence=91.0) for _ in image_paths]

        def close(self) -> None:
            nonlocal close_calls
            close_calls += 1

    initialize_database(db_path)

    with connect(db_path) as connection:
        scan_root(
            connection,
            root=photo_dir,
            recursive=True,
            extensions=(".jpg",),
            preview_dir=preview_dir,
        )

        summary = score_files(
            connection,
            learned_backend_name="topiq_nr",
            learned_backend_factory=lambda model_name: ClosableBackend(),
        )

    assert summary.files_scored == 1
    assert close_calls == 1





def test_select_analysis_path_accepts_pil_native_formats_and_rejects_optional_deps(tmp_path: Path) -> None:
    heif_file = tmp_path / "photo.heif"
    heif_file.write_bytes(b"fake-heif-data")
    webp_file = tmp_path / "photo.webp"
    webp_file.write_bytes(b"fake-webp-data")
    heic_file = tmp_path / "photo.heic"
    heic_file.write_bytes(b"fake-heic-data")
    cr2_file = tmp_path / "photo.cr2"
    cr2_file.write_bytes(b"fake-cr2-data")

    # .webp is PIL-native — should be accepted for direct analysis.
    assert select_analysis_path(str(webp_file), None, None) == webp_file

    # .heif/.heic require pillow_heif which may not be installed — should be
    # rejected to avoid the silent 50.0 fallback score poisoning.
    assert select_analysis_path(str(heif_file), None, None) is None
    assert select_analysis_path(str(heic_file), None, None) is None

    # RAW formats always need a generated preview for analysis.
    assert select_analysis_path(str(cr2_file), None, None) is None


def test_score_can_generate_missing_preview_on_demand_for_preview_only_formats(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "previews"
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    raw_path = photo_dir / "sample.cr2"
    raw_path.write_bytes(b"fake-raw")

    ready_preview = preview_dir / "sample-preview.jpg"
    preview_dir.mkdir(parents=True, exist_ok=True)
    create_image(ready_preview)

    class FakeLearnedBackend:
        name = "topiq_nr"
        model_version = "fake:topiq_nr"

        def score_paths(self, image_paths, *, batch_size: int = 4, resource_profile: str | None = None):
            return [LearnedScoreResult(raw_score=0.82, normalized_score=82.0, confidence=91.0) for _ in image_paths]

    def fake_generate_preview(source_path: Path, generated_preview_dir: Path):
        assert source_path == raw_path
        assert generated_preview_dir == preview_dir
        return preview_module.PreviewResult(
            path=str(ready_preview),
            status="ready",
            width=120,
            height=80,
            capture_time=None,
            error_text=None,
        )

    monkeypatch.setattr(preview_module, "generate_preview", fake_generate_preview)

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

        summary = score_files(
            connection,
            learned_backend_name="topiq_nr",
            preview_dir=preview_dir,
            learned_backend_factory=lambda model_name: FakeLearnedBackend(),
        )
        row = connection.execute(
            """
            SELECT f.preview_status, f.preview_path, s.overall_score
            FROM files f
            LEFT JOIN scores s ON s.file_id = f.id
            LIMIT 1
            """
        ).fetchone()

    assert summary.files_considered == 1
    assert summary.files_scored == 1
    assert summary.files_skipped == 0
    assert row["preview_status"] == "ready"
    assert Path(row["preview_path"]).resolve() == ready_preview.resolve()
    assert row["overall_score"] == 82.0


def test_score_generates_missing_tiff_preview_after_fast_scan(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "previews"
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()

    tiff_path = photo_dir / "sample.tiff"
    Image.new("RGB", (120, 80), color=(35, 80, 145)).save(tiff_path, format="TIFF")

    ready_preview = preview_dir / "sample-tiff-preview.jpg"
    preview_dir.mkdir(parents=True, exist_ok=True)
    create_image(ready_preview)

    class FakeLearnedBackend:
        name = "topiq_nr"
        model_version = "fake:topiq_nr"

        def score_paths(self, image_paths, *, batch_size: int = 4, resource_profile: str | None = None):
            return [LearnedScoreResult(raw_score=0.83, normalized_score=83.0, confidence=92.0) for _ in image_paths]

    def fake_generate_previews_parallel(source_paths, generated_preview_dir: Path, *, max_workers=None, progress_callback=None):
        assert source_paths == [tiff_path]
        assert generated_preview_dir == preview_dir
        if progress_callback is not None:
            progress_callback(1, 1)
        return [
            preview_module.PreviewResult(
                path=str(ready_preview),
                status="ready",
                width=120,
                height=80,
                capture_time=None,
                error_text=None,
            )
        ]

    monkeypatch.setattr(scoring_module, "generate_previews_parallel", fake_generate_previews_parallel)

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

        summary = score_files(
            connection,
            learned_backend_name="topiq_nr",
            preview_dir=preview_dir,
            learned_backend_factory=lambda model_name: FakeLearnedBackend(),
        )
        row = connection.execute(
            """
            SELECT f.preview_status, f.preview_path, s.overall_score
            FROM files f
            LEFT JOIN scores s ON s.file_id = f.id
            LIMIT 1
            """
        ).fetchone()

    assert summary.files_considered == 1
    assert summary.files_scored == 1
    assert summary.files_skipped == 0
    assert row["preview_status"] == "ready"
    assert Path(row["preview_path"]).resolve() == ready_preview.resolve()
    assert row["overall_score"] == 83.0


def test_default_preview_workers_uses_higher_parallelism_budget(monkeypatch) -> None:
    import shotsieve.learned_iqa as learned_iqa_module

    monkeypatch.setattr(learned_iqa_module, "_effective_cpu_count", lambda: 16)
    # Mock RAM to 5024 MB so the RAM-based cap constrains aggressive mode
    # to 30 workers: (5024 - 1024) * 0.75 / 100 = 30
    monkeypatch.setattr(learned_iqa_module, "detect_system_ram_mb", lambda: 5024)
    # Clear the hardware capabilities cache so the mocked RAM takes effect.
    monkeypatch.setattr(learned_iqa_module, "_cached_hw_capabilities", None)

    assert scoring_module._default_preview_workers("aggressive") == 30
    assert scoring_module._default_preview_workers("normal") == 16
    assert scoring_module._default_preview_workers("low") == 8


def test_score_files_reports_loading_phase_before_first_scoring_update(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "previews"
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    create_image(photo_dir / "sample.jpg")

    progress_updates: list[AnalysisProgress] = []

    class FakeLearnedBackend:
        name = "topiq_nr"
        model_version = "fake:topiq_nr"

        def score_paths(self, image_paths, *, batch_size: int = 4, resource_profile: str | None = None):
            return [LearnedScoreResult(raw_score=0.82, normalized_score=82.0, confidence=91.0) for _ in image_paths]

    initialize_database(db_path)

    with connect(db_path) as connection:
        scan_root(
            connection,
            root=photo_dir,
            recursive=True,
            extensions=(".jpg",),
            preview_dir=preview_dir,
        )

        summary = score_files(
            connection,
            learned_backend_name="topiq_nr",
            learned_backend_factory=lambda model_name: FakeLearnedBackend(),
            progress_callback=progress_updates.append,
        )

    assert summary.files_scored == 1
    assert progress_updates
    assert progress_updates[0].phase == "loading"
    assert progress_updates[0].files_processed == 0
    assert progress_updates[0].files_total == 1
    assert progress_updates[-1].phase == "scoring"
    assert progress_updates[-1].files_processed == 1


def test_score_files_emits_preview_phase_zero_progress_before_parallel_work(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "previews"
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()

    tiff_path = photo_dir / "sample.tiff"
    Image.new("RGB", (120, 80), color=(35, 80, 145)).save(tiff_path, format="TIFF")

    ready_preview = preview_dir / "sample-tiff-preview.jpg"
    preview_dir.mkdir(parents=True, exist_ok=True)
    create_image(ready_preview)

    progress_updates: list[AnalysisProgress] = []

    class FakeLearnedBackend:
        name = "topiq_nr"
        model_version = "fake:topiq_nr"

        def score_paths(self, image_paths, *, batch_size: int = 4, resource_profile: str | None = None):
            return [LearnedScoreResult(raw_score=0.83, normalized_score=83.0, confidence=92.0) for _ in image_paths]

    def fake_generate_previews_parallel(source_paths, generated_preview_dir: Path, *, max_workers=None, progress_callback=None):
        assert source_paths == [tiff_path]
        assert generated_preview_dir == preview_dir
        assert progress_callback is not None
        return [
            preview_module.PreviewResult(
                path=str(ready_preview),
                status="ready",
                width=120,
                height=80,
                capture_time=None,
                error_text=None,
            )
        ]

    monkeypatch.setattr(scoring_module, "generate_previews_parallel", fake_generate_previews_parallel)

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

        summary = score_files(
            connection,
            learned_backend_name="topiq_nr",
            preview_dir=preview_dir,
            learned_backend_factory=lambda model_name: FakeLearnedBackend(),
            progress_callback=progress_updates.append,
        )

    assert summary.files_scored == 1
    assert any(
        update.phase == "generating_previews"
        and update.files_processed == 0
        and update.files_total == 1
        for update in progress_updates
    )


def test_compare_learned_models_emits_preview_phase_zero_progress_before_parallel_work(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "previews"
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()

    tiff_path = photo_dir / "sample.tiff"
    Image.new("RGB", (120, 80), color=(35, 80, 145)).save(tiff_path, format="TIFF")

    ready_preview = preview_dir / "sample-tiff-preview.jpg"
    preview_dir.mkdir(parents=True, exist_ok=True)
    create_image(ready_preview)

    progress_updates: list[AnalysisProgress] = []

    class FakeLearnedBackend:
        name = "topiq_nr"
        model_version = "fake:topiq_nr"

        def score_paths(self, image_paths, *, batch_size: int = 4, resource_profile: str | None = None):
            return [LearnedScoreResult(raw_score=0.83, normalized_score=83.0, confidence=92.0) for _ in image_paths]

    def fake_generate_previews_parallel(source_paths, generated_preview_dir: Path, *, max_workers=None, progress_callback=None):
        assert source_paths == [tiff_path]
        assert generated_preview_dir == preview_dir
        assert progress_callback is not None
        return [
            preview_module.PreviewResult(
                path=str(ready_preview),
                status="ready",
                width=120,
                height=80,
                capture_time=None,
                error_text=None,
            )
        ]

    monkeypatch.setattr(scoring_module, "generate_previews_parallel", fake_generate_previews_parallel)

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

        comparison = compare_learned_models(
            connection,
            model_names=["topiq_nr"],
            preview_dir=preview_dir,
            learned_backend_factory=lambda model_name: FakeLearnedBackend(),
            progress_callback=progress_updates.append,
        )

    assert comparison.files_compared == 1
    assert any(
        update.phase == "generating_previews"
        and update.files_processed == 0
        and update.files_total == 1
        for update in progress_updates
    )


def create_image(path: Path) -> None:
    image = Image.new("RGB", (120, 80), color=(40, 90, 160))
    image.save(path, format="JPEG")


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