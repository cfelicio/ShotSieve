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
from shotsieve.scoring import AnalysisProgress, score_files, select_analysis_path


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




def test_prepare_analysis_candidates_prefers_generated_preview_and_returns_persistence_results(tmp_path: Path, monkeypatch) -> None:
    preview_dir = tmp_path / "previews"
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()

    tiff_path = photo_dir / "sample.tiff"
    Image.new("RGB", (120, 80), color=(35, 80, 145)).save(tiff_path, format="TIFF")

    ready_preview = preview_dir / "sample-tiff-preview.jpg"
    preview_dir.mkdir(parents=True, exist_ok=True)
    create_image(ready_preview)

    def fake_generate_previews_parallel(source_paths, generated_preview_dir: Path, *, max_workers=None, progress_callback=None, raw_preview_mode="auto"):
        assert source_paths == [tiff_path]
        assert generated_preview_dir == preview_dir
        assert max_workers == 7
        assert raw_preview_mode == "auto"
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
        raw_preview_mode="auto",
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

    def fake_generate_previews_parallel(source_paths, generated_preview_dir: Path, *, max_workers=None, progress_callback=None, raw_preview_mode="auto"):
        assert source_paths == [tiff_path, raw_path]
        assert generated_preview_dir == preview_dir
        assert raw_preview_mode == "auto"
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
        raw_preview_mode="auto",
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

    def fake_generate_preview(source_path: Path, generated_preview_dir: Path, *, raw_preview_mode: str = "auto"):
        assert source_path == raw_path
        assert generated_preview_dir == preview_dir
        assert raw_preview_mode == "auto"
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

    def fake_generate_previews_parallel(source_paths, generated_preview_dir: Path, *, max_workers=None, progress_callback=None, raw_preview_mode="auto"):
        assert source_paths == [tiff_path]
        assert generated_preview_dir == preview_dir
        if progress_callback is not None:
            progress_callback(1, 1)
        assert raw_preview_mode == "auto"
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

    def fake_generate_previews_parallel(source_paths, generated_preview_dir: Path, *, max_workers=None, progress_callback=None, raw_preview_mode="auto"):
        assert source_paths == [tiff_path]
        assert generated_preview_dir == preview_dir
        assert progress_callback is not None
        assert raw_preview_mode == "auto"
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