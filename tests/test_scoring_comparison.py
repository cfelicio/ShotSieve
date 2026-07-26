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
from shotsieve.scoring import AnalysisProgress, compare_learned_models


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