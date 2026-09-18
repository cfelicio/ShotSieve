from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import warnings

import numpy as np
from PIL import Image
import pytest

from shotsieve import preview as preview_module
from shotsieve import scanner as scanner_module
from shotsieve import learned_iqa_preprocessing as preprocessing_module
from shotsieve.db import connect, initialize_database
from shotsieve.image_conversion import (
    DEFAULT_MAX_DECODE_PIXELS,
    enforce_decode_budget,
    IMAGE_CONVERSION_VERSION,
    ImageDecodeLimitError,
    TRANSPARENCY_MATTE,
    prepare_image_for_rgb,
)
from shotsieve.learned_iqa import LearnedScoreResult
from shotsieve.learned_iqa_preprocessing import _load_single_image
from shotsieve.scanner import scan_root
from shotsieve.scoring import score_files
from shotsieve.schema import SCHEMA_SQL


def test_rgba_and_palette_transparency_are_composited_on_the_same_white_matte() -> None:
    rgba = Image.new("RGBA", (2, 1))
    rgba.putdata([(255, 0, 0, 0), (0, 255, 0, 128)])

    palette = Image.new("P", (2, 1))
    palette.putpalette([255, 0, 0, 0, 255, 0] + [0] * (256 * 3 - 6))
    palette.putdata([0, 1])
    palette.info["transparency"] = bytes([0, 128])

    expected = [(255, 255, 255), (127, 255, 127)]
    assert TRANSPARENCY_MATTE == (255, 255, 255)
    rgba_converted = prepare_image_for_rgb(rgba)
    palette_converted = prepare_image_for_rgb(palette)
    assert rgba_converted.getpixel((0, 0)) == expected[0]
    assert rgba_converted.getpixel((1, 0)) == expected[1]
    assert palette_converted.getpixel((0, 0)) == expected[0]
    assert palette_converted.getpixel((1, 0)) == expected[1]


def test_decode_budget_defaults_to_64_megapixels_and_can_be_lowered_per_call() -> None:
    assert DEFAULT_MAX_DECODE_PIXELS == 64_000_000
    enforce_decode_budget(Path("phone.jpg"), 6_048, 8_064)

    with pytest.raises(ImageDecodeLimitError, match="safe decode budget"):
        enforce_decode_budget(Path("phone.jpg"), 6_048, 8_064, max_pixels=40_000_000)


def test_scoring_preprocessing_matches_shared_conversion_for_transparent_png(tmp_path: Path) -> None:
    source_path = tmp_path / "transparent.png"
    source = Image.new("RGBA", (2, 2))
    source.putdata([(255, 0, 0, 0), (0, 0, 255, 255), (0, 255, 0, 128), (10, 20, 30, 255)])
    source.save(source_path)

    with Image.open(source_path) as opened:
        expected = np.asarray(prepare_image_for_rgb(opened), dtype=np.float32) / 255.0

    actual = _load_single_image(source_path, image_size=2)

    np.testing.assert_allclose(actual, expected)


def test_preview_uses_shared_transparency_conversion_and_records_current_version(tmp_path: Path) -> None:
    source_path = tmp_path / "transparent.png"
    preview_dir = tmp_path / "previews"
    source = Image.new("RGBA", (32, 16), color=(255, 0, 0, 0))
    for x in range(16, 32):
        for y in range(16):
            source.putpixel((x, y), (0, 0, 255, 255))
    source.save(source_path)

    result = preview_module.generate_preview(source_path, preview_dir)

    assert result.status == "ready"
    assert result.path is not None
    with Image.open(result.path) as generated:
        transparent_pixel = generated.getpixel((8, 8))
        opaque_pixel = generated.getpixel((24, 8))

    assert all(channel >= 245 for channel in transparent_pixel)
    assert opaque_pixel[2] >= 245
    assert IMAGE_CONVERSION_VERSION == "rgba-white-matte-v1"


def test_shared_conversion_preserves_exif_orientation_and_high_bit_grayscale() -> None:
    oriented = Image.new("RGB", (2, 1))
    oriented.putdata([(255, 0, 0), (0, 0, 255)])
    exif = Image.Exif()
    exif[274] = 6
    oriented.info["exif"] = exif.tobytes()

    assert prepare_image_for_rgb(oriented).size == (1, 2)

    grayscale = Image.new("I;16", (2, 1))
    grayscale.putdata([0, 32768])
    converted = prepare_image_for_rgb(grayscale)
    assert converted.getpixel((0, 0)) == (0, 0, 0)
    midpoint = converted.getpixel((1, 0))
    assert midpoint[0] == midpoint[1] == midpoint[2]
    assert 126 <= midpoint[0] <= 129


def test_scan_regenerates_legacy_preview_conversion_without_touching_review_state(
    tmp_path: Path,
    monkeypatch,
) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "previews"
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    source_path = photo_dir / "sample.png"
    Image.new("RGBA", (2, 1), color=(255, 0, 0, 0)).save(source_path)
    initialize_database(db_path)

    with connect(db_path) as connection:
        scan_root(connection, root=photo_dir, extensions=(".png",), preview_dir=preview_dir)
        row = connection.execute("SELECT id, preview_path FROM files").fetchone()
        connection.execute(
            "INSERT INTO review_state(file_id, decision_state, updated_time) VALUES (?, 'keep', 'now')",
            (row["id"],),
        )
        connection.execute(
            "UPDATE files SET preview_conversion_version = 'legacy-preview-v0' WHERE id = ?",
            (row["id"],),
        )

        calls: list[Path] = []

        def regenerate(path: Path, generated_preview_dir: Path, *, raw_preview_mode: str = "auto"):
            calls.append(path)
            return preview_module.PreviewResult(
                path=str(Path(row["preview_path"])),
                status="ready",
                width=2,
                height=1,
                capture_time=None,
            )

        monkeypatch.setattr(scanner_module, "generate_preview", regenerate)
        scan_root(connection, root=photo_dir, extensions=(".png",), preview_dir=preview_dir)

        refreshed = connection.execute(
            "SELECT preview_conversion_version FROM files WHERE id = ?",
            (row["id"],),
        ).fetchone()
        decision = connection.execute(
            "SELECT decision_state FROM review_state WHERE file_id = ?",
            (row["id"],),
        ).fetchone()

    assert calls == [source_path]
    assert refreshed["preview_conversion_version"] == IMAGE_CONVERSION_VERSION
    assert decision["decision_state"] == "keep"


def test_scoring_rescores_when_image_conversion_version_is_legacy(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "previews"
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    Image.new("RGBA", (2, 1), color=(255, 0, 0, 0)).save(photo_dir / "sample.png")
    initialize_database(db_path)

    score_values = iter((82.0, 47.0))

    class FakeBackend:
        name = "topiq_nr"
        model_version = "fake:conversion"

        def score_paths(self, image_paths, *, batch_size: int = 4, resource_profile: str | None = None):
            score = next(score_values)
            return [
                LearnedScoreResult(
                    raw_score=score / 100.0,
                    normalized_score=score,
                    confidence=91.0,
                )
                for _ in image_paths
            ]

    with connect(db_path) as connection:
        scan_root(connection, root=photo_dir, extensions=(".png",), preview_dir=preview_dir)
        first = score_files(
            connection,
            learned_backend_name="topiq_nr",
            learned_backend_factory=lambda _model_name: FakeBackend(),
        )
        connection.execute("UPDATE scores SET image_conversion_version = 'legacy-score-v0'")
        second = score_files(
            connection,
            learned_backend_name="topiq_nr",
            learned_backend_factory=lambda _model_name: FakeBackend(),
        )
        row = connection.execute(
            "SELECT overall_score, image_conversion_version FROM scores"
        ).fetchone()

    assert first.files_scored == 1
    assert second.files_scored == 1
    assert row["overall_score"] == 47.0
    assert row["image_conversion_version"] == IMAGE_CONVERSION_VERSION


def test_database_migrates_conversion_cache_columns(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    db_path.parent.mkdir()
    legacy_schema = SCHEMA_SQL.replace("    preview_conversion_version TEXT,\n", "").replace(
        "    image_conversion_version TEXT,\n", ""
    )
    with sqlite3.connect(db_path) as connection:
        connection.executescript(legacy_schema)

    initialize_database(db_path)

    with connect(db_path) as connection:
        file_columns = {row["name"] for row in connection.execute("PRAGMA table_info(files)")}
        score_columns = {row["name"] for row in connection.execute("PRAGMA table_info(scores)")}

    assert "preview_conversion_version" in file_columns
    assert "image_conversion_version" in score_columns


def test_generate_preview_rejects_oversized_source_before_conversion(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_path = tmp_path / "oversized.jpg"
    source_path.write_bytes(b"header-only fixture")

    class HeaderOnlyImage:
        size = (DEFAULT_MAX_DECODE_PIXELS + 1, 1)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def getexif(self):
            return {}

    monkeypatch.setattr(preview_module.Image, "open", lambda _path: HeaderOnlyImage())
    monkeypatch.setattr(
        preview_module.ImageOps,
        "exif_transpose",
        lambda _image: pytest.fail("oversized source should not be converted"),
    )

    result = preview_module.generate_preview(source_path, tmp_path / "previews")

    assert result.status == "failed"
    assert result.path is None
    assert result.error_text is not None
    assert "safe decode budget" in result.error_text


def test_learned_preprocessing_rejects_oversized_source_before_conversion(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_path = tmp_path / "oversized.jpg"
    source_path.write_bytes(b"header-only fixture")

    class HeaderOnlyImage:
        size = (DEFAULT_MAX_DECODE_PIXELS + 1, 1)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(preprocessing_module.Image, "open", lambda _path: HeaderOnlyImage())
    monkeypatch.setattr(
        preprocessing_module,
        "prepare_image_for_rgb",
        lambda _image: pytest.fail("oversized source should not be converted"),
    )

    with pytest.raises(ImageDecodeLimitError, match="safe decode budget"):
        preprocessing_module._load_single_image(source_path, image_size=2)


def test_concurrent_image_warnings_are_attributed_to_the_opening_file(
    monkeypatch,
    caplog,
) -> None:
    source_paths = [Path("first.jpg"), Path("second.jpg")]

    class WarningImage:
        size = (2, 2)

        def __init__(self, source_path: Path) -> None:
            self.source_path = source_path

        def __enter__(self):
            warnings.warn(
                f"decoder warning for {self.source_path.name}",
                RuntimeWarning,
            )
            return Image.new("RGB", self.size, color="black")

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(
        preprocessing_module.Image,
        "open",
        lambda path: WarningImage(Path(path)),
    )

    caplog.set_level("WARNING", logger="shotsieve.learned_iqa_preprocessing")
    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(lambda path: preprocessing_module._load_single_image(path, 2), source_paths))

    warning_messages = [record.getMessage() for record in caplog.records]
    assert sum("first.jpg" in message for message in warning_messages) == 1
    assert sum("second.jpg" in message for message in warning_messages) == 1
