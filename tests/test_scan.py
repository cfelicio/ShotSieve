"""Tests for scan lifecycle, preview invalidation, and scan path handling."""
from __future__ import annotations

from pathlib import Path

from PIL import Image
import pytest

from shotsieve.db import connect, database, initialize_database
from shotsieve.models import ScanSummary
from shotsieve.preview import PreviewResult
from shotsieve.scanner import FileDiscoveryError, ScanInterrupted, _process_scan_batch, scan_root

from conftest import create_image as shared_create_image

def test_scan_populates_cache_and_preview(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "previews"
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    shared_create_image(photo_dir / "sample.jpg")

    initialize_database(db_path)

    with connect(db_path) as connection:
        summary = scan_root(
            connection,
            root=photo_dir,
            recursive=True,
            extensions=(".jpg",),
            preview_dir=preview_dir,
        )
        row = connection.execute(
            "SELECT path, width, height, preview_path, preview_status, scan_status FROM files"
        ).fetchone()

    assert summary.files_seen == 1
    assert summary.files_added == 1
    assert summary.files_failed == 0
    assert row["path"].endswith("sample.jpg")
    assert row["width"] == 120
    assert row["height"] == 80
    assert row["preview_status"] == "ready"
    assert row["scan_status"] == "new"
    assert Path(row["preview_path"]).exists()


def test_scan_root_passes_raw_preview_mode_to_preview_generation(monkeypatch, tmp_path: Path) -> None:
    import shotsieve.scanner as scanner_module

    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "previews"
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    raw_path = photo_dir / "sample.nef"
    raw_path.write_bytes(b"fake-raw")

    initialize_database(db_path)

    captured: dict[str, object] = {}

    def fake_generate_preview(
        path: Path,
        generated_preview_dir: Path,
        *,
        raw_preview_mode: str = "auto",
        max_decode_pixels: int = 64_000_000,
    ):
        captured["path"] = path
        captured["preview_dir"] = generated_preview_dir
        captured["raw_preview_mode"] = raw_preview_mode
        captured["max_decode_pixels"] = max_decode_pixels
        return PreviewResult(
            path=str((generated_preview_dir / "sample.jpg").resolve()),
            status="ready",
            width=120,
            height=80,
            capture_time=None,
            error_text=None,
        )

    monkeypatch.setattr(scanner_module, "generate_preview", fake_generate_preview)

    with connect(db_path) as connection:
        summary = scan_root(
            connection,
            root=photo_dir,
            recursive=True,
            extensions=(".nef",),
            preview_dir=preview_dir,
            raw_preview_mode="high-quality",
            max_decode_pixels=48_000_000,
        )

    assert summary.files_seen == 1
    assert captured == {
        "path": raw_path,
        "preview_dir": preview_dir,
        "raw_preview_mode": "high-quality",
        "max_decode_pixels": 48_000_000,
    }


def test_scan_marks_unchanged_on_repeat_scan(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "previews"
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    shared_create_image(photo_dir / "sample.jpg")

    initialize_database(db_path)

    with connect(db_path) as connection:
        scan_root(
            connection,
            root=photo_dir,
            recursive=True,
            extensions=(".jpg",),
            preview_dir=preview_dir,
        )
        summary = scan_root(
            connection,
            root=photo_dir,
            recursive=True,
            extensions=(".jpg",),
            preview_dir=preview_dir,
        )

    assert summary.files_seen == 1
    assert summary.files_added == 0
    assert summary.files_updated == 0
    assert summary.files_unchanged == 1


def test_scan_preserves_deleted_files_on_rescan_until_explicit_cleanup(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "previews"
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    sample_path = photo_dir / "sample.jpg"
    shared_create_image(sample_path)

    initialize_database(db_path)

    with connect(db_path) as connection:
        scan_root(
            connection,
            root=photo_dir,
            recursive=True,
            extensions=(".jpg",),
            preview_dir=preview_dir,
        )
        sample_path.unlink()
        summary = scan_root(
            connection,
            root=photo_dir,
            recursive=True,
            extensions=(".jpg",),
            preview_dir=preview_dir,
        )
        count = connection.execute("SELECT COUNT(*) AS count FROM files").fetchone()["count"]

    assert summary.files_seen == 0
    assert summary.files_removed == 0
    assert count == 1


def test_unavailable_scan_preserves_existing_catalog_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "previews"
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    shared_create_image(photo_dir / "sample.jpg")

    initialize_database(db_path)

    with connect(db_path) as connection:
        scan_root(connection, root=photo_dir, recursive=True, extensions=(".jpg",), preview_dir=preview_dir)

        with pytest.raises(OSError, match="Unable to enumerate"):
            scan_root(
                connection,
                root=tmp_path / "unavailable-library",
                recursive=True,
                extensions=(".jpg",),
                preview_dir=preview_dir,
            )

        rows = connection.execute("SELECT path FROM files ORDER BY path").fetchall()

    assert [row["path"] for row in rows] == [str((photo_dir / "sample.jpg").resolve())]


def test_failed_discovery_diagnostic_survives_database_rollback(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "previews"
    missing_root = tmp_path / "unavailable-library"
    initialize_database(db_path)

    with pytest.raises(FileDiscoveryError):
        with connect(db_path) as connection:
            scan_root(
                connection,
                root=missing_root,
                recursive=True,
                extensions=(".jpg",),
                preview_dir=preview_dir,
                generate_previews=False,
            )

    with database(db_path) as connection:
        run = connection.execute(
            """
            SELECT root_path, started_time, completed_time, files_seen,
                   files_added, files_updated, files_unchanged, files_removed,
                   status, error_text
            FROM scan_runs ORDER BY id DESC LIMIT 1
            """
        ).fetchone()

    assert run["root_path"] == str(missing_root.resolve())
    assert run["started_time"]
    assert run["completed_time"]
    assert run["files_seen"] == 0
    assert run["status"] == "failed"
    assert "Unable to enumerate" in run["error_text"]


def test_failed_scan_after_prior_batch_preserves_processed_diagnostic_counts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from shotsieve import scanner as scanner_module

    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "previews"
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    for index in range(101):
        shared_create_image(photo_dir / f"sample-{index}.jpg")
    initialize_database(db_path)

    original_process_batch = scanner_module._process_scan_batch
    batch_calls = 0

    def fail_second_batch(*args, **kwargs):
        nonlocal batch_calls
        batch_calls += 1
        if batch_calls == 2:
            raise RuntimeError("late scan failure")
        return original_process_batch(*args, **kwargs)

    monkeypatch.setattr(scanner_module, "_process_scan_batch", fail_second_batch)

    with pytest.raises(RuntimeError, match="late scan failure"):
        with database(db_path) as connection:
            scan_root(
                connection,
                root=photo_dir,
                recursive=True,
                extensions=(".jpg",),
                preview_dir=preview_dir,
                generate_previews=False,
            )

    with database(db_path) as connection:
        file_count = connection.execute("SELECT COUNT(*) AS count FROM files").fetchone()["count"]
        run = connection.execute(
            "SELECT files_seen, files_added, status, error_text FROM scan_runs ORDER BY id DESC LIMIT 1"
        ).fetchone()

    assert file_count == 0
    assert run["files_seen"] == 100
    assert run["files_added"] == 100
    assert run["status"] == "failed"
    assert run["error_text"] == "late scan failure"


def test_cancelled_scan_persists_incomplete_diagnostic_after_rollback(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "previews"
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    shared_create_image(photo_dir / "sample.jpg")
    initialize_database(db_path)

    def cancel_scan() -> None:
        raise InterruptedError("cancelled during discovery")

    with pytest.raises(ScanInterrupted, match="cancelled during discovery"):
        with database(db_path) as connection:
            scan_root(
                connection,
                root=photo_dir,
                recursive=True,
                extensions=(".jpg",),
                preview_dir=preview_dir,
                generate_previews=False,
                cancel_check=cancel_scan,
            )

    with database(db_path) as connection:
        run = connection.execute(
            "SELECT files_seen, status, error_text FROM scan_runs ORDER BY id DESC LIMIT 1"
        ).fetchone()

    assert run["files_seen"] == 0
    assert run["status"] == "failed"
    assert "cancelled during discovery" in run["error_text"]


def test_scan_excludes_absolute_folder_rules_but_preserves_prior_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "previews"
    photo_dir = tmp_path / "photos"
    excluded_dir = photo_dir / "Others" / "Beatrice Low Res"
    photo_dir.mkdir()
    excluded_dir.mkdir(parents=True)
    shared_create_image(photo_dir / "keep.jpg")
    shared_create_image(excluded_dir / "exclude-1.jpg")
    shared_create_image(excluded_dir / "exclude-2.jpg")

    initialize_database(db_path)

    with connect(db_path) as connection:
        scan_root(
            connection,
            root=photo_dir,
            recursive=True,
            extensions=(".jpg",),
            preview_dir=preview_dir,
            generate_previews=False,
        )
        summary = scan_root(
            connection,
            root=photo_dir,
            recursive=True,
            extensions=(".jpg",),
            preview_dir=preview_dir,
            generate_previews=False,
            ignore_rules=(str(excluded_dir.resolve()),),
        )
        paths = [row["path"] for row in connection.execute("SELECT path FROM files ORDER BY path").fetchall()]

    assert summary.files_seen == 1
    assert summary.files_removed == 0
    assert paths == [
        str((photo_dir / "Others" / "Beatrice Low Res" / "exclude-1.jpg").resolve()),
        str((photo_dir / "Others" / "Beatrice Low Res" / "exclude-2.jpg").resolve()),
        str((photo_dir / "keep.jpg").resolve()),
    ]


def test_scan_applies_file_ignore_rules_consistently(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "previews"
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    shared_create_image(photo_dir / "keep.jpg")
    shared_create_image(photo_dir / "ignored.jpg")

    initialize_database(db_path)

    with connect(db_path) as connection:
        summary = scan_root(
            connection,
            root=photo_dir,
            recursive=False,
            extensions=(".jpg",),
            preview_dir=preview_dir,
            generate_previews=False,
            ignore_rules=("ignored.jpg",),
        )
        paths = [row["path"] for row in connection.execute("SELECT path FROM files").fetchall()]

    assert summary.files_seen == 1
    assert paths == [str((photo_dir / "keep.jpg").resolve())]


def test_scan_rescan_preserves_sibling_prefix_root_entries(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "previews"
    root_main = tmp_path / "photos"
    root_sibling = tmp_path / "photos-archive"
    root_main.mkdir()
    root_sibling.mkdir()

    main_file = root_main / "main.jpg"
    sibling_file = root_sibling / "sibling.jpg"
    shared_create_image(main_file)
    shared_create_image(sibling_file)

    initialize_database(db_path)

    with connect(db_path) as connection:
        scan_root(connection, root=root_main, recursive=True, extensions=(".jpg",), preview_dir=preview_dir)
        scan_root(connection, root=root_sibling, recursive=True, extensions=(".jpg",), preview_dir=preview_dir)

        main_file.unlink()
        scan_root(connection, root=root_main, recursive=True, extensions=(".jpg",), preview_dir=preview_dir)

        paths = [row["path"] for row in connection.execute("SELECT path FROM files ORDER BY path").fetchall()]

    assert any(path.endswith("sibling.jpg") for path in paths)
    assert any(path.endswith("main.jpg") for path in paths)


def test_scan_respects_offset_without_limit(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "previews"
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()

    for index in range(5):
        shared_create_image(photo_dir / f"sample-{index}.jpg")

    initialize_database(db_path)

    with connect(db_path) as connection:
        summary = scan_root(
            connection,
            root=photo_dir,
            recursive=True,
            limit=None,
            offset=2,
            extensions=(".jpg",),
            preview_dir=preview_dir,
        )
        count = connection.execute("SELECT COUNT(*) AS count FROM files").fetchone()["count"]

    assert summary.files_seen == 3
    assert summary.files_added == 3
    assert count == 3


def test_scan_continues_after_preview_failure(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "previews"
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    shared_create_image(photo_dir / "good.jpg")
    (photo_dir / "broken.jpg").write_bytes(b"not-a-real-image")

    initialize_database(db_path)

    with connect(db_path) as connection:
        summary = scan_root(
            connection,
            root=photo_dir,
            recursive=True,
            extensions=(".jpg",),
            preview_dir=preview_dir,
        )
        rows = connection.execute(
            "SELECT path, preview_status, scan_status, last_error FROM files ORDER BY path"
        ).fetchall()
        run = connection.execute(
            "SELECT status, error_text FROM scan_runs ORDER BY id DESC LIMIT 1"
        ).fetchone()

    assert summary.files_seen == 2
    assert summary.files_failed == 1
    assert rows[0]["preview_status"] == "failed"
    assert rows[0]["scan_status"] == "error"
    assert rows[0]["last_error"]
    assert rows[1]["preview_status"] == "ready"
    assert run["status"] == "completed_with_errors"
    assert run["error_text"]


def test_scan_rescan_clears_stale_error_after_repair(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "previews"
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    sample_path = photo_dir / "sample.jpg"
    sample_path.write_bytes(b"not-a-real-image")

    initialize_database(db_path)

    with connect(db_path) as connection:
        first_summary = scan_root(
            connection,
            root=photo_dir,
            recursive=True,
            extensions=(".jpg",),
            preview_dir=preview_dir,
        )
        failed_row = connection.execute(
            "SELECT preview_status, last_error FROM files"
        ).fetchone()

        shared_create_image(sample_path)

        second_summary = scan_root(
            connection,
            root=photo_dir,
            recursive=True,
            extensions=(".jpg",),
            preview_dir=preview_dir,
        )
        repaired_row = connection.execute(
            "SELECT preview_path, preview_status, width, height, last_error, scan_status FROM files"
        ).fetchone()

    assert first_summary.files_failed == 1
    assert failed_row["preview_status"] == "failed"
    assert failed_row["last_error"]

    assert second_summary.files_failed == 0
    assert second_summary.files_updated == 1
    assert repaired_row["preview_status"] == "ready"
    assert repaired_row["last_error"] is None
    assert repaired_row["width"] == 120
    assert repaired_row["height"] == 80
    assert repaired_row["scan_status"] == "updated"
    assert Path(repaired_row["preview_path"]).exists()


def test_scan_rescan_replaces_stale_preview_metadata_when_file_breaks(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "previews"
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    sample_path = photo_dir / "sample.jpg"
    shared_create_image(sample_path)

    initialize_database(db_path)

    with connect(db_path) as connection:
        scan_root(
            connection,
            root=photo_dir,
            recursive=True,
            extensions=(".jpg",),
            preview_dir=preview_dir,
        )
        first_row = connection.execute(
            "SELECT preview_path, preview_status, width, height FROM files"
        ).fetchone()
        assert first_row["preview_status"] == "ready"
        assert Path(first_row["preview_path"]).exists()

        sample_path.write_bytes(b"not-a-real-image")

        summary = scan_root(
            connection,
            root=photo_dir,
            recursive=True,
            extensions=(".jpg",),
            preview_dir=preview_dir,
        )
        broken_row = connection.execute(
            "SELECT preview_path, preview_status, width, height, last_error, scan_status FROM files"
        ).fetchone()

    assert summary.files_failed == 1
    assert summary.files_updated == 1
    assert broken_row["preview_status"] == "failed"
    assert broken_row["preview_path"] is None
    assert broken_row["width"] is None
    assert broken_row["height"] is None
    assert broken_row["last_error"]
    assert broken_row["scan_status"] == "error"


def test_scan_rescan_clears_stale_error_when_unchanged_file_regenerates_preview(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "previews"
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    sample_path = photo_dir / "sample.jpg"
    shared_create_image(sample_path)

    initialize_database(db_path)

    with connect(db_path) as connection:
        scan_root(
            connection,
            root=photo_dir,
            recursive=True,
            extensions=(".jpg",),
            preview_dir=preview_dir,
        )
        connection.execute(
            "UPDATE files SET preview_status = 'failed', last_error = 'stale failure'"
        )

        summary = scan_root(
            connection,
            root=photo_dir,
            recursive=True,
            extensions=(".jpg",),
            preview_dir=preview_dir,
        )
        row = connection.execute(
            "SELECT preview_status, last_error, width, height, scan_status FROM files"
        ).fetchone()

    assert summary.files_updated == 0
    assert summary.files_unchanged == 1
    assert row["preview_status"] == "ready"
    assert row["last_error"] is None
    assert row["width"] == 120
    assert row["height"] == 80
    assert row["scan_status"] == "unchanged"




def test_scan_ignores_generated_preview_directory(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "photos" / "previews"
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    shared_create_image(photo_dir / "source.jpg")

    initialize_database(db_path)

    with connect(db_path) as connection:
        first_summary = scan_root(
            connection,
            root=photo_dir,
            recursive=True,
            extensions=(".jpg",),
            preview_dir=preview_dir,
        )
        second_summary = scan_root(
            connection,
            root=photo_dir,
            recursive=True,
            extensions=(".jpg",),
            preview_dir=preview_dir,
        )
        count = connection.execute("SELECT COUNT(*) AS count FROM files").fetchone()["count"]

    assert first_summary.files_seen == 1
    assert second_summary.files_seen == 1
    assert count == 1


def test_scan_rescan_skips_preview_for_unchanged_files(tmp_path: Path) -> None:
    """Verify that a second scan with unchanged files does NOT regenerate previews."""
    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "previews"
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    shared_create_image(photo_dir / "sample.jpg")

    initialize_database(db_path)

    with connect(db_path) as connection:
        scan_root(
            connection,
            root=photo_dir,
            recursive=True,
            extensions=(".jpg",),
            preview_dir=preview_dir,
        )

        # Record preview file modification time after first scan.
        row = connection.execute(
            "SELECT preview_path FROM files"
        ).fetchone()
        preview_path = Path(row["preview_path"])
        assert preview_path.exists()
        first_mtime = preview_path.stat().st_mtime

        # Small delay so any rewrite would produce a different mtime.
        import time
        time.sleep(0.05)

        summary = scan_root(
            connection,
            root=photo_dir,
            recursive=True,
            extensions=(".jpg",),
            preview_dir=preview_dir,
        )

    # Preview file should NOT have been rewritten.
    assert preview_path.stat().st_mtime == first_mtime
    assert summary.files_unchanged == 1
    assert summary.files_updated == 0


def test_scan_rescan_regenerates_missing_preview_for_unchanged_source(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "previews"
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    shared_create_image(photo_dir / "sample.jpg")

    initialize_database(db_path)

    with connect(db_path) as connection:
        scan_root(
            connection,
            root=photo_dir,
            recursive=True,
            extensions=(".jpg",),
            preview_dir=preview_dir,
        )
        row = connection.execute("SELECT preview_path FROM files").fetchone()
        preview_path = Path(row["preview_path"])
        assert preview_path.exists()

        preview_path.unlink()
        assert not preview_path.exists()

        summary = scan_root(
            connection,
            root=photo_dir,
            recursive=True,
            extensions=(".jpg",),
            preview_dir=preview_dir,
        )
        refreshed = connection.execute("SELECT preview_path, preview_status FROM files").fetchone()

    assert Path(refreshed["preview_path"]).exists()
    assert refreshed["preview_status"] == "ready"
    assert summary.files_seen == 1


def test_scan_marks_row_status_updated_when_source_changes(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "previews"
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    sample_path = photo_dir / "sample.jpg"
    shared_create_image(sample_path)

    initialize_database(db_path)

    with connect(db_path) as connection:
        scan_root(
            connection,
            root=photo_dir,
            recursive=True,
            extensions=(".jpg",),
            preview_dir=preview_dir,
        )

        image = Image.new("RGB", (300, 200), color=(10, 20, 30))
        image.save(sample_path, format="JPEG")

        summary = scan_root(
            connection,
            root=photo_dir,
            recursive=True,
            extensions=(".jpg",),
            preview_dir=preview_dir,
        )
        row = connection.execute("SELECT scan_status FROM files").fetchone()

    assert summary.files_updated == 1
    assert row["scan_status"] == "updated"


def test_scan_reuses_single_process_pool_across_multiple_batches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import shotsieve.scanner as scanner_module

    pool_creations = {"count": 0}

    class CountingPool:
        def __init__(self, max_workers=None):
            from concurrent.futures import ThreadPoolExecutor

            pool_creations["count"] += 1
            self._delegate = ThreadPoolExecutor(max_workers=max_workers)

        def submit(self, fn, *args, **kwargs):
            return self._delegate.submit(fn, *args, **kwargs)

        def shutdown(self, wait=True):
            self._delegate.shutdown(wait=wait)

    monkeypatch.setattr(scanner_module.concurrent.futures, "ProcessPoolExecutor", CountingPool)

    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "previews"
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()

    for index in range(250):
        image = Image.new("RGB", (16, 16), color=(index % 255, 10, 20))
        image.save(photo_dir / f"sample-{index}.jpg", format="JPEG")

    initialize_database(db_path)

    with connect(db_path) as connection:
        scan_root(
            connection,
            root=photo_dir,
            recursive=True,
            extensions=(".jpg",),
            preview_dir=preview_dir,
            generate_previews=True,
        )

    assert pool_creations["count"] == 1


@pytest.mark.parametrize("execution", ["inline", "parallel"])
def test_scan_batch_execution_strategies_share_accounting(
    execution: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inline and pooled batches persist the same success/failure totals."""
    from concurrent.futures import ThreadPoolExecutor
    import shotsieve.scanner as scanner_module

    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "previews"
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    paths = [photo_dir / name for name in ("good-a.jpg", "broken.jpg", "good-b.jpg")]
    for path in paths:
        path.write_bytes(b"source")
    initialize_database(db_path)

    def fake_gather_file_metadata(path: Path, **_kwargs) -> dict:
        if path.name == "broken.jpg":
            raise OSError("metadata failure")
        return {
            "path": str(path),
            "path_key": scanner_module.canonical_path_key(path),
            "size_bytes": 6,
            "modified_time": 1.0,
            "format": "jpg",
            "last_scan_time": "scan-time",
            "width": 120,
            "height": 80,
            "capture_time": None,
            "preview_path": None,
            "preview_status": "ready",
            "preview_conversion_version": "test",
            "last_error": None,
            "scan_status": "new",
            "analysis_status": None,
            "analysis_error": None,
            "last_analysis_time": None,
            "preserve_metadata": False,
        }

    monkeypatch.setattr(scanner_module, "gather_file_metadata", fake_gather_file_metadata)
    monkeypatch.setattr(scanner_module, "_POOL_THRESHOLD", 100 if execution == "inline" else 0)

    with connect(db_path) as connection:
        summary = ScanSummary()
        executor = ThreadPoolExecutor(max_workers=2) if execution == "parallel" else None
        try:
            _process_scan_batch(
                paths,
                connection,
                summary,
                2,
                preview_dir=preview_dir,
                rescan_all=False,
                generate_previews=True,
                executor=executor,
            )
        finally:
            if executor is not None:
                executor.shutdown(wait=True)
        stored_count = connection.execute("SELECT COUNT(*) AS count FROM files").fetchone()["count"]

    assert (summary.files_added, summary.files_failed, summary.last_batch_error, stored_count) == (
        2,
        1,
        "metadata failure",
        2,
    )


def test_parallel_scan_persists_results_in_input_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Worker completion order must not change catalog insertion order."""
    from concurrent.futures import Future
    import shotsieve.scanner as scanner_module

    db_path = tmp_path / "data" / "shotsieve.db"
    preview_dir = tmp_path / "previews"
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    paths = [photo_dir / f"sample-{index}.jpg" for index in range(4)]
    initialize_database(db_path)

    def fake_gather_file_metadata(path: Path, **_kwargs) -> dict:
        return {
            "path": str(path),
            "path_key": scanner_module.canonical_path_key(path),
            "size_bytes": 6,
            "modified_time": 1.0,
            "format": "jpg",
            "last_scan_time": "scan-time",
            "width": 120,
            "height": 80,
            "capture_time": None,
            "preview_path": None,
            "preview_status": "ready",
            "preview_conversion_version": "test",
            "last_error": None,
            "scan_status": "new",
            "analysis_status": None,
            "analysis_error": None,
            "last_analysis_time": None,
            "preserve_metadata": False,
        }

    class ImmediatePool:
        def submit(self, function, *args, **kwargs):
            future = Future()
            try:
                future.set_result(function(*args, **kwargs))
            except BaseException as exc:
                future.set_exception(exc)
            return future

    def reverse_wait(futures, *, return_when):
        _ = return_when
        return list(reversed(list(futures))), []

    monkeypatch.setattr(scanner_module, "gather_file_metadata", fake_gather_file_metadata)
    monkeypatch.setattr(scanner_module, "_POOL_THRESHOLD", 0)
    monkeypatch.setattr(scanner_module.concurrent.futures, "wait", reverse_wait)

    with connect(db_path) as connection:
        summary = ScanSummary()
        _process_scan_batch(
            paths,
            connection,
            summary,
            2,
            preview_dir=preview_dir,
            rescan_all=False,
            generate_previews=True,
            executor=ImmediatePool(),
        )
        stored_paths = [
            row["path"] for row in connection.execute("SELECT path FROM files ORDER BY id").fetchall()
        ]

    assert summary.files_added == len(paths)
    assert stored_paths == [str(path) for path in paths]


