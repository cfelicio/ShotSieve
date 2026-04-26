import io
import os
from pathlib import Path
import sys
from types import SimpleNamespace

from shotsieve import preview as preview_module


class _FakeFuture:
    def __init__(self, result=None, error: Exception | None = None) -> None:
        self._result = result
        self._error = error

    def result(self):
        if self._error is not None:
            raise self._error
        return self._result


class _FakeExecutor:
    def __init__(self, *, max_workers: int | None, future_map: dict[Path, _FakeFuture]) -> None:
        self.max_workers = max_workers
        self._future_map = future_map

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def submit(self, fn, source_path: Path, preview_dir: Path, **kwargs):
        assert fn is preview_module.generate_preview
        assert preview_dir.name == "previews"
        assert kwargs.get("raw_preview_mode", "auto") in {"fast", "auto", "high-quality"}
        return self._future_map[source_path]


def test_generate_previews_parallel_records_failed_worker_results(monkeypatch, tmp_path: Path) -> None:
    preview_dir = tmp_path / "previews"
    source_paths = [
        tmp_path / "first.jpg",
        tmp_path / "broken.jpg",
        tmp_path / "third.jpg",
    ]

    future_map = {
        source_paths[0]: _FakeFuture(
            result=preview_module.PreviewResult(
                path=str(preview_dir / "first-preview.jpg"),
                status="ready",
                width=120,
                height=80,
                capture_time=None,
                error_text=None,
            )
        ),
        source_paths[1]: _FakeFuture(error=RuntimeError("worker exploded")),
        source_paths[2]: _FakeFuture(
            result=preview_module.PreviewResult(
                path=str(preview_dir / "third-preview.jpg"),
                status="ready",
                width=90,
                height=60,
                capture_time=None,
                error_text=None,
            )
        ),
    }

    def fake_process_pool_executor(*, max_workers: int | None = None):
        return _FakeExecutor(max_workers=max_workers, future_map=future_map)

    def fake_as_completed(futures):
        assert set(futures) == set(future_map.values())
        yield future_map[source_paths[2]]
        yield future_map[source_paths[1]]
        yield future_map[source_paths[0]]

    progress_updates: list[tuple[int, int]] = []
    def capture_progress(completed: int, total: int) -> None:
        progress_updates.append((completed, total))

    monkeypatch.setattr("concurrent.futures.ProcessPoolExecutor", fake_process_pool_executor)
    monkeypatch.setattr("concurrent.futures.as_completed", fake_as_completed)

    results = preview_module.generate_previews_parallel(
        source_paths,
        preview_dir,
        max_workers=3,
        progress_callback=capture_progress,
    )

    assert [result.status for result in results] == ["ready", "failed", "ready"]
    assert results[0].path == str(preview_dir / "first-preview.jpg")
    assert results[1].path is None
    assert results[1].width is None
    assert results[1].height is None
    assert results[1].capture_time is None
    assert results[1].error_text == "worker exploded"
    assert results[2].path == str(preview_dir / "third-preview.jpg")
    assert progress_updates == [(1, 3), (2, 3), (3, 3)]


def test_generate_preview_captures_nonfatal_decoder_stderr_as_issue_text(monkeypatch, tmp_path: Path) -> None:
    source_path = tmp_path / "sample.jpg"
    preview_dir = tmp_path / "previews"
    source_path.write_bytes(b"fake-image")

    class FakeImage:
        size = (120, 80)
        mode = "RGB"

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def thumbnail(self, *_args, **_kwargs):
            return None

        def save(self, path, **_kwargs):
            Path(path).write_bytes(b"preview")

        def getexif(self):
            return {}

    def fake_open(path):
        assert path == source_path
        print("Corrupt JPEG data: 463 extraneous bytes before marker 0xee", file=sys.stderr)
        return FakeImage()

    monkeypatch.setattr(preview_module.threading, "active_count", lambda: 1)
    monkeypatch.setattr(preview_module.Image, "open", fake_open)
    monkeypatch.setattr(preview_module.ImageOps, "exif_transpose", lambda image: image)

    result = preview_module.generate_preview(source_path, preview_dir)

    assert result.status == "ready"
    assert result.path is not None
    assert result.error_text == "sample.jpg: Corrupt JPEG data: 463 extraneous bytes before marker 0xee"


def test_generate_preview_failure_keeps_exception_and_decoder_issue_text(monkeypatch, tmp_path: Path) -> None:
    source_path = tmp_path / "broken.jpg"
    preview_dir = tmp_path / "previews"
    source_path.write_bytes(b"fake-image")

    class BrokenImage:
        def __enter__(self):
            print("Corrupt JPEG data: premature end of data segment", file=sys.stderr)
            raise OSError("decoder exploded")

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(preview_module.threading, "active_count", lambda: 1)
    monkeypatch.setattr(preview_module.Image, "open", lambda path: BrokenImage())

    result = preview_module.generate_preview(source_path, preview_dir)

    assert result.status == "failed"
    assert result.path is None
    assert result.error_text == (
        "decoder exploded | broken.jpg: Corrupt JPEG data: premature end of data segment"
    )


def test_generate_preview_captures_low_level_decoder_stderr_with_source_file_context(
    monkeypatch,
    tmp_path: Path,
    capfd,
) -> None:
    source_path = tmp_path / "sample.jpg"
    preview_dir = tmp_path / "previews"
    source_path.write_bytes(b"fake-image")

    class FakeImage:
        size = (120, 80)
        mode = "RGB"

        def __enter__(self):
            os.write(2, b"Corrupt JPEG data: 463 extraneous bytes before marker 0xee\n")
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def thumbnail(self, *_args, **_kwargs):
            return None

        def save(self, path, **_kwargs):
            Path(path).write_bytes(b"preview")

        def getexif(self):
            return {}

    monkeypatch.setattr(preview_module.threading, "active_count", lambda: 1)
    monkeypatch.setattr(preview_module.Image, "open", lambda path: FakeImage())
    monkeypatch.setattr(preview_module.ImageOps, "exif_transpose", lambda image: image)

    result = preview_module.generate_preview(source_path, preview_dir)
    captured = capfd.readouterr()

    assert result.status == "ready"
    assert result.path is not None
    assert result.error_text == "sample.jpg: Corrupt JPEG data: 463 extraneous bytes before marker 0xee"
    assert captured.out == ""
    assert captured.err.strip() == "sample.jpg: Corrupt JPEG data: 463 extraneous bytes before marker 0xee"


def test_generate_preview_skips_low_level_stderr_capture_when_multiple_threads_are_active(
    monkeypatch,
    tmp_path: Path,
    capfd,
) -> None:
    source_path = tmp_path / "sample.jpg"
    preview_dir = tmp_path / "previews"
    source_path.write_bytes(b"fake-image")

    class FakeImage:
        size = (120, 80)
        mode = "RGB"

        def __enter__(self):
            os.write(2, b"Corrupt JPEG data: thread-unsafe warning\n")
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def thumbnail(self, *_args, **_kwargs):
            return None

        def save(self, path, **_kwargs):
            Path(path).write_bytes(b"preview")

        def getexif(self):
            return {}

    monkeypatch.setattr(preview_module.threading, "active_count", lambda: 2)
    monkeypatch.setattr(preview_module.Image, "open", lambda path: FakeImage())
    monkeypatch.setattr(preview_module.ImageOps, "exif_transpose", lambda image: image)

    result = preview_module.generate_preview(source_path, preview_dir)
    captured = capfd.readouterr()

    assert result.status == "ready"
    assert result.path is not None
    assert result.error_text is None
    assert "Corrupt JPEG data: thread-unsafe warning" in captured.err


def test_generate_preview_falls_back_when_stderr_fd_duplication_is_unavailable(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "sample.jpg"
    preview_dir = tmp_path / "previews"
    source_path.write_bytes(b"fake-image")

    class FakeImage:
        size = (120, 80)
        mode = "RGB"

        def __enter__(self):
            print("decoder warning routed through sys.stderr", file=sys.stderr)
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def thumbnail(self, *_args, **_kwargs):
            return None

        def save(self, path, **_kwargs):
            Path(path).write_bytes(b"preview")

        def getexif(self):
            return {}

    original_dup = os.dup

    def fake_dup(fd: int):
        if fd == 2:
            raise OSError("stderr fd unavailable")
        return original_dup(fd)

    monkeypatch.setattr(preview_module.threading, "active_count", lambda: 1)
    monkeypatch.setattr(preview_module.os, "dup", fake_dup)
    monkeypatch.setattr(preview_module.Image, "open", lambda path: FakeImage())
    monkeypatch.setattr(preview_module.ImageOps, "exif_transpose", lambda image: image)

    result = preview_module.generate_preview(source_path, preview_dir)

    assert result.status == "ready"
    assert result.path is not None
    assert result.error_text == "sample.jpg: decoder warning routed through sys.stderr"


def test_generate_raw_preview_falls_back_to_demosaic_when_embedded_thumbnail_is_too_small(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "sample.nef"
    preview_dir = tmp_path / "previews"
    source_path.write_bytes(b"fake-raw")

    thumb_buffer = io.BytesIO()
    preview_module.Image.new("RGB", (640, 480), color="red").save(thumb_buffer, format="JPEG")
    thumb_bytes = thumb_buffer.getvalue()

    rendered_image = preview_module.Image.new("RGB", (1600, 1200), color="blue")
    postprocess_calls: list[dict[str, object]] = []

    class FakeRawImage:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def extract_thumb(self):
            return SimpleNamespace(format="jpeg", data=thumb_bytes)

        def postprocess(self, **kwargs):
            postprocess_calls.append(dict(kwargs))
            return "rendered-array"

    monkeypatch.setattr(
        preview_module,
        "rawpy",
        SimpleNamespace(
            imread=lambda _path: FakeRawImage(),
            ThumbFormat=SimpleNamespace(JPEG="jpeg", BITMAP="bitmap"),
        ),
    )
    monkeypatch.setattr(preview_module.Image, "fromarray", lambda _array: rendered_image.copy())

    result = preview_module.generate_raw_preview(source_path, preview_dir)

    assert result.status == "ready"
    assert result.path is not None
    assert postprocess_calls == [{"use_camera_wb": True, "no_auto_bright": False}]
    generated_preview = preview_module.Image.open(result.path)
    assert max(generated_preview.size) == preview_module.MAX_PREVIEW_SIZE[0]


def test_generate_raw_preview_uses_embedded_thumbnail_when_it_is_large_enough(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "sample.cr3"
    preview_dir = tmp_path / "previews"
    source_path.write_bytes(b"fake-raw")

    thumb_buffer = io.BytesIO()
    preview_module.Image.new("RGB", (1024, 768), color="green").save(thumb_buffer, format="JPEG")
    thumb_bytes = thumb_buffer.getvalue()

    class FakeRawImage:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def extract_thumb(self):
            return SimpleNamespace(format="jpeg", data=thumb_bytes)

        def postprocess(self, **_kwargs):
            raise AssertionError("postprocess should not be used when the embedded thumbnail is sufficient")

    monkeypatch.setattr(
        preview_module,
        "rawpy",
        SimpleNamespace(
            imread=lambda _path: FakeRawImage(),
            ThumbFormat=SimpleNamespace(JPEG="jpeg", BITMAP="bitmap"),
        ),
    )

    result = preview_module.generate_raw_preview(source_path, preview_dir)

    assert result.status == "ready"
    assert result.path is not None
    assert Path(result.path).read_bytes() == thumb_bytes


def test_generate_raw_preview_fast_mode_uses_small_embedded_thumbnail(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "sample.raf"
    preview_dir = tmp_path / "previews"
    source_path.write_bytes(b"fake-raw")

    thumb_buffer = io.BytesIO()
    preview_module.Image.new("RGB", (640, 480), color="purple").save(thumb_buffer, format="JPEG")
    thumb_bytes = thumb_buffer.getvalue()

    class FakeRawImage:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def extract_thumb(self):
            return SimpleNamespace(format="jpeg", data=thumb_bytes)

        def postprocess(self, **_kwargs):
            raise AssertionError("fast mode should not demosaic when an embedded thumbnail exists")

    monkeypatch.setattr(
        preview_module,
        "rawpy",
        SimpleNamespace(
            imread=lambda _path: FakeRawImage(),
            ThumbFormat=SimpleNamespace(JPEG="jpeg", BITMAP="bitmap"),
        ),
    )

    result = preview_module.generate_raw_preview(source_path, preview_dir, raw_preview_mode="fast")

    assert result.status == "ready"
    assert result.path is not None
    assert Path(result.path).read_bytes() == thumb_bytes


def test_generate_raw_preview_high_quality_mode_skips_embedded_thumbnail(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "sample.dng"
    preview_dir = tmp_path / "previews"
    source_path.write_bytes(b"fake-raw")

    thumb_buffer = io.BytesIO()
    preview_module.Image.new("RGB", (2048, 1536), color="orange").save(thumb_buffer, format="JPEG")
    thumb_bytes = thumb_buffer.getvalue()

    rendered_image = preview_module.Image.new("RGB", (1500, 1000), color="yellow")
    calls = {"extract_thumb": 0, "postprocess": 0}

    class FakeRawImage:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def extract_thumb(self):
            calls["extract_thumb"] += 1
            return SimpleNamespace(format="jpeg", data=thumb_bytes)

        def postprocess(self, **_kwargs):
            calls["postprocess"] += 1
            return "rendered-array"

    monkeypatch.setattr(
        preview_module,
        "rawpy",
        SimpleNamespace(
            imread=lambda _path: FakeRawImage(),
            ThumbFormat=SimpleNamespace(JPEG="jpeg", BITMAP="bitmap"),
        ),
    )
    monkeypatch.setattr(preview_module.Image, "fromarray", lambda _array: rendered_image.copy())

    result = preview_module.generate_raw_preview(source_path, preview_dir, raw_preview_mode="high-quality")

    assert result.status == "ready"
    assert result.path is not None
    assert calls == {"extract_thumb": 0, "postprocess": 1}
