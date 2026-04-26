from __future__ import annotations

import io
import os
import sys
import tempfile
import threading
from contextlib import contextmanager, redirect_stderr
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from types import TracebackType
from typing import Callable, Sequence

from PIL import Image, ImageOps, UnidentifiedImageError

from shotsieve.config import (
    ALL_PREVIEWABLE_EXTENSIONS,
    DEFAULT_RAW_PREVIEW_MODE,
    RAW_CAMERA_EXTENSIONS,
)
from shotsieve.db import normalize_resolved_path

try:
    from pillow_heif import register_heif_opener
except ImportError:
    register_heif_opener = None

try:
    import rawpy
except ImportError:
    rawpy = None

if register_heif_opener is not None:
    register_heif_opener()

MAX_PREVIEW_SIZE = (1024, 1024)
MIN_RAW_THUMBNAIL_LONG_EDGE = max(MAX_PREVIEW_SIZE)
CAPTURE_TIME_EXIF_TAGS = (36867, 36868, 306)
_STDERR_CAPTURE_LOCK = threading.Lock()


@dataclass(slots=True)
class PreviewResult:
    path: str | None
    status: str
    width: int | None
    height: int | None
    capture_time: str | None
    error_text: str | None = None


def _format_nonfatal_issue(source_path: Path, stderr_text: str) -> str | None:
    issue_lines = [line.strip() for line in stderr_text.splitlines() if line.strip()]
    if not issue_lines:
        return None
    return f"{source_path.name}: {' | '.join(issue_lines)}"


def _combine_failure_error_text(exc: Exception, issue_text: str | None) -> str:
    failure_text = str(exc).strip() or exc.__class__.__name__
    if issue_text:
        return f"{failure_text} | {issue_text}"
    return failure_text


def _emit_nonfatal_issue(issue_text: str | None) -> None:
    if issue_text:
        print(issue_text, file=sys.stderr)


@contextmanager
def _captured_stderr(stderr_buffer: io.StringIO):
    if threading.active_count() > 1:
        with redirect_stderr(stderr_buffer):
            yield
        return

    with _STDERR_CAPTURE_LOCK:
        target_fds: set[int] = {2}
        try:
            target_fds.add(sys.stderr.fileno())
        except (AttributeError, OSError, ValueError):
            pass

        encoding = getattr(sys.stderr, "encoding", None) or "utf-8"
        errors = getattr(sys.stderr, "errors", None) or "replace"
        try:
            saved_fds = {fd: os.dup(fd) for fd in target_fds}
        except OSError:
            with redirect_stderr(stderr_buffer):
                yield
            return

        try:
            with tempfile.TemporaryFile(mode="w+", encoding=encoding, errors=errors) as captured_stream:
                try:
                    sys.stderr.flush()
                except Exception:
                    pass

                captured_exc_info: tuple[type[BaseException] | None, BaseException | None, TracebackType | None] | None = None
                with redirect_stderr(captured_stream):
                    try:
                        for fd in target_fds:
                            os.dup2(captured_stream.fileno(), fd)
                    except OSError:
                        for fd, saved_fd in saved_fds.items():
                            try:
                                os.dup2(saved_fd, fd)
                            except OSError:
                                continue
                        with redirect_stderr(stderr_buffer):
                            yield
                        return
                    try:
                        yield
                    except BaseException:
                        captured_exc_info = sys.exc_info()
                    finally:
                        try:
                            captured_stream.flush()
                        except Exception:
                            pass
                        for fd, saved_fd in saved_fds.items():
                            os.dup2(saved_fd, fd)

                captured_stream.seek(0)
                stderr_buffer.write(captured_stream.read())
                if captured_exc_info is not None:
                    captured_exc = captured_exc_info[1]
                    if captured_exc is None:
                        raise RuntimeError("stderr capture failed without an active exception")
                    raise captured_exc.with_traceback(captured_exc_info[2])
        finally:
            for saved_fd in saved_fds.values():
                os.close(saved_fd)


def generate_preview(
    source_path: Path,
    preview_dir: Path,
    *,
    raw_preview_mode: str = DEFAULT_RAW_PREVIEW_MODE,
) -> PreviewResult:
    suffix = source_path.suffix.casefold()
    if suffix not in ALL_PREVIEWABLE_EXTENSIONS:
        return PreviewResult(
            path=None,
            status="unsupported",
            width=None,
            height=None,
            capture_time=None,
        )

    if suffix in RAW_CAMERA_EXTENSIONS:
        return generate_raw_preview(source_path, preview_dir, raw_preview_mode=raw_preview_mode)

    preview_dir.mkdir(parents=True, exist_ok=True)
    preview_path, stale_preview_paths = preview_output_paths(source_path, preview_dir)

    stderr_buffer = io.StringIO()
    try:
        with _captured_stderr(stderr_buffer):
            with Image.open(source_path) as image:
                image = ImageOps.exif_transpose(image)
                capture_time = extract_capture_time(image)
                width, height = image.size

                if image.mode != "RGB":
                    image = image.convert("RGB")

                image.thumbnail(MAX_PREVIEW_SIZE, Image.Resampling.LANCZOS)
                image.save(preview_path, format="JPEG", quality=85, optimize=False)
                cleanup_stale_preview_paths(stale_preview_paths)
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        issue_text = _format_nonfatal_issue(source_path, stderr_buffer.getvalue())
        return PreviewResult(
            path=None,
            status="failed",
            width=None,
            height=None,
            capture_time=None,
            error_text=_combine_failure_error_text(exc, issue_text),
        )

    issue_text = _format_nonfatal_issue(source_path, stderr_buffer.getvalue())
    _emit_nonfatal_issue(issue_text)

    return PreviewResult(
        path=str(preview_path.resolve()),
        status="ready",
        width=width,
        height=height,
        capture_time=capture_time,
        error_text=issue_text,
    )


def generate_raw_preview(
    source_path: Path,
    preview_dir: Path,
    *,
    raw_preview_mode: str = DEFAULT_RAW_PREVIEW_MODE,
) -> PreviewResult:
    if rawpy is None:
        return PreviewResult(
            path=None,
            status="unsupported",
            width=None,
            height=None,
            capture_time=None,
            error_text="RAW preview support requires the optional rawpy dependency",
        )

    preview_dir.mkdir(parents=True, exist_ok=True)
    preview_path, stale_preview_paths = preview_output_paths(source_path, preview_dir)

    stderr_buffer = io.StringIO()
    try:
        with _captured_stderr(stderr_buffer):
            with rawpy.imread(str(source_path)) as raw_image:
                # Fast path: extract the embedded JPEG thumbnail (most RAW files have one).
                # Use it only when it is large enough for our target preview size;
                # tiny embedded thumbnails look visibly softer than the source.
                result = _try_extract_raw_thumbnail(
                    raw_image,
                    preview_path,
                    raw_preview_mode=raw_preview_mode,
                )
                if result is not None:
                    cleanup_stale_preview_paths(stale_preview_paths)
                    result.error_text = _format_nonfatal_issue(source_path, stderr_buffer.getvalue())
                    return result

                # Slow fallback: full Bayer demosaicing for RAW files without thumbnails.
                rgb = raw_image.postprocess(use_camera_wb=True, no_auto_bright=False)
    except (OSError, ValueError, RuntimeError) as exc:
        issue_text = _format_nonfatal_issue(source_path, stderr_buffer.getvalue())
        return PreviewResult(
            path=None,
            status="failed",
            width=None,
            height=None,
            capture_time=None,
            error_text=_combine_failure_error_text(exc, issue_text),
        )

    image = Image.fromarray(rgb)
    width, height = image.size
    image.thumbnail(MAX_PREVIEW_SIZE, Image.Resampling.LANCZOS)
    image.save(preview_path, format="JPEG", quality=85, optimize=False)
    cleanup_stale_preview_paths(stale_preview_paths)

    issue_text = _format_nonfatal_issue(source_path, stderr_buffer.getvalue())
    _emit_nonfatal_issue(issue_text)

    return PreviewResult(
        path=str(preview_path.resolve()),
        status="ready",
        width=width,
        height=height,
        capture_time=None,
        error_text=issue_text,
    )


def _try_extract_raw_thumbnail(
    raw_image,
    preview_path: Path,
    *,
    raw_preview_mode: str = DEFAULT_RAW_PREVIEW_MODE,
) -> PreviewResult | None:
    """Try to extract the embedded JPEG thumbnail from a RAW file.

    Returns a PreviewResult on success, or None if no usable thumbnail exists.
    """
    if raw_preview_mode == "high-quality":
        return None

    try:
        thumb = raw_image.extract_thumb()
    except Exception:
        return None

    capture_time = None
    thumb_format = getattr(rawpy, "ThumbFormat", None)
    jpeg_format = getattr(thumb_format, "JPEG", object())
    bitmap_format = getattr(thumb_format, "BITMAP", object())

    if thumb.format == jpeg_format:
        # Inspect the embedded JPEG before trusting it as our preview source.
        try:
            with Image.open(io.BytesIO(thumb.data)) as image:
                width, height = image.size
                if not _raw_thumbnail_is_acceptable(width, height, raw_preview_mode=raw_preview_mode):
                    return None
                capture_time = extract_capture_time(image)
                # Resize if the embedded thumbnail exceeds our preview size.
                if width > MAX_PREVIEW_SIZE[0] or height > MAX_PREVIEW_SIZE[1]:
                    image = ImageOps.exif_transpose(image)
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                    image.thumbnail(MAX_PREVIEW_SIZE, Image.Resampling.LANCZOS)
                    image.save(preview_path, format="JPEG", quality=85, optimize=False)
                    width, height = image.size
                else:
                    preview_path.write_bytes(thumb.data)
        except (OSError, UnidentifiedImageError):
            # Thumbnail was written but unreadable — treat as failed extraction.
            return None

        return PreviewResult(
            path=str(preview_path.resolve()),
            status="ready",
            width=width,
            height=height,
            capture_time=capture_time,
        )

    if thumb.format == bitmap_format:
        # Bitmap thumbnail — decode via PIL and save as JPEG.
        image = Image.fromarray(thumb.data)
        width, height = image.size
        if not _raw_thumbnail_is_acceptable(width, height, raw_preview_mode=raw_preview_mode):
            return None
        image.thumbnail(MAX_PREVIEW_SIZE, Image.Resampling.LANCZOS)
        image.save(preview_path, format="JPEG", quality=85, optimize=False)

        return PreviewResult(
            path=str(preview_path.resolve()),
            status="ready",
            width=width,
            height=height,
            capture_time=capture_time,
        )

    return None


def _raw_thumbnail_is_acceptable(
    width: int,
    height: int,
    *,
    raw_preview_mode: str = DEFAULT_RAW_PREVIEW_MODE,
) -> bool:
    if raw_preview_mode == "fast":
        return True
    return max(width, height) >= MIN_RAW_THUMBNAIL_LONG_EDGE


def stable_preview_name(source_path: Path) -> str:
    return sha1(normalize_resolved_path(source_path).encode("utf-8")).hexdigest()


def preview_name_candidates(source_path: Path) -> tuple[str, ...]:
    resolved_path = str(source_path.resolve())
    hash_inputs = (
        normalize_resolved_path(source_path),
        resolved_path,
        resolved_path.casefold(),
    )
    candidates: list[str] = []
    seen: set[str] = set()
    for hash_input in hash_inputs:
        candidate = sha1(hash_input.encode("utf-8")).hexdigest()
        if candidate in seen:
            continue
        seen.add(candidate)
        candidates.append(candidate)
    return tuple(candidates)


def preview_output_paths(source_path: Path, preview_dir: Path) -> tuple[Path, tuple[Path, ...]]:
    preview_path = preview_dir / f"{stable_preview_name(source_path)}.jpg"
    stale_candidate_names = stale_preview_cleanup_candidates(source_path)
    stale_paths = tuple(
        preview_dir / f"{candidate_name}.jpg"
        for candidate_name in stale_candidate_names
        if candidate_name != preview_path.stem
    )
    return preview_path, stale_paths


def stale_preview_cleanup_candidates(source_path: Path) -> tuple[str, ...]:
    legacy_name = sha1(str(source_path.resolve()).encode("utf-8")).hexdigest()
    stable_name = stable_preview_name(source_path)
    if legacy_name == stable_name:
        return ()
    return (legacy_name,)


def cleanup_stale_preview_paths(paths: Sequence[Path]) -> None:
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            continue


def delete_managed_preview_file(
    preview_path: str | Path | None,
    *,
    source_path: str | Path | None = None,
    preview_cache_root: Path | None,
    allow_path_parent_fallback: bool = False,
    suppress_errors: bool = False,
) -> bool:
    if preview_path is None:
        return False
    if preview_cache_root is None:
        return False

    preview_file = Path(preview_path).expanduser()

    try:
        resolved_preview = preview_file.resolve()
    except OSError:
        if suppress_errors:
            return False
        raise

    resolved_root = resolved_preview.parent
    if preview_cache_root is not None:
        try:
            resolved_root = preview_cache_root.expanduser().resolve()
        except OSError:
            if suppress_errors:
                return False
            raise

        if not _is_within_dir(resolved_preview, resolved_root):
            if not allow_path_parent_fallback:
                return False
            resolved_root = resolved_preview.parent

    cleanup_targets = [resolved_preview]
    if source_path is not None:
        resolved_source = Path(source_path).expanduser().resolve()
        cleanup_targets = [
            (resolved_root / f"{candidate_name}.jpg").resolve()
            for candidate_name in preview_name_candidates(resolved_source)
        ]
        if resolved_preview not in cleanup_targets:
            return False
    elif not _looks_like_managed_preview_file(resolved_preview):
        return False

    deleted_any = False
    for cleanup_target in cleanup_targets:
        try:
            cleanup_target.unlink(missing_ok=True)
            deleted_any = True
        except OSError:
            if not suppress_errors:
                raise

    return deleted_any


def clear_preview_cache_dir(
    preview_cache_root: Path | None,
    *,
    suppress_errors: bool = False,
) -> int:
    if preview_cache_root is None:
        return 0

    try:
        preview_root = preview_cache_root.expanduser().resolve()
    except OSError:
        if suppress_errors:
            return 0
        raise

    if not preview_root.exists():
        return 0

    removed_count = 0

    for path in sorted(preview_root.iterdir(), key=lambda candidate: candidate.name.casefold()):
        try:
            if path.is_file() or path.is_symlink():
                if not _looks_like_managed_preview_file(path):
                    continue
                path.unlink(missing_ok=True)
                removed_count += 1
        except OSError:
            if not suppress_errors:
                raise

    return removed_count


def _is_within_dir(path: Path, candidate_dir: Path) -> bool:
    try:
        path.relative_to(candidate_dir)
        return True
    except ValueError:
        return False


def _looks_like_managed_preview_file(path: Path) -> bool:
    stem = path.stem.casefold()
    return path.suffix.casefold() == ".jpg" and len(stem) == 40 and all(char in "0123456789abcdef" for char in stem)


def extract_capture_time(image: Image.Image) -> str | None:
    exif = image.getexif()
    if not exif:
        return None

    for tag in CAPTURE_TIME_EXIF_TAGS:
        value = exif.get(tag)
        if value:
            return str(value)

    return None


def preview_capabilities() -> dict[str, str]:
    return {
        "heif_decoder": "pillow-heif" if register_heif_opener is not None else "none",
        "raw_decoder": "rawpy" if rawpy is not None else "none",
    }


def generate_previews_parallel(
    source_paths: Sequence[Path], 
    preview_dir: Path, 
    *, 
    max_workers: int | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    raw_preview_mode: str = DEFAULT_RAW_PREVIEW_MODE,
) -> list[PreviewResult]:
    """Generate previews in parallel, optionally reporting progress via *progress_callback(completed, total)*.

    Uses ProcessPoolExecutor because preview generation is CPU-bound
    (image decoding, resizing, JPEG encoding) and Python's GIL prevents
    threads from running this work in true parallel.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    total = len(source_paths)
    if total == 0:
        return []

    if total == 1:
        result = generate_preview(source_paths[0], preview_dir, raw_preview_mode=raw_preview_mode)
        if progress_callback is not None:
            progress_callback(1, 1)
        return [result]

    import os
    workers = max_workers or max(4, (os.cpu_count() or 4))

    # Map futures back to their original index so results stay ordered.
    results: list[PreviewResult | None] = [None] * total

    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_index = {
            executor.submit(generate_preview, path, preview_dir, raw_preview_mode=raw_preview_mode): idx
            for idx, path in enumerate(source_paths)
        }
        completed = 0
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                results[idx] = PreviewResult(
                    path=None,
                    status="failed",
                    width=None,
                    height=None,
                    capture_time=None,
                    error_text=str(exc),
                )
            completed += 1
            if progress_callback is not None:
                progress_callback(completed, total)

    return results  # type: ignore[return-value]
