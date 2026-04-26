from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_PREVIEW_DIRNAME = "previews"
DEFAULT_RAW_PREVIEW_MODE = "auto"
RAW_PREVIEW_MODES = ("fast", "auto", "high-quality")

# ── Canonical extension sets (single source of truth) ──────────────────
# All other modules should import from here instead of defining their own.

RAW_CAMERA_EXTENSIONS = frozenset({
    ".3fr", ".arw", ".cr2", ".cr3", ".dng",
    ".nef", ".orf", ".raf", ".rw2", ".raw",
})
HEIF_EXTENSIONS = frozenset({".heic", ".heif"})
PIL_PREVIEWABLE_EXTENSIONS = frozenset({".jpeg", ".jpg", ".png", ".tif", ".tiff"})

#: All formats for which a generated JPEG preview can be produced.
ALL_PREVIEWABLE_EXTENSIONS = RAW_CAMERA_EXTENSIONS | HEIF_EXTENSIONS | PIL_PREVIEWABLE_EXTENSIONS

#: Formats where we strongly prefer a generated preview over the source
#: file (RAW/HEIF/TIFF are browser-fragile or decoding-heavy).
PREVIEW_PRIORITY_EXTENSIONS = RAW_CAMERA_EXTENSIONS | HEIF_EXTENSIONS | frozenset({".tif", ".tiff"})

#: Formats browsers can render natively.
BROWSER_SAFE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".avif"})

#: Formats that ``_load_single_image`` (PIL) can open without optional deps.
PIL_ANALYSIS_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"})

DEFAULT_SUPPORTED_EXTENSIONS = tuple(sorted(
    RAW_CAMERA_EXTENSIONS | HEIF_EXTENSIONS | PIL_PREVIEWABLE_EXTENSIONS
))


@dataclass(slots=True)
class AppConfig:
    db_path: Path
    preview_dir: Path
    supported_extensions: tuple[str, ...]
    raw_preview_mode: str


def resolve_db_path(raw_path: str) -> Path:
    return Path(raw_path).expanduser().resolve()


def resolve_preview_dir(raw_path: str | None, *, db_path: Path) -> Path:
    if raw_path:
        return Path(raw_path).expanduser().resolve()

    return (db_path.parent / DEFAULT_PREVIEW_DIRNAME).resolve()


def parse_extensions(raw_extensions: str | None) -> tuple[str, ...]:
    if raw_extensions is None:
        return DEFAULT_SUPPORTED_EXTENSIONS

    parsed: list[str] = []

    for value in raw_extensions.split(","):
        extension = value.strip().casefold()
        if not extension:
            continue
        if extension in {"raw", ".raw"}:
            parsed.extend(RAW_CAMERA_EXTENSIONS)
            continue
        if not extension.startswith("."):
            extension = f".{extension}"
        parsed.append(extension)

    unique_extensions = tuple(sorted(set(parsed)))
    if not unique_extensions:
        raise ValueError("At least one scan extension must be provided")

    return unique_extensions


def normalize_raw_preview_mode(raw_mode: str | None) -> str:
    normalized = (raw_mode or DEFAULT_RAW_PREVIEW_MODE).strip().casefold()
    if normalized not in RAW_PREVIEW_MODES:
        raise ValueError(f"raw_preview_mode must be one of: {', '.join(RAW_PREVIEW_MODES)}")
    return normalized


def build_config(
    raw_db_path: str,
    *,
    raw_preview_dir: str | None = None,
    raw_extensions: str | None = None,
    raw_preview_mode: str | None = None,
) -> AppConfig:
    db_path = resolve_db_path(raw_db_path)
    return AppConfig(
        db_path=db_path,
        preview_dir=resolve_preview_dir(raw_preview_dir, db_path=db_path),
        supported_extensions=parse_extensions(raw_extensions),
        raw_preview_mode=normalize_raw_preview_mode(raw_preview_mode),
    )