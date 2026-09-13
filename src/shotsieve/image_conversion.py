"""Shared image conversion rules for previews and learned-IQA inputs."""

from __future__ import annotations

from contextlib import contextmanager
import sys
import threading
from pathlib import Path
from typing import Iterator
import warnings

from PIL import Image, ImageOps


# Transparent pixels need a deterministic background because previews and model
# inputs are both RGB. White keeps transparent image edges from becoming dark
# halos and is also the background documented to users.
TRANSPARENCY_MATTE = (255, 255, 255)

# Stored previews and scores are tied to this conversion policy. Increment the
# version whenever the resulting RGB pixels can change.
IMAGE_CONVERSION_VERSION = "rgba-white-matte-v1"

# A source image is decoded before Pillow can resize it.  Keep this limit
# below Pillow's default decompression-bomb threshold so normal preview and
# model-input conversion cannot allocate hundreds of megabytes for one file.
# Existing generated previews are already bounded and are selected before a
# source decode, so this only applies to exceptional fallback inputs.
MAX_DECODE_PIXELS = 40_000_000

_IMAGE_HEADER_WARNING_LOCK = threading.Lock()

_HIGH_BIT_GRAYSCALE_MODES = {"I;16", "I;16L", "I;16B", "I;16N"}


class ImageDecodeLimitError(ValueError):
    """Raised when decoding a source would exceed the fixed pixel budget."""

    def __init__(
        self,
        path: str | Path,
        *,
        width: int | None = None,
        height: int | None = None,
        detail: str | None = None,
        max_pixels: int = MAX_DECODE_PIXELS,
    ) -> None:
        self.path = Path(path)
        self.width = width
        self.height = height
        self.max_pixels = max_pixels

        if width is not None and height is not None:
            pixels = width * height
            message = (
                f"Image decode refused for '{self.path.name}': {width:,} x {height:,} "
                f"({pixels:,} pixels) exceeds the safe decode budget of "
                f"{max_pixels:,} pixels. Use an existing bounded preview or a smaller source."
            )
        else:
            message = f"Image decode refused for '{self.path.name}': {detail or 'the decoder rejected an oversized image.'}"

        super().__init__(message)


def enforce_decode_budget(
    path: str | Path,
    width: int,
    height: int,
    *,
    max_pixels: int = MAX_DECODE_PIXELS,
) -> None:
    """Reject an image before a conversion can materialize its full pixels."""
    if width < 1 or height < 1:
        return
    if width * height > max_pixels:
        raise ImageDecodeLimitError(
            path,
            width=width,
            height=height,
            max_pixels=max_pixels,
        )


def _format_captured_warnings(captured: list[warnings.WarningMessage]) -> str | None:
    messages: list[str] = []
    for warning in captured:
        message = str(warning.message).strip()
        if not message:
            continue
        category = getattr(warning.category, "__name__", "Warning")
        messages.append(f"{category}: {message}")
    return " | ".join(messages) or None


@contextmanager
def open_image_with_warnings(
    path: object,
    *,
    diagnostic_path: str | Path | None = None,
) -> Iterator[tuple[Image.Image, str | None]]:
    """Open an image and return warnings emitted during header inspection.

    Pillow's warning filters are process-global.  The lock keeps this short
    header-inspection section isolated when learned-IQA loads images in
    worker threads.  The image remains open after the warning scope ends, so
    expensive conversion is not serialized and warnings from other work
    cannot be assigned to this file.
    """
    image_context = None
    image = None
    captured: list[warnings.WarningMessage] = []

    with _IMAGE_HEADER_WARNING_LOCK:
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            try:
                image_context = Image.open(path)
                enter_method = getattr(image_context, "__enter__", None)
                image = enter_method() if callable(enter_method) else image_context
            except Image.DecompressionBombError as exc:
                raise ImageDecodeLimitError(
                    diagnostic_path or str(path),
                    detail=str(exc),
                ) from exc
            captured = recorded

    warning_text = _format_captured_warnings(captured)
    try:
        yield image, warning_text
    except BaseException:
        exc_info = sys.exc_info()
        exit_method = getattr(image_context, "__exit__", None)
        if callable(exit_method):
            if exit_method(*exc_info):
                return
        else:
            close_method = getattr(image_context, "close", None)
            if callable(close_method):
                close_method()
        raise
    else:
        exit_method = getattr(image_context, "__exit__", None)
        if callable(exit_method):
            exit_method(None, None, None)
        else:
            close_method = getattr(image_context, "close", None)
            if callable(close_method):
                close_method()


def prepare_image_for_rgb(
    image: Image.Image,
    *,
    apply_exif_orientation: bool = True,
) -> Image.Image:
    """Return an oriented RGB image using the shared alpha/mode policy.

    Palette transparency and byte alpha tables are normalized by Pillow's
    RGBA conversion. High-bit grayscale is explicitly reduced to 8-bit first
    so its midtones remain proportional instead of being truncated.
    """
    if apply_exif_orientation:
        image = ImageOps.exif_transpose(image)

    if image.mode in _HIGH_BIT_GRAYSCALE_MODES:
        image = image.point(lambda value: value / 257).convert("L")

    if image.mode == "RGB":
        return image

    rgba = image.convert("RGBA")
    matte = Image.new("RGBA", rgba.size, (*TRANSPARENCY_MATTE, 255))
    return Image.alpha_composite(matte, rgba).convert("RGB")


__all__ = [
    "ImageDecodeLimitError",
    "IMAGE_CONVERSION_VERSION",
    "MAX_DECODE_PIXELS",
    "TRANSPARENCY_MATTE",
    "enforce_decode_budget",
    "open_image_with_warnings",
    "prepare_image_for_rgb",
]
