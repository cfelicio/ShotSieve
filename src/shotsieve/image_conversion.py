"""Shared image conversion rules for previews and learned-IQA inputs."""

from __future__ import annotations

from PIL import Image, ImageOps


# Transparent pixels need a deterministic background because previews and model
# inputs are both RGB. White keeps transparent image edges from becoming dark
# halos and is also the background documented to users.
TRANSPARENCY_MATTE = (255, 255, 255)

# Stored previews and scores are tied to this conversion policy. Increment the
# version whenever the resulting RGB pixels can change.
IMAGE_CONVERSION_VERSION = "rgba-white-matte-v1"

_HIGH_BIT_GRAYSCALE_MODES = {"I;16", "I;16L", "I;16B", "I;16N"}


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
    "IMAGE_CONVERSION_VERSION",
    "TRANSPARENCY_MATTE",
    "prepare_image_for_rgb",
]
