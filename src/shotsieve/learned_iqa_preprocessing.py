from __future__ import annotations

import functools
import math
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageOps


def _is_cuda_tensor_device(tensor_device: object | None) -> bool:
    if tensor_device is None:
        return False

    device_type = getattr(tensor_device, "type", None)
    if isinstance(device_type, str):
        return device_type.casefold() == "cuda"

    return str(tensor_device).strip().casefold().startswith("cuda")


def _load_single_image(path: Path, image_size: int) -> np.ndarray:
    """Load and preprocess a single image for model inference."""
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
        return np.asarray(image, dtype=np.float32) / 255.0


def _arrays_to_tensor(arrays: list[np.ndarray], *, torch_module, tensor_device=None, use_channels_last: bool = False):
    """Convert pre-loaded numpy arrays into a stacked batch tensor."""
    tensors = [torch_module.from_numpy(array).permute(2, 0, 1) for array in arrays]
    batch_tensor = torch_module.stack(tensors, dim=0)
    uses_cuda_transfer = _is_cuda_tensor_device(tensor_device)

    channels_last = getattr(torch_module, "channels_last", None)
    if use_channels_last and channels_last is not None:
        to_method = getattr(batch_tensor, "to", None)
        if callable(to_method):
            batch_tensor = to_method(memory_format=channels_last)

    if tensor_device is not None:
        if uses_cuda_transfer:
            pin_memory = getattr(batch_tensor, "pin_memory", None)
            if callable(pin_memory):
                batch_tensor = pin_memory()
            to_method = getattr(batch_tensor, "to", None)
            if callable(to_method):
                batch_tensor = to_method(tensor_device, non_blocking=True)
        else:
            to_method = getattr(batch_tensor, "to", None)
            if callable(to_method):
                batch_tensor = to_method(tensor_device)

    return batch_tensor


def load_batch_tensor(
    image_paths: Sequence[Path],
    *,
    image_size: int,
    torch_module,
    tensor_device=None,
    executor=None,
    use_channels_last: bool = False,
):
    from concurrent.futures import ThreadPoolExecutor

    loader = functools.partial(_load_single_image, image_size=image_size)
    if executor is not None:
        arrays = list(executor.map(loader, image_paths))
    else:
        with ThreadPoolExecutor(max_workers=len(image_paths)) as pool:
            arrays = list(pool.map(loader, image_paths))

    return _arrays_to_tensor(
        arrays,
        torch_module=torch_module,
        tensor_device=tensor_device,
        use_channels_last=use_channels_last,
    )


def flatten_tensor(value) -> list[float]:
    flat = value.detach().cpu().reshape(-1).tolist()
    return [float(item) for item in flat]


def confidence_values(distribution, *, torch_module) -> list[float | None]:
    if distribution is None:
        return []

    values = distribution.detach().cpu()
    if values.ndim == 1:
        values = values.unsqueeze(0)

    if values.ndim != 2 or values.shape[1] <= 1:
        return [None] * values.shape[0]

    row_sums = values.sum(dim=1, keepdim=True)
    if not torch_module.allclose(row_sums, torch_module.ones_like(row_sums), atol=1e-3):
        probabilities = torch_module.softmax(values, dim=1)
    else:
        probabilities = values.clamp_min(1e-12)

    entropy = -(probabilities * probabilities.log()).sum(dim=1)
    normalized_entropy = entropy / math.log(probabilities.shape[1])
    confidence = (1.0 - normalized_entropy).clamp(0.0, 1.0) * 100.0
    return [float(item) for item in confidence.tolist()]


def normalize_score(raw_score: float, *, score_range: str, lower_better: bool) -> float:
    lower_bound, upper_bound = parse_score_range(score_range)
    if upper_bound <= lower_bound:
        return max(0.0, min(100.0, raw_score))

    clamped = min(max(raw_score, lower_bound), upper_bound)
    normalized = ((clamped - lower_bound) / (upper_bound - lower_bound)) * 100.0
    if lower_better:
        normalized = 100.0 - normalized
    return max(0.0, min(100.0, normalized))


def parse_score_range(score_range: str) -> tuple[float, float]:
    cleaned = score_range.replace("~", "")
    parts = [part.strip() for part in cleaned.split(",") if part.strip()]
    if len(parts) != 2:
        return 0.0, 1.0

    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        return 0.0, 1.0


__all__ = [
    "_arrays_to_tensor",
    "_load_single_image",
    "confidence_values",
    "flatten_tensor",
    "load_batch_tensor",
    "normalize_score",
    "parse_score_range",
]
