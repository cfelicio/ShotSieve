from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

DEFAULT_BATCH_SIZE = 4
DEFAULT_MODEL_NAME = "topiq_nr"
DEFAULT_INPUT_SIZE = 384
DEFAULT_DEVICE_POLICY = "auto (platform-aware accelerator, then cpu)"
DEFAULT_INPUT_SIZES = {
    "topiq_nr": 384,
    "clipiqa": 224,
}
MAX_BATCH_SIZES = {
    "topiq_nr": 4,
    "clipiqa": 4,
}
MODEL_WEIGHT_MB = {
    "clipiqa": 600,
    "topiq_nr": 80,
}
PER_IMAGE_ACTIVATION_MB = {
    "clipiqa": 60,
    "topiq_nr": 8,
}
DEVICE_TARGET_ALIASES = {
    "": "auto",
    "auto": "auto",
    "gpu": "auto",
    "cpu": "cpu",
    "cuda": "cuda",
    "cuda:0": "cuda",
    "nvidia": "cuda",
    "xpu": "xpu",
    "intel": "intel",
    "directml": "directml",
    "dml": "directml",
    "amd": "amd",
    "mps": "mps",
    "apple": "apple",
}
MODEL_NAME_ALIASES = {
    "topiq-nr": "topiq_nr",
    "topiq_nr": "topiq_nr",
    "topiq-nr-flive": "topiq_nr-flive",
    "topiq_nr-flive": "topiq_nr-flive",
    "topiq_nr_flive": "topiq_nr-flive",
    "topiq-nr-spaq": "topiq_nr-spaq",
    "topiq_nr-spaq": "topiq_nr-spaq",
    "topiq_nr_spaq": "topiq_nr-spaq",
    "arniqa": "arniqa",
    "arniqa-spaq": "arniqa-spaq",
    "arniqa_spaq": "arniqa-spaq",
    "tres": "tres",
    "clipiqa": "clipiqa",
    "quali-clip": "qualiclip",
    "qualiclip": "qualiclip",
    "q-align": "qalign",
    "qalign": "qalign",
}
_SUPPORTED_RUNTIME_TARGETS = ("auto", "cpu", "cuda", "xpu", "directml", "mps", "nvidia", "amd", "intel", "apple")


@dataclass(frozen=True, slots=True)
class LearnedModelSpec:
    canonical_id: str
    aliases: tuple[str, ...]
    label: str
    description: str
    supported_runtimes: tuple[str, ...]
    input_size: int
    default_batch_size: int
    max_batch_size: int
    resource_labels: tuple[str, ...]
    cache_families: tuple[str, ...]
    first_use_disclosure: str

    def to_payload(self, *, available: bool = False) -> dict[str, object]:
        payload = asdict(self)
        payload["aliases"] = list(self.aliases)
        payload["supported_runtimes"] = list(self.supported_runtimes)
        payload["resource_labels"] = list(self.resource_labels)
        payload["cache_families"] = list(self.cache_families)
        payload["available"] = available
        return payload


_COMMON_RUNTIME_POLICY = ("cpu", "cuda", "xpu", "directml", "mps")
MODEL_CATALOG = (
    LearnedModelSpec(
        canonical_id="topiq_nr",
        aliases=("topiq_nr", "topiq-nr"),
        label="TOPIQ (Recommended)",
        description="Fast, stable all-rounder for general photo-quality ranking.",
        supported_runtimes=_COMMON_RUNTIME_POLICY,
        input_size=384,
        default_batch_size=4,
        max_batch_size=4,
        resource_labels=("moderate model", "moderate memory"),
        cache_families=("Hugging Face Hub cache", "Torch/PyIQA cache"),
        first_use_disclosure="First use may download the ResNet-50 semantic backbone and the CFANet checkpoint cfanet_nr_koniq_res50-9a73138b.pth.",
    ),
    LearnedModelSpec(
        canonical_id="clipiqa",
        aliases=("clipiqa",),
        label="CLIPIQA",
        description="CLIP-based quality scorer for a complementary second opinion.",
        supported_runtimes=_COMMON_RUNTIME_POLICY,
        input_size=224,
        default_batch_size=4,
        max_batch_size=4,
        resource_labels=("larger model", "higher memory"),
        cache_families=("Torch/CLIP cache",),
        first_use_disclosure="First use may download the OpenAI CLIP RN50 checkpoint; plain CLIPIQA uses its packaged prompt pairs and does not add a separate CLIPIQA checkpoint.",
    ),
)

# These are derived views of the single product catalog. The old advanced
# names remain normalizable so historical rows can still be displayed, but
# they are deliberately absent from the supported product set.
SUPPORTED_MODEL_NAMES = tuple(spec.canonical_id for spec in MODEL_CATALOG)
MODERN_MODEL_NAMES = SUPPORTED_MODEL_NAMES
UI_MODEL_CATALOG = SUPPORTED_MODEL_NAMES
_MODEL_SPEC_BY_ID = {spec.canonical_id: spec for spec in MODEL_CATALOG}


def supported_learned_models() -> tuple[str, ...]:
    return SUPPORTED_MODEL_NAMES


def supported_runtime_targets() -> tuple[str, ...]:
    return _SUPPORTED_RUNTIME_TARGETS


def normalize_model_name(model_name: str) -> str:
    normalized = model_name.strip().casefold().replace(" ", "")
    return MODEL_NAME_ALIASES.get(normalized, normalized)


def is_supported_model_name(model_name: str) -> bool:
    return normalize_model_name(model_name) in _MODEL_SPEC_BY_ID


def validate_model_name(model_name: str) -> str:
    canonical = normalize_model_name(model_name)
    if canonical in _MODEL_SPEC_BY_ID:
        return canonical

    supported = ", ".join(SUPPORTED_MODEL_NAMES)
    raise ValueError(
        f"Learned IQA model '{model_name}' is unknown or disabled for new runs. "
        f"Supported models: {supported}."
    )


def model_catalog_payload(*, available_models: Sequence[str] | None = None) -> list[dict[str, object]]:
    available = {
        normalize_model_name(model_name)
        for model_name in (available_models or ())
        if is_supported_model_name(model_name)
    }
    return [spec.to_payload(available=spec.canonical_id in available) for spec in MODEL_CATALOG]


def preferred_model_names(models: set[str]) -> list[str]:
    normalized = {normalize_model_name(model) for model in models}
    return [model for model in SUPPORTED_MODEL_NAMES if model in normalized]


def is_model_runtime_compatible(model_name: str, *, torch_version: str | None, runtime: str | None = None) -> bool:
    _ = torch_version
    normalized_model = normalize_model_name(model_name)
    normalized_runtime = (runtime or "").strip().casefold()
    spec = _MODEL_SPEC_BY_ID.get(normalized_model)
    if spec is None:
        return False
    return not normalized_runtime or normalized_runtime in spec.supported_runtimes


def runtime_compatible_model_names(model_names: Sequence[str], *, torch_version: str | None, runtime: str | None = None) -> list[str]:
    compatible: list[str] = []
    seen: set[str] = set()
    for model_name in model_names:
        normalized = normalize_model_name(model_name)
        if normalized in seen:
            continue
        if is_model_runtime_compatible(normalized, torch_version=torch_version, runtime=runtime):
            compatible.append(normalized)
            seen.add(normalized)
    return compatible


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_DEVICE_POLICY",
    "DEFAULT_INPUT_SIZE",
    "DEFAULT_INPUT_SIZES",
    "DEFAULT_MODEL_NAME",
    "DEVICE_TARGET_ALIASES",
    "MAX_BATCH_SIZES",
    "MODEL_CATALOG",
    "MODEL_NAME_ALIASES",
    "MODEL_WEIGHT_MB",
    "MODERN_MODEL_NAMES",
    "PER_IMAGE_ACTIVATION_MB",
    "SUPPORTED_MODEL_NAMES",
    "UI_MODEL_CATALOG",
    "LearnedModelSpec",
    "is_model_runtime_compatible",
    "model_catalog_payload",
    "normalize_model_name",
    "preferred_model_names",
    "runtime_compatible_model_names",
    "supported_learned_models",
    "supported_runtime_targets",
    "validate_model_name",
]
