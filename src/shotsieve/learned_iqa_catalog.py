from __future__ import annotations

from typing import Sequence


DEFAULT_BATCH_SIZE = 4
DEFAULT_MODEL_NAME = "topiq_nr"
DEFAULT_INPUT_SIZE = 384
DEFAULT_DEVICE_POLICY = "auto (platform-aware accelerator, then cpu)"
DEFAULT_INPUT_SIZES = {
    "topiq_nr": 384,
    "topiq_nr-flive": 384,
    "topiq_nr-spaq": 384,
}
MAX_BATCH_SIZES = {
    "qalign": 1,
    "qualiclip": 1,
}
MODEL_WEIGHT_MB = {
    "qalign": 7000,
    "qualiclip": 1200,
    "clipiqa": 600,
    "tres": 400,
    "topiq_nr": 80,
    "topiq_nr-flive": 80,
    "topiq_nr-spaq": 80,
    "arniqa": 100,
    "arniqa-spaq": 100,
}
PER_IMAGE_ACTIVATION_MB = {
    "qalign": 800,
    "qualiclip": 120,
    "clipiqa": 60,
    "tres": 50,
    "topiq_nr": 8,
    "topiq_nr-flive": 8,
    "topiq_nr-spaq": 8,
    "arniqa": 10,
    "arniqa-spaq": 10,
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
SUPPORTED_MODEL_NAMES = tuple(dict.fromkeys(MODEL_NAME_ALIASES.values()))
MODERN_MODEL_NAMES = (
    "topiq_nr",
    "arniqa",
    "qalign",
)
UI_MODEL_CATALOG = (
    "topiq_nr",
    "clipiqa",
    "qalign",
)
_SUPPORTED_RUNTIME_TARGETS = ("auto", "cpu", "cuda", "xpu", "directml", "mps", "nvidia", "amd", "intel", "apple")


def supported_learned_models() -> tuple[str, ...]:
    return SUPPORTED_MODEL_NAMES


def supported_runtime_targets() -> tuple[str, ...]:
    return _SUPPORTED_RUNTIME_TARGETS


def normalize_model_name(model_name: str) -> str:
    normalized = model_name.strip().casefold().replace(" ", "")
    return MODEL_NAME_ALIASES.get(normalized, normalized)


def preferred_model_names(models: set[str]) -> list[str]:
    return [model for model in MODERN_MODEL_NAMES if model in models]


def is_model_runtime_compatible(model_name: str, *, torch_version: str | None, runtime: str | None = None) -> bool:
    _ = torch_version
    normalized_model = normalize_model_name(model_name)
    normalized_runtime = (runtime or "").strip().casefold()

    if normalized_model == "qalign" and normalized_runtime in {"cpu", "directml"}:
        return False

    return True


def runtime_compatible_model_names(model_names: Sequence[str], *, torch_version: str | None, runtime: str | None = None) -> list[str]:
    return [
        model_name
        for model_name in model_names
        if is_model_runtime_compatible(model_name, torch_version=torch_version, runtime=runtime)
    ]


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_DEVICE_POLICY",
    "DEFAULT_INPUT_SIZE",
    "DEFAULT_INPUT_SIZES",
    "DEFAULT_MODEL_NAME",
    "DEVICE_TARGET_ALIASES",
    "MAX_BATCH_SIZES",
    "MODEL_NAME_ALIASES",
    "MODEL_WEIGHT_MB",
    "MODERN_MODEL_NAMES",
    "PER_IMAGE_ACTIVATION_MB",
    "SUPPORTED_MODEL_NAMES",
    "UI_MODEL_CATALOG",
    "is_model_runtime_compatible",
    "normalize_model_name",
    "preferred_model_names",
    "runtime_compatible_model_names",
    "supported_learned_models",
    "supported_runtime_targets",
]
