# -*- mode: python ; coding: utf-8 -*-

import os
import sys
import importlib.util
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules, collect_all

# Add src to sys.path to help PyInstaller find the module
sys.path.insert(0, os.path.abspath("src"))

# Nuclear collection for core dependencies
datas = []
binaries = []
hiddenimports = []

# Collect ShotSieve package
datas += collect_data_files("shotsieve")

# Collect CLIP tokenizer assets used by clipiqa/qualiclip models
if importlib.util.find_spec("clip") is not None:
    datas += collect_data_files("clip")

# Enable torchless CUDA runtime-pack builds when requested by the build pipeline.
skip_bundled_torch = os.environ.get("SHOTSIEVE_SKIP_BUNDLED_TORCH", "").strip().casefold() in {"1", "true", "yes", "on"}


def _entry_text(entry):
    if isinstance(entry, tuple):
        combined = " ".join(str(part) for part in entry if part is not None)
    else:
        combined = str(entry)

    return combined.replace("\\", "/").casefold()


def _is_torch_related(entry):
    normalized = _entry_text(entry)
    return (
        "torchvision" in normalized
        or "torchaudio" in normalized
        or "functorch" in normalized
        or normalized.startswith("torch")
        or "/torch/" in normalized
        or " triton" in normalized
        or "/triton/" in normalized
    )


def _is_torch_test_asset(entry):
    """Exclude PyTorch's interpreter test fixture from production bundles."""
    return "torch/bin/test_interpreter_async.pt" in _entry_text(entry)


def _without_torch_test_assets(entries):
    return [entry for entry in entries if not _is_torch_test_asset(entry)]

# Collect difficult dependencies using collect_all
difficult_packages = ["pyiqa", "numpy", "PIL", "fastapi", "uvicorn", "jinja2", "icecream", "setuptools", "pip"]
if not skip_bundled_torch:
    difficult_packages.extend(["torch", "torchvision"])

for pkg in difficult_packages:
    try:
        if importlib.util.find_spec(pkg) is not None:
            tmp_datas, tmp_binaries, tmp_hiddenimports = collect_all(pkg)
            datas += tmp_datas
            binaries += tmp_binaries
            hiddenimports += tmp_hiddenimports
    except Exception as e:
        print(f"Warning: Failed to collect_all for {pkg}: {e}")

# The Linux PyTorch wheel includes this interpreter test fixture. It is not
# required at runtime and must not be mistaken for an application model file
# by the portable-bundle smoke test.
datas = _without_torch_test_assets(datas)
binaries = _without_torch_test_assets(binaries)

analysis_excludes = []
if skip_bundled_torch:
    datas = [entry for entry in datas if not _is_torch_related(entry)]
    binaries = [entry for entry in binaries if not _is_torch_related(entry)]
    hiddenimports = [entry for entry in hiddenimports if not _is_torch_related(entry)]
    analysis_excludes.extend([
        "torch",
        "torchvision",
        "torchaudio",
        "functorch",
        "triton",
    ])

# Ensure basic imports are included
hiddenimports += [
    "pkg_resources",
    "modulefinder",
    "shotsieve",
    "shotsieve.desktop",
    "shotsieve.web",
    "shotsieve.learned_iqa",
    "uvicorn.protocols.http.h11_impl",
    "uvicorn.protocols.http.httptools_impl",
    "uvicorn.protocols.websockets.websockets_impl",
    "uvicorn.protocols.websockets.wsproto_impl",
    "uvicorn.loop.asyncio",
    "uvicorn.loop.uvloop",
    "email.mime.text",
    "email.mime.multipart"
]

if importlib.util.find_spec("pkg_resources") is not None:
    try:
        hiddenimports += collect_submodules("pkg_resources")
    except Exception:
        pass

a = Analysis(
    ["src/shotsieve/desktop.py"],
    pathex=["src"],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=analysis_excludes,
    noarchive=False,
)
# PyInstaller hooks can add package data after the explicit collect_all calls
# above. Apply the production-bundle filter again to the final TOCs so Torch's
# interpreter test fixture cannot leak into the staged bundle.
a.datas = _without_torch_test_assets(a.datas)
a.binaries = _without_torch_test_assets(a.binaries)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ShotSieve",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="ShotSieve",
)
