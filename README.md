# ShotSieve

ShotSieve is a local-first photo culling app. It analyzes images on your
machine, ranks them with optional learned image-quality models, and gives you
the final keep/reject decision in a visual review workflow. The default server
binds to loopback; your photo library is not uploaded by ShotSieve.

<img width="1844" height="1251" alt="ShotSieve review UI" src="https://github.com/user-attachments/assets/31cb0d90-6ec9-4e2e-88f9-9ecfbaecdca6" />

## What it does

- Scans local photo folders and builds a reusable catalog.
- Scores supported images with learned quality models when the optional model
  runtime is available.
- Compares models on the same library.
- Provides Review filters, keep/reject decisions, and copy, move, and delete
  actions.
- Keeps the catalog, previews, model caches, and review UI local by default.

## Choose a package or runtime

Use the package that matches the runtime you have actually validated:

| Runtime | Best fit | Boundary |
|---|---|---|
| CPU | Any supported source or packaged install | Broadest fallback; learned models still need enough RAM and disk. |
| NVIDIA CUDA | Supported NVIDIA GPUs | The current x64 path uses PyTorch 2.14.0 from the cu130 index and covers the architectures published by that wheel. Driver, VRAM, and model support still apply. |
| Intel XPU | Intel GPU/OS combinations in the pinned PyTorch XPU matrix | Available as Windows/Linux runtime packs and as a source-install track. |
| AMD ROCm | AMD's current OS/driver matrix | The Windows/Linux packs use the stable ROCm 10.0 PyTorch packages with `gfx1103` kernels, including the Radeon 780M. Hardware support remains specific to AMD's live matrix. |
| Apple MPS | Apple Silicon Macs supported by the installed PyTorch build | Available as a macOS arm64 runtime pack; Intel Macs use CPU. |

A runtime pack proves that the software stack can be built, not that every GPU
from that vendor is supported. Auto mode can fall back to CPU and reports the
reason; an explicit `cuda`, `xpu`, `rocm`, or `mps` request fails when that
runtime is unusable. See the [build guide](docs/building.md),
[Intel XPU guide](docs/intel-xpu.md), and [AMD ROCm guide](docs/amd-rocm.md)
for the pinned boundaries and validation commands.

The release matrix contains ten runtime packs:

- Windows: CPU, NVIDIA CUDA, Intel XPU, AMD ROCm
- Linux: CPU, NVIDIA CUDA, Intel XPU, AMD ROCm
- macOS arm64: CPU and Apple MPS

Current launcher names are:

| Platform | Runtime | Launcher |
|---|---|---|
| Windows | CPU | `ShotSieve-CPU.exe` |
| Windows | NVIDIA CUDA | `ShotSieve-NVIDIA-CUDA.exe` |
| Windows | Intel XPU | `ShotSieve-Intel-XPU.exe` |
| Windows | AMD ROCm | `ShotSieve-AMD-ROCm.exe` |
| Linux | CPU | `ShotSieve-CPU` |
| Linux | NVIDIA CUDA | `ShotSieve-NVIDIA-CUDA` |
| Linux | Intel XPU | `ShotSieve-Intel-XPU` |
| Linux | AMD ROCm | `ShotSieve-AMD-ROCm` |
| macOS | CPU | `ShotSieve-CPU` |
| macOS | Apple MPS | `ShotSieve-Apple-MPS` |

The current target and launcher names are required for fresh downloads. Retired
vendor-only names and target aliases are not accepted by the current release
tooling.

## Quick start

For a source or editable install:

```bash
python -m pip install -e .
python -m pip install -e ".[learned-iqa,format-loaders]"
shotsieve-desktop
```

The first command is enough for cataloging and Review. The optional extras add
learned-IQA scoring and HEIF/RAW loaders. For a tested, pinned source setup
for learned-IQA, follow [docs/building.md](docs/building.md#tested-source-install-for-learned-iqa).

For a downloaded runtime pack, launch the bundled `ShotSieve-*` executable
from the archive instead of `shotsieve-desktop`.

Useful options:

```bash
shotsieve-desktop --data-dir ./shot-data
shotsieve-desktop --model-cache-dir ./model-cache
shotsieve-desktop --host 127.0.0.1 --port 9001 --no-browser
```

Source checkouts and frozen bundles default to a `data/` directory next to the
checkout or launcher. Installed packages outside a checkout use the platform
app-data directory. `--data-dir` overrides either choice. The database is
`data/shotsieve.db`, previews are under `data/previews/`, and runtime sidecars
and install logs are under `data/runtime/`.

Portable runtime packs are torchless: PyTorch is installed into the selected
sidecar on first use after interactive consent or explicit
`SHOTSIEVE_BOOTSTRAP_AUTO_INSTALL_TORCH=1`. A declined, offline, or failed
install leaves Catalog and Review available, but learned-IQA scoring disabled.
Model weights are separate downloads and are never included in a release
archive. Use Settings > **Prepare selected model** after the runtime is ready.

## Typical workflow

1. Open `Library` and add one or more photo folders.
2. Run `Analyze` to scan and, when a model is ready, score supported images.
3. Use `Review` to filter the catalog and make keep/reject decisions.
4. Use `Compare` to evaluate models on the same library when needed.
5. Use `Settings` for runtime information, model preparation, missing-entry
   cleanup, resource settings, and decision CSV export.

The supported model catalog is intentionally small:

- `topiq_nr` - TOPIQ, the default all-rounder.
- `clipiqa` - a complementary CLIP-based scorer.
- `qrealign-mini` - the compact Q-ReAlign Mini model; first use downloads
  approximately 2.2 GB of safetensors plus its processor and configuration.

Model availability is discovered from the installed runtime. A model listed in
the catalog is not a certification of every accelerator; the local runtime
must initialize successfully. Model and checkpoint terms are separate from
ShotSieve's license; see [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).

## Catalog and file-operation safety

An ordinary scan only updates files it discovers in that run. It does not
remove cached rows, scores, or review decisions outside the selected roots,
recursion, extension, or ignore-rule coverage. Enumeration errors are shown as
errors rather than being treated as an empty folder.

Removing verified missing entries is a separate reviewed action in Settings.
Unavailable or unmounted roots are treated as unknown and are not treated as a
list of missing files.

Copy, move, and delete are explicit actions. Their operation results retain
per-file status, error details, and retry safety. Completed mutations remain
visible if a later file fails or a job is cancelled; uncertain mutations are
left for manual inspection rather than silently retried.

Review uses one shared local catalog. The active library scope is shown
separately from **All libraries (global)**, and the queue is paged at 60 photos
by default. Filters include score, format, dimensions, megapixels, file size,
metadata completeness, path, and review state.

The default review server is `127.0.0.1:8765` and accepts requests only from
loopback clients with a loopback Host header. `--host` changes the address the
server binds to; it does not enable remote or LAN access.

## Documentation

- [docs/building.md](docs/building.md) - source installs, testing, runtime
  packs, and release builds.
- [docs/intel-xpu.md](docs/intel-xpu.md) - Intel XPU prerequisites and smoke
  evidence.
- [docs/amd-rocm.md](docs/amd-rocm.md) - AMD ROCm prerequisites and smoke
  evidence.
- [docs/performance-measurement.md](docs/performance-measurement.md) - opt-in
  catalog, scan, preview, and model-performance measurements.
- [docs/frontend-workflows.md](docs/frontend-workflows.md) - frontend module
  ownership and public workflow facade.
- [docs/accessibility-checklist.md](docs/accessibility-checklist.md) - manual
  visual-usability checks; it is not a broad accessibility conformance claim.
- [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) - model, package, and
  runtime license boundaries.
- [CHANGELOG.md](CHANGELOG.md) - release history.

## License

ShotSieve is licensed under the [GNU Affero General Public License v3.0 or
later](LICENSE). Optional learned-IQA packages, model weights, base models,
vendor runtimes, and drivers have their own terms. Review those terms before
redistribution or commercial use.
