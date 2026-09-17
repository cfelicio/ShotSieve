# ShotSieve

ShotSieve is a local-first photo culling app for people who want AI help without handing their library to a cloud service. You point it at a folder, let it analyze the images on your machine, then make the final keep/reject decisions yourself in the desktop review workflow.

<img width="1844" height="1251" alt="image" src="https://github.com/user-attachments/assets/31cb0d90-6ec9-4e2e-88f9-9ecfbaecdca6" />

## Release 0.4.0 highlights

Release 0.4.0 established ShotSieve's current learned-IQA lineup and runtime
direction. The 0.4.x fixes are recorded in [CHANGELOG.md](CHANGELOG.md).

- Replaced the retired Q-Align integration with **Q-ReAlign Mini**
  (`qrealign-mini`), while keeping **TOPIQ (Recommended)** as the default and
  **CLIPIQA** as the complementary comparison model.
- Added Intel XPU and AMD ROCm runtime tracks. These use matching PyTorch
  wheels, drivers, and model validation, and are now packaged for supported
  Windows and Linux release targets.
- Retired the unsupported legacy Windows GPU adapter and removed its package,
  runtime selection, and release target.


## What ShotSieve does

- Scores images with learned image-quality models on supported runtimes
- Lets you compare models on the same library before trusting one for a bigger culling pass
- Gives you a visual `Review` workflow for keep/reject decisions, filtering, and batch actions
- Stays local-first: the review UI runs on loopback and your library stays on your machine

## Who it is for

ShotSieve is aimed at photographers and hobbyists who:

- shoot large folders and need help narrowing them down
- want AI ranking to help find low quality photos (e.g. blurry, out of focus, over/under exposed)
- prefer a local workflow over uploading a library to a hosted service

## Which package should I download?

If you are choosing between the packaged builds, pick the one that matches your hardware:

| Package | Best for | What it means |
|---|---|---|
| `CPU` | Most machines with a supported OS/Python or packaged target | Runs entirely on the processor. Slowest, but the broadest fallback; learned models still need enough RAM, disk, and compatible packages. |
| `NVIDIA / CUDA` | NVIDIA GPUs covered by the pinned PyTorch CUDA build | Current cu130 x64 targets cover Turing (`sm_75`), Ampere (`sm_80`, `sm_86`), Hopper (`sm_90`), and Blackwell (`sm_100`, `sm_120`). Maxwell, Pascal, and Volta are not covered by this cu130 release path. |
| `Intel XPU` | Intel GPU/device and OS combinations in the PyTorch XPU matrix | Current PyTorch 2.14 validation lists Arc A/B, selected Core Ultra Arc graphics, and Data Center GPU Max; driver, OS, and model support still apply. |
| `AMD ROCm` | AMD hardware in the pinned ROCm/PyTorch matrix | ROCm 7.2.1 is Linux-first and lists selected Radeon RX 7000/9000, PRO/AI PRO, and Ryzen AI hardware; Windows is the narrower PyTorch-only path. |
| `Apple Silicon / MPS` | Apple Silicon Macs with an MPS-enabled PyTorch build | Requires a supported macOS/device combination (current PyTorch guidance is macOS 14+); it does not cover Intel Macs or arbitrary Metal devices. |

Practical rule of thumb:

- If you have an NVIDIA GPU, choose **CUDA**.
- If you are on Apple Silicon, choose **MPS**.
- If you are on Windows without a validated CUDA, XPU, or explicitly supported ROCm runtime, choose **CPU**.
- If you just want the most reliable option or are unsure, choose **CPU**.

Intel XPU and AMD ROCm are available both as packaged runtime downloads and
source-install tracks. See [docs/intel-xpu.md](docs/intel-xpu.md) and
[docs/amd-rocm.md](docs/amd-rocm.md) for the pinned install, driver boundary,
and one-image evidence workflows.

### Runtime support boundaries

The package name is not a universal hardware guarantee. Local builds and
GitHub release builds use the same target-specific dependency constraints, but
GitHub runners build the archives on generic machines and cannot test every
GPU generation. A runtime pack being available only proves that its software
stack can be packaged; the device, operating system, driver, model, and VRAM
requirements must still match the relevant vendor matrix.

- **CUDA:** the release path uses PyTorch 2.14.0 from the cu130 index. On the
  current x64 PyTorch matrix, that means Turing and newer through Blackwell
  (`sm_75`, `sm_80`, `sm_86`, `sm_90`, `sm_100`, and `sm_120`). The app checks
  the active GPU's compiled kernels at startup, so a stale cu126 environment
  on an RTX 50-series card is rejected instead of being called compatible.
- **Intel XPU:** use only Intel GPU and OS combinations listed by the pinned
  PyTorch XPU release. The current guide names Arc A/B, Meteor Lake-H, Arrow
  Lake-H, Lunar Lake, Panther Lake with narrower OS requirements, and Data
  Center GPU Max. Other Intel graphics are not certified by the pack.
- **AMD ROCm:** use only hardware in AMD's ROCm 7.2.1 matrix. The current
  Radeon path is selected RX 7000/9000 plus listed PRO/AI PRO products, with
  selected Ryzen AI APUs on the PyTorch path. Linux has the broader stack;
  Windows is limited to the documented PyTorch combination. Older or
  unlisted Radeon, Instinct, and APU devices are not implied to work.
- **Apple MPS:** requires macOS with an MPS-enabled Apple device and matching
  PyTorch build. Intel Macs, non-Apple GPUs, and unsupported macOS versions
  use CPU instead.
- **CPU:** is the fallback when an accelerator is missing, unsupported, out of
  VRAM, or fails model initialization. CPU compatibility still depends on the
  packaged/source platform and the selected model's memory requirements.

Auto mode may fall back to CPU and reports the accelerator reason. An explicit
`cuda`, `xpu`, `rocm`, or `mps` request remains an error when that runtime is
not usable. The detailed pinned instructions and vendor links are in
[docs/building.md](docs/building.md), [docs/intel-xpu.md](docs/intel-xpu.md),
and [docs/amd-rocm.md](docs/amd-rocm.md).

GitHub runtime-pack releases publish ten downloads: Windows and Linux CPU,
NVIDIA/CUDA, Intel/XPU, and AMD/ROCm, plus macOS arm64 CPU and Apple/MPS.
They also publish the Python source distribution and wheel plus a checksummed
bootstrap manifest. Model weights are never included in release archives.

## Quick start with `shotsieve-desktop`

For source installs and editable installs, use the Python entry point when you want ShotSieve to manage the local DB location and open the review UI directly:

```bash
shotsieve-desktop
```

Downloaded bundles use platform- and runtime-specific launcher names instead of the `shotsieve-desktop` command:

| Platform | Runtime pack | Launcher |
|---|---|---|
| Windows | CPU | `ShotSieve-CPU.exe` |
| Windows | NVIDIA / CUDA | `ShotSieve-NVIDIA.exe` |
| Windows | Intel / XPU | `ShotSieve-Intel.exe` |
| Windows | AMD / ROCm | `ShotSieve-AMD.exe` |
| Linux | CPU | `ShotSieve-CPU` |
| Linux | NVIDIA / CUDA | `ShotSieve-NVIDIA` |
| Linux | Intel / XPU | `ShotSieve-Intel` |
| Linux | AMD / ROCm | `ShotSieve-AMD` |
| macOS | CPU | `ShotSieve-CPU` |
| macOS | Apple Silicon / MPS | `ShotSieve-MPS` |

So the quick rule is:

- if you installed ShotSieve from Python packaging, launch `shotsieve-desktop`
- if you downloaded a runtime pack, launch the bundled `ShotSieve-*` app inside that archive

Useful flags:

```bash
shotsieve-desktop --data-dir ./shot-data
shotsieve-desktop --model-cache-dir ./model-cache
shotsieve-desktop --host 127.0.0.1 --port 9001 --no-browser
```

Default data location for downloaded bundles:

- Downloaded Windows, Linux, and macOS bundles keep writable app state in a local `data/` folder next to the launcher
- Editable source checkouts (`pip install -e .` from this repository) now also default to a local `data/` folder at the project root
- The main database lives at `data/shotsieve.db`
- Generated previews live under `data/previews/`
- Runtime sidecars, repairs, and related pip logs live under `data/runtime/`

Installed packages outside a source checkout still fall back to an OS-level app data location (`%LOCALAPPDATA%\ShotSieve` on Windows, falling back to `%APPDATA%` if needed, otherwise `~/.shotsieve`). On any install style, `--data-dir` overrides the default.

## How you use it

The intended workflow is simple and visual:

1. Launch `shotsieve-desktop`
2. Choose a photo folder in `Library`
3. Run `Analyze` to scan the folder and score supported images
4. Start in `Review` and work through the images with keep/reject decisions
5. Use `Compare` if you want to test different models on the same library
6. Use `Settings` for runtime info, resource profile, and maintenance actions

If you are using a downloaded runtime pack, replace step 1 with the matching bundled launcher from the table above.

ShotSieve is meant to speed up your judgment, not replace it. The app helps surface likely throwaways; you still make the final call.

### Scan safety and cached catalog entries

An ordinary scan updates files it can actually discover. It preserves catalog rows, scores, and keep/reject decisions that are outside the selected scan coverage or temporarily unavailable, including when a scan is nonrecursive or uses changed extension/ignore filters. If the root or a subdirectory cannot be enumerated, the scan reports the path and operating-system error instead of treating the location as an empty folder.

Removing verified missing entries is a separate reviewed-cleanup workflow. In Settings, choose **Review Missing Entries** to preview every candidate and any affected review decisions for the selected root(s), then explicitly confirm the removal. ShotSieve checks that each root can be fully enumerated; unavailable or unmounted roots are reported as unknown and are never treated as missing files. The apply step also rejects a stale preview if sources, candidates, or review state changed.

Failed or cancelled scans remain visible in the recent scan diagnostics with the affected root, timestamps, error, and processed counts. Multi-root scans commit each successful root independently; if a later root fails, the scan job is failed and reports successful, failed, and not-processed roots instead of presenting the job as fully successful. Cancellation keeps any work completed before the cancellation request.

Previews and learned-IQA inputs use the same image conversion policy. EXIF orientation is applied, palette and alpha transparency are composited onto a white matte, and high-bit grayscale is scaled to 8-bit without truncating its midtones. The conversion policy is versioned: an older preview or score is regenerated or rescored before it is reused, while catalog review decisions remain unchanged.

Fallback source decoding defaults to a 64-million-pixel (64 MP) budget before conversion, because resizing cannot prevent the full source from being materialized first. Adjust **Maximum Source Decode (MP)** in Settings when working with larger originals; the accepted range is 1–256 MP. Existing ready previews remain usable, and an oversized source without one is recorded as a per-file preview or scoring failure with recovery guidance. RAW files still prefer a valid embedded thumbnail, including when full sensor demosaicing would exceed the selected limit. Decoder warnings are retained with the affected file when the decoder exposes them, without redirecting concurrent worker output into another file's diagnostic.

### File-operation results

Copy, move, and delete operations keep the existing aggregate counts and now also return a per-file result through the operation status/result endpoints. Each result identifies the source, destination when known, action stage, OS error details, guarded source/destination state (`present`, `missing`, or `unknown`), observation errors, and whether retrying that file is safe. Preview cleanup warnings are reported separately from transfer failures. A cancelled or fatally stopped operation retains completed rows and every later frozen selection as unprocessed; if a mutation may have happened, both paths and the original plus observation errors are retained and that file is not automatically retried. Catalog failures are reconciled before a move is compensated. An async operation job can therefore be terminally failed while its result endpoint still returns the retained file summary.

The Library workspace retains the latest operation result until you dismiss or replace it. It shows action-specific completed, partial, failed, and unprocessed counts, a bounded list of paths/stages/details, and buttons to copy or download the complete JSON result. Successful selections are removed after a terminal result while failed and unprocessed files remain selected; only files marked safe by the operation contract can be retried. Rejected-file deletion uses the same tracked operation flow as selected deletion, copying, and moving.

Retry keeps the original Review scope, obtains a matching selection revision for each 500-file chunk, and retains the combined result when a later chunk fails or is cancelled. If a job status or result cannot be confirmed, ShotSieve keeps the job recoverable, blocks new mutations, and provides **Check status**; once the job is terminal, the workspace is refreshed. Downloading the JSON result remains available after cancellation.

Settings also provides **Download decisions CSV** for a selected library root. It exports every approved, rejected, or both marked decisions in that root—not just the current Review page—with `file_id`, `decision`, `source_path`, `library_root`, and `decision_updated_time` columns. The export is read-only, spreadsheet-friendly UTF-8, and formula-safe for text cells.

## Models and runtimes

The supported in-app model catalog is intentionally small:

- `topiq_nr` is the default model
- `clipiqa` is a fast secondary option for quick comparisons
- `qrealign-mini` is the compact Q-ReAlign Mini option for CPU and compatible accelerators

TReS, QualiCLIP, ARNIQA, and other PyIQA names are not supported for new scoring or comparison runs. Q-ReAlign Mini is a distinct Qwen3.5-VL-based checkpoint and is catalog-compatible with CPU, CUDA, ROCm, XPU, and MPS, but that is not blanket certification: the local runtime must initialize it successfully and each release claim needs target-specific validation. Stored scores retain their saved model name; disabling or replacing a model does not relabel or delete those rows.

Runtime names you may see in settings or developer docs:

- `cpu`: no GPU acceleration
- `cuda`: NVIDIA GPU acceleration
- `xpu`: Intel accelerator path where the local PyTorch runtime exposes it
- `rocm`: AMD ROCm path where the local HIP-enabled PyTorch runtime exposes it
- `mps`: Apple Silicon GPU acceleration

The Settings model list is populated from runtime discovery. If discovery or initialization is unavailable, the list stays empty instead of claiming that a model is ready. Auto mode may fall back to CPU and reports the failed accelerator reason; an explicitly requested unavailable runtime fails with recovery guidance.

The release and sidecar paths use the tested learned-IQA package set `pyiqa==0.1.16`, `timm==1.0.29`, `huggingface-hub==1.31.0`, `transformers==5.17.0`, and `openai-clip==1.0.1`, `accelerate==1.15.0`, `sentencepiece==0.2.2`, and `einops==0.8.2`. CPU, CUDA, and Apple MPS targets use `torch==2.14.0` with `torchvision==0.29.0`; CUDA selects the cu130 index and CPU selects the PyTorch CPU index. XPU targets use the pinned `2.14.0+xpu` pair from the official XPU index. ROCm targets use AMD's exact ROCm 7.2.1 PyTorch wheels with separate Windows/Linux constraints. The matching target's driver and device support remain required.

The CUDA startup probe checks the active GPU compute capability against the
kernels compiled into the installed PyTorch wheel. This prevents a stale CUDA
12.6 sidecar from being accepted on newer GPUs such as RTX 50-series
`sm_120`; the current cu130 sidecar is selected for the supported PyTorch
architecture set. NVIDIA hardware outside the kernels published by the
selected PyTorch build is reported as unavailable and falls back to CPU, rather
than being advertised as universally supported. A CUDA-compatible driver and
enough VRAM for the selected model are still required.

The manual/weekly model smoke workflow installs these constraints, runs `pip check`, prepares TOPIQ, CLIPIQA, and Q-ReAlign Mini in separate fresh caches, and repeats all checks offline in a new process. It records resolved versions and retains sanitized JSON diagnostics on failure; caches, weights, and generated images are not uploaded.

Runtime setup may occur during an interactive first launch when learned-IQA support is missing; the catalog and Review UI remain usable without it. Model weights remain a separate **Prepare selected model** operation, with the existing first-use and license guidance.

In Settings, **Prepare selected model** downloads any missing assets through the normal learned-IQA backend and validates one generated image using CPU for CPU-compatible models or the selected accelerator for Q-ReAlign Mini. Preparation is for the selected model only; scoring still validates the requested runtime when used. The small readiness record under the app data directory retains the last check (`not_checked`, `preparing`, `prepared`, `failed`, or `runtime_unavailable`), cache paths and effective cache-volume free space, dependency fingerprint, tested runtime, and sanitized recovery diagnostics. If a process ends during preparation, the next startup downgrades the orphaned `preparing` state to an interrupted failure. `/api/options` only reads that record and performs local metadata/disk checks; it does not download assets or construct a model.

Scoring and Compare use the same sanitized diagnostic contract as Prepare. A failed model initialization or job reports the model, requested and actual runtime when known, effective cache paths/volumes, offline flags, exception chain, category, and recovery action through the job status/result API. Unknown causes remain visible without credentials, proxy passwords, Hub tokens, or URL query credentials. The recovery action links back to **Prepare selected model**, which is a last successful check rather than a promise that a later accelerator run will remain available.

Use `--model-cache-dir` when a portable or shared cache root is needed. The option supplies defaults for Hugging Face and Torch subdirectories while preserving explicitly set `HF_HOME`, `HF_HUB_CACHE`, and `TORCH_HOME`; existing caches are not moved. For offline use, prepare the selected model once, shut down ShotSieve, copy the complete compatible cache tree and app-data readiness record, then validate in a fresh process with the relevant offline environment settings.

## Review UI

The review server binds to `127.0.0.1:8765` by default. The UI is organized around four tabs:

1. `Library` for managing multiple workspace source folders, setting directory ignore rules, and launching scan/score actions.
2. `Compare` for side-by-side learned-model benchmarking.
3. `Review` for queue navigation, filtering (by score, format, megapixels, size, and completeness), sorting, marking, and export/delete flows.
4. `Settings` for runtime info, resource profile, and maintenance actions.

ShotSieve keeps one shared local catalog so previously generated previews and scores can be reused when you return to a folder. The selected `Library` folder(s) define the active scope for Review by default: its discovered, scored, rejected, and selected totals are shown separately from the explicitly labelled **All cached libraries** totals. Review also names the active scope above its filters.

Review keeps navigation bounded even for a large shared catalog: it displays one page at a time (60 photos by default) and loads queue thumbnails only as they approach view. The explicit **All libraries (global)** view can still be slower than a small active library because it intentionally searches the whole catalog.

### Metadata-Led Culling & Filtering
The Review interface supports advanced resolution and format culling tools to isolate target files:
- **Format Groups**: Filter by JPEG, PNG, TIFF, HEIF, RAW, or other extensions.
- **Resolution & Size**: Set Megapixel (MP) ranges, exact width/height constraints, or source file size (MB) ranges.
- **Completeness**: Isolate files with missing/unreadable metadata to investigate scan issues.
- **Sorting**: Sort by Resolution, File Size, Format Name, Width, or Height.
- **Metadata Tag Strip**: The selected photo detail panel displays a chip strip showing the file's format, pixel dimensions, megapixels, aspect ratio, and formatted byte size.

Choose **All libraries (global)** from the Review scope selector only when you intentionally want to browse the full catalog. The standard rejected-file action is always **Delete rejected in this library** and its confirmation names the active library paths; it does not delete rejected files from other cached libraries.

Current keyboard shortcuts in review mode:

| Key | Action |
|---|---|
| `↓` / `→` | Next photo |
| `↑` / `←` | Previous photo |
| `S` | Keep current photo or selected photos |
| `R` | Reject current photo or selected photos |
| `Esc` | Close the lightbox or active overlay |

## For developers and builders

If you are installing from source, running tests, or building release archives, use the dedicated build guide:

- [docs/building.md](docs/building.md)

That guide covers:

- source installs and optional extras
- test and lint commands
- Playwright setup
- Windows runtime-pack build instructions

For contributors investigating large-catalog responsiveness, [docs/performance-measurement.md](docs/performance-measurement.md) describes the opt-in catalog benchmark and safe local real-photo measurements.

The browser-focused frontend checks are intentionally about **visual QA** and visual usability for the photo-review workflow, not broad accessibility conformance claims.

The frontend workflow facade is documented in [docs/frontend-workflows.md](docs/frontend-workflows.md). `app-workflows.js` keeps the stable `ShotSieveWorkflows` public surface and composes the dedicated library, export, compare, and polling modules; feature behavior belongs to the module that owns it.

## Security and licensing

- [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md): third-party model and library licensing
- [LICENSE](LICENSE): GNU Affero General Public License v3.0 or later (AGPLv3+)

ShotSieve uses AI models and libraries with their own terms. The learned-IQA extra currently pins `pyiqa==0.1.16`, whose PolyForm Noncommercial license and included notices must be reviewed with the model/checkpoint terms before commercial use. No model weights are bundled; supported assets may be downloaded into the configured upstream caches. Q-ReAlign Mini remains subject to its model card, Qwen3.5-VL/base-model terms, and the separate PyIQA package terms. See [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) for the per-component audit boundary.

## Project docs

- [README.md](README.md): user-facing overview, quick start, and package selection guide
- [CHANGELOG.md](CHANGELOG.md): release history, starting with the initial `0.1.0` release
- [docs/building.md](docs/building.md): source install, testing, linting, and release-build instructions
- [docs/performance-measurement.md](docs/performance-measurement.md): opt-in catalog, scan, preview, and learned-IQA performance measurement guide
