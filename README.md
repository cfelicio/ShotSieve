# ShotSieve

ShotSieve is a local-first photo culling app for people who want AI help without handing their library to a cloud service. You point it at a folder, let it analyze the images on your machine, then make the final keep/reject decisions yourself in the desktop review workflow.

<img width="1844" height="1251" alt="image" src="https://github.com/user-attachments/assets/31cb0d90-6ec9-4e2e-88f9-9ecfbaecdca6" />


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
| `CPU` | Any machine, safest fallback | Runs entirely on the processor. Slowest, but the most compatible. |
| `NVIDIA / CUDA` | Windows or Linux machines with an NVIDIA GPU | Best choice when you have a supported NVIDIA card and want the fastest learned-IQA scoring. |
| `Intel XPU` (source install) | Supported Intel accelerators with the matching PyTorch XPU runtime | Source-install path; not a packaged runtime download. |
| `Apple Silicon / MPS` | Recent Macs with Apple Silicon | Best choice on Apple Silicon when you want GPU acceleration without a separate CUDA stack. |

Practical rule of thumb:

- If you have an NVIDIA GPU, choose **CUDA**.
- If you are on Apple Silicon, choose **MPS**.
- If you are on Windows without a validated CUDA or XPU runtime, choose **CPU**.
- If you just want the most reliable option or are unsure, choose **CPU**.

Intel XPU remains a source-install/runtime option today, but it is not one of the packaged runtime downloads listed above. See [docs/intel-xpu.md](docs/intel-xpu.md) for the pinned Windows/Linux install and one-image evidence workflow.

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
| Linux | CPU | `ShotSieve-CPU` |
| Linux | NVIDIA / CUDA | `ShotSieve-NVIDIA` |
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

Fallback source decoding is capped at 40 million pixels before conversion, because resizing cannot prevent the full source from being materialized first. Existing ready previews remain usable; an oversized source without one is recorded as a per-file preview or scoring failure with recovery guidance. RAW files still prefer a valid embedded thumbnail, including when full sensor demosaicing would exceed the limit. Decoder warnings are retained with the affected file when the decoder exposes them, without redirecting concurrent worker output into another file's diagnostic.

### File-operation results

Copy, move, and delete operations keep the existing aggregate counts and now also return a per-file result through the operation status/result endpoints. Each result identifies the source, destination when known, action stage, OS error details, guarded source/destination state (`present`, `missing`, or `unknown`), observation errors, and whether retrying that file is safe. Preview cleanup warnings are reported separately from transfer failures. A cancelled or fatally stopped operation retains completed rows and every later frozen selection as unprocessed; if a mutation may have happened, both paths and the original plus observation errors are retained and that file is not automatically retried. Catalog failures are reconciled before a move is compensated. An async operation job can therefore be terminally failed while its result endpoint still returns the retained file summary.

The Library workspace retains the latest operation result until you dismiss or replace it. It shows action-specific completed, partial, failed, and unprocessed counts, a bounded list of paths/stages/details, and buttons to copy or download the complete JSON result. Successful selections are removed after a terminal result while failed and unprocessed files remain selected; only files marked safe by the operation contract can be retried. Rejected-file deletion uses the same tracked operation flow as selected deletion, copying, and moving.

Retry keeps the original Review scope, obtains a matching selection revision for each 500-file chunk, and retains the combined result when a later chunk fails or is cancelled. If a job status or result cannot be confirmed, ShotSieve keeps the job recoverable, blocks new mutations, and provides **Check status**; once the job is terminal, the workspace is refreshed. Downloading the JSON result remains available after cancellation.

Settings also provides **Download decisions CSV** for a selected library root. It exports every approved, rejected, or both marked decisions in that root—not just the current Review page—with `file_id`, `decision`, `source_path`, `library_root`, and `decision_updated_time` columns. The export is read-only, spreadsheet-friendly UTF-8, and formula-safe for text cells.

## Models and runtimes

The supported in-app model catalog is intentionally small:

- `topiq_nr` is the default model
- `clipiqa` is a fast secondary option for quick comparisons
- `qalign` is the large accelerator-backed option for CUDA and Apple MPS

TReS, QualiCLIP, ARNIQA, and other PyIQA names are not supported for new scoring or comparison runs. Q-Align is not available on CPU or XPU. Older stored scores remain readable and are shown using their saved raw model name; disabling a model does not delete those rows.

Runtime names you may see in settings or developer docs:

- `cpu`: no GPU acceleration
- `cuda`: NVIDIA GPU acceleration
- `xpu`: Intel accelerator path for source installs where the local PyTorch runtime exposes it
- `mps`: Apple Silicon GPU acceleration

The Settings model list is populated from runtime discovery. If discovery or initialization is unavailable, the list stays empty instead of claiming that a model is ready. Auto mode may fall back to CPU and reports the failed accelerator reason; an explicitly requested unavailable runtime fails with recovery guidance.

The release and sidecar paths use the tested learned-IQA package set `pyiqa==0.1.16`, `timm==1.0.29`, `huggingface-hub==1.31.0`, `transformers==5.17.0`, and `openai-clip==1.0.1`, `accelerate==1.15.0`, `sentencepiece==0.2.2`, and `einops==0.8.2`. CPU, CUDA, and Apple MPS targets use `torch==2.14.0` with `torchvision==0.29.0`; CUDA selects the cu130 index and CPU selects the PyTorch CPU index. Windows AMD hardware falls back to CPU until a native ROCm path is validated. The retired DirectML package and release target are not installed or published.

The manual/weekly model smoke workflow installs these constraints, runs `pip check`, prepares TOPIQ and CLIPIQA in separate fresh caches, and repeats both checks offline in a new process. It records resolved versions and retains sanitized JSON diagnostics on failure; caches, weights, and generated images are not uploaded.

Optional AI runtime packages are not downloaded during an ordinary noninteractive launch. The catalog and Review UI remain usable without learned-IQA support. Use Settings > **Install / Repair AI support** for one explicit best-effort runtime installation/repair action; it shows the selected target and runtime/cache paths, retains sanitized failures for retry, and may require a restart. This action installs runtime packages only. Model weights remain a separate **Prepare selected model** operation, with the existing first-use and license guidance.

In Settings, **Prepare selected model** downloads any missing assets through the normal learned-IQA backend and validates one generated image on CPU. Preparation is for the selected model only; it does not make accelerator readiness claims, and scoring still validates the requested runtime when used. The small readiness record under the app data directory retains the last check (`not_checked`, `preparing`, `prepared`, `failed`, or `runtime_unavailable`), cache paths and effective cache-volume free space, dependency fingerprint, tested runtime, and sanitized recovery diagnostics. If a process ends during preparation, the next startup downgrades the orphaned `preparing` state to an interrupted failure. `/api/options` only reads that record and performs local metadata/disk checks; it does not download assets or construct a model.

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

## Security and licensing

- [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md): third-party model and library licensing
- [LICENSE](LICENSE): GNU Affero General Public License v3.0 or later (AGPLv3+)

ShotSieve uses AI models and libraries with their own terms. The learned-IQA extra currently pins `pyiqa==0.1.16`, whose PolyForm Noncommercial license and included notices must be reviewed with the model/checkpoint terms before commercial use. No model weights are bundled; supported assets may be downloaded into the configured upstream caches. Q-Align remains subject to its model-card and underlying base-model terms and is supported only on validated accelerator paths. See [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) for the per-component audit boundary.

## Project docs

- [README.md](README.md): user-facing overview, quick start, and package selection guide
- [CHANGELOG.md](CHANGELOG.md): release history, starting with the initial `0.1.0` release
- [docs/building.md](docs/building.md): source install, testing, linting, and release-build instructions
- [docs/performance-measurement.md](docs/performance-measurement.md): opt-in catalog, scan, preview, and learned-IQA performance measurement guide
