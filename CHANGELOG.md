# Changelog

All notable changes to this project will be documented in this file.

## [0.4.4] - 2026-09-16

### Changed

- Added an adjustable maximum source decode budget to Settings. It
  defaults to 64 MP, is applied consistently to scans, re-scoring, and model
  comparison, and is bounded to prevent unsafe memory requests.
- Expanded the runtime documentation to state the pinned CUDA, Intel XPU,
  AMD ROCm, Apple MPS, and CPU support boundaries. Packaged local/GitHub
  builds are documented as target-specific dependency bundles, not universal
  GPU certification; unsupported accelerators fall back to CPU in Auto mode.

### Fixed

- Fixed successful offline model preparation records being downgraded to
  `not_checked` on the next options request when a required cache directory
  had not yet been created.
- Hardened native lightbox focus restoration across browser dialog timing
  differences, including repeated open requests.
- Hardened scan-job cancellation so a cancellation that arrives during final
  aggregation cannot be reported as a completed scan.
- CUDA startup now validates the active GPU's compute capability against the
  installed PyTorch kernels. Incompatible cached wheels, such as cu126 on an
  `sm_120` GPU, no longer pass the `cuda.is_available()` check and fail later
  during learned-IQA initialization; they are repaired when consent is given
  or reported with a CPU fallback.
- Fixed the Export dialog's **Browse** action so it resolves the folder
  browser through the completed library workflow bridge instead of an
  undefined early-captured handler.
- Fixed export-operation retries to resolve `runTrackedOperation` through the
  same late-bound workflow bridge.
- Preserved caller selection order during export so cancellation retry IDs
  remain correct across platform-dependent scan insertion orders.
- Preserved caller selection order for delete and cache operations so
  cancellation retries remain correct regardless of database ID ordering.
- Made filesystem discovery and parallel scan persistence deterministic across
  operating systems and worker completion order.

## [0.4.3] - 2026-09-16

### Added

- Added packaged Windows and Linux Intel XPU and AMD ROCm runtime targets to
  the release matrix and local Windows release script.
- Added Q-ReAlign Mini to the weekly/manual learned-model smoke matrix so its
  CPU availability is checked alongside TOPIQ and CLIPIQA.

### Changed

- Updated the release documentation and landing page to describe the ten-pack
  CPU, CUDA, XPU, ROCm, and Apple MPS artifact matrix.
- Removed the retired legacy Windows GPU adapter from the runtime resolver,
  release tooling, documentation, and regression coverage.
- Windows release builds now enforce each target's declared Python version and
  automatically select the matching Python launcher entry when building all
  targets together.

### Fixed

- Prevented Python 3.14-built CUDA bundles from emitting PyTorch's
  `torch.jit.load` compatibility warning by enforcing the target's Python 3.13
  environment.
- Fixed lightbox focus restoration so closing the native dialog immediately
  returns focus to the photo that opened it.

### Removed

- Removed the redundant Settings AI Support panel and its status/install API.
  Learned-IQA runtime preparation and selected-model preparation remain
  available through the normal startup and model workflows.

## [0.4.2] - 2026-09-15

### Fixed

- Fixed Q-ReAlign Mini preparation on standard Windows accounts by forcing
  Hugging Face Hub to use copy-based caching, avoiding the WinError 1314
  symlink privilege failure. This may use more disk space but does not require
  Developer Mode or administrator privileges.
- Fixed frozen-runtime installation of source-only `openai-clip` with a
  verified direct source-archive fallback, so pip no longer launches the
  packaged ShotSieve executable as a PEP 517 build interpreter.
- Fixed legacy saved `topiq` settings being reported as retired; they now map
  to `TOPIQ (Recommended)`. Retired-model messaging now clearly distinguishes
  old settings from the current recommended model.
- Added regression coverage for Windows Hub cache policy, WinError 1314
  diagnostics, legacy TOPIQ normalization, and frozen `openai-clip` setup.

## [0.4.1] - 2026-09-15

### Fixed

- Fixed first-run Windows CUDA learned-IQA installation so `timm` and
  `accelerate` no longer ask pip to replace the already-loaded PyTorch runtime
  or its locked CUDA DLLs. Their direct runtime dependencies are installed
  explicitly by the sidecar bootstrap.
- Added frozen-runtime pip options for source-only `openai-clip`; the complete
  direct source-archive fallback is included in 0.4.2.
- Added regression coverage for the torch-safe learned-IQA sidecar install
  options.

## [0.4.0] - 2026-09-14

### Changed

- Enforced Q-ReAlign Mini's immutable checkpoint revision during loading and in
  saved score versions; invalidated preparation records made by the unpinned
  loader. The release test workflow now installs its required Chromium browser.
- Made state-reset browser tests use the shared model-options fixture so their
  setup does not depend on the host's optional AI installation.
- Fixed frontend operation-result reconciliation so completed disk-delete
  operations refresh the review queue and clamp pagination after the last page
  is removed.
- Split the frontend workflow domains into injected operation-result/retry,
  export-dialog, library-operation, analysis, and browser modules. The stable
  `ShotSieveWorkflows` API and script load order remain compatible while result
  fields, retry behavior, operation polling, and user-facing messages are
  preserved.
- Consolidated the repeated desktop and sidecar runtime-support delegates behind
  one shared facade while retaining their private module aliases and dynamic
  monkeypatch/test seams.
- Narrowed Python HTTP route boundaries with explicit family dependency views and
  aggregator-supplied callbacks. Handler context/dependency assembly is separate
  from the request class, and route families no longer resolve the aggregator via
  `sys.modules`; existing route payloads, factories, and monkeypatch seams remain
  compatible.
- Extracted multi-root scan job execution into a dedicated runner with an
  immutable request snapshot, explicit per-root attempts, and centralized
  result/diagnostic finalization while preserving pagination, progress,
  cancellation, transaction, and route payload contracts.
- Split selected-model preparation into explicit context, storage, backend,
  generated-image validation, and durable-record phases while preserving the
  public preparation contract, state transitions, atomic writes, diagnostics,
  cancellation behavior, and backend cleanup.
- Staged export, delete, and root-scoped missing-entry cleanup into explicit
  per-operation filesystem, catalog, compensation, and cleanup phases. Private
  row-state objects retain observed-missing, deleted, not-processed, and
  catalog-uncertain progress while preserving the existing result and retry
  contracts.
- Shared bulk delete/export selection parsing, revision validation, consistent
  snapshotting, and frozen-ID materialization while keeping their filesystem and
  result contracts separate. Centralized operation-job lock, progress,
  cancellation, result-retention, failure, and lock-release handling across
  delete, export, cache clear, AI-support installation, and model preparation.
- Shared the frontend scan/score tracked-job lifecycle for start, tracking,
  polling, cleanup, abort, and recovery handling while preserving their
  workflow-specific estimates, payloads, progress phases, and result messages.
- Kept Review list, count, selection-revision, and bulk-selection requests on
  one normalized filter contract, including edge-size filters, so visible
  totals and guarded bulk operations target the same rows.
- Made the frontend workflow facade composition-only: library, export, and
  compare behavior now stays in its owning module while `ShotSieveWorkflows`
  preserves the existing public method names and cross-module operation wiring.
- Refactored scanner discovery, batch flushing, executor setup, and scan-run
  finalization into focused helpers while preserving progress, cancellation,
  preview/cache, partial-commit, diagnostic, and accounting behavior across
  inline and pooled processing paths.
- Centralized HTTP response header/body writing and client-disconnect handling
  across static, JSON, download, and JSON-error responses while preserving each
  endpoint's status and cache/download header policy.
- Refactored single-model scoring and learned-model comparison into focused
  planning, preview-outcome, batch-execution, and result-aggregation helpers
  while preserving scoring APIs, progress phases, summaries, diagnostics,
  comparison payloads, backend release behavior, and injection seams.
- Expanded the opt-in performance baseline to 100,000 rows in one active library plus global-scope coverage, with early/middle/deep Review pages, score/path/date sorts, representative filters, named-machine metadata, and separate timings/query plans for list, count, and selection revision.
- Added one shared, versioned image conversion policy for previews and learned-IQA inputs: EXIF-aware RGBA normalization, white-matte transparency compositing, and high-bit grayscale preservation. Legacy previews and scores are invalidated by conversion version without changing catalog decisions.
- Ordinary scans now preserve cached catalog rows, scores, and review decisions that are outside the current scan coverage.
- Removed the unchecked global “missing files” cache-cleanup action and replaced it with a root-scoped preview and confirmation workflow.
- Added a root-scoped **Review Missing Entries** maintenance flow that previews every candidate and affected review decision before confirmation.
- Scan jobs now retain durable failed/incomplete diagnostics and report per-root outcomes for multi-root jobs; earlier successful roots remain committed when a later root fails.
- Copy, move, and delete operations now return shared per-file outcomes with stages, paths, OS errors, retry safety, partial/unprocessed counts, and separate cleanup warnings.
- Added retained Library operation results with bounded path/stage details, full JSON download, safe retry, and status checking for uncertain jobs; successful IDs are removed while failed/unprocessed IDs remain selected.
- Retry now preserves the original Review scope, refreshes its revision for every chunk, aggregates all file outcomes, and retains pending work when cancellation or a later failure stops the retry.
- Scan, score, comparison, preparation, and file-operation jobs now retain unresolved identity after status loss, block new mutations until recovery, and refresh the workspace after terminal **Check status** recovery.
- Rejected-file deletion now uses the tracked async operation flow, matching selected deletion and export/move behavior.
- Added a read-only, root-scoped **Download decisions CSV** fallback for approved, rejected, or both decisions across the full matching result set.
- Retired the unsupported legacy Windows GPU adapter and release target. Native
  Intel XPU and AMD ROCm tracks now have their own runtime selection and build
  paths.
- Added a source-only Intel XPU track with pinned PyTorch XPU wheels, Windows/Linux installation guidance, native tensor verification, and one-image smoke reports that capture runtime, score, cache, and peak-memory evidence. No packaged XPU target is advertised.
- Added a Linux-first, source-only AMD ROCm track with exact AMD ROCm 7.2.1 PyTorch wheels, HIP-aware runtime resolution, AMD/ROCm smoke evidence, narrow Windows guidance, and no packaged ROCm target.
- Added Q-ReAlign Mini (`qrealign-mini`) as the sole Q-ReAlign catalog entry, including Qwen3.5-VL model metadata, a four-image batch cap, revision-aware readiness fingerprints, sanitized diagnostics, and updated model/license notices. Fresh CUDA/CPU/XPU/ROCm evidence remains open.
- Recorded the W05 exact Windows CPU bundle gate: Python 3.13.14/Torch 2.14.0, frozen launcher and archive checks, isolated scan, and one TOPIQ score passed; Q-ReAlign Mini, remaining targets, accelerator checks, and release/human gates remain open.
- Documented and regression-tested the retained preview/configuration/request compatibility seams, and split the large async-route and scan/schema test files by behavior without changing their coverage.
- Removed the obsolete predecessor integration from the selectable learned-IQA catalog; stored scores are not migrated or relabeled.
- Removed the process-global Torch loader workaround. Runtime discovery now reports empty/unavailable model state when the supported models are not ready, and explicit unavailable accelerators fail with actionable guidance while Auto reports CPU fallback reasons.
- Added selected-model preparation with runtime-specific validation, atomic readiness records, cache-path/version invalidation, sanitized failure diagnostics, and retained cancellation state.
- Made optional AI runtime acquisition explicit: ordinary noninteractive launches no longer install or repair learned-IQA/CUDA sidecars automatically; Settings now provides one job-backed Install / Repair AI support action with target/cache paths, diagnostics, retry, and restart guidance. The existing `SHOTSIEVE_BOOTSTRAP_AUTO_INSTALL_*` environment variables remain available for deliberate startup automation.
- Reused sanitized model diagnostics across preparation, scoring, and comparison jobs, including model/runtime/cache-volume context, offline and recovery classification, orphaned-preparation recovery, and retained API reports when readiness persistence fails.
- Added the optional `--model-cache-dir` startup setting; explicit Hugging Face/Torch cache environment settings remain authoritative.
- Aligned model smoke, release-target, and sidecar installs on the tested learned-IQA dependency stack, with release-time `pip check` validation.
- Model smoke now records resolved dependency versions and retains sanitized JSON failure reports without uploading caches, weights, or generated images.
- CI now runs on direct `main` pushes, adds a Python 3.14 core suite, smoke-tests an installed wheel outside the checkout, and fails browser coverage when Chromium cannot launch in CI while preserving local skips.
- Pinned the optional learned-IQA integration to `pyiqa==0.1.16` and documented separate package, checkpoint, cache, and Q-ReAlign/Qwen3.5-VL base-model licensing boundaries.
- Added offline pull-request CI and a separate weekly/manual CPU smoke workflow for TOPIQ and CLIPIQA; accelerator-specific Q-ReAlign Mini smoke remains a target-host release gate and never publishes weights or photos.
- Added a fixed 40-million-pixel decode budget for exceptional preview and direct learned-IQA source fallbacks, with per-file resource-limit diagnostics and file-attributed header warnings. Ready previews and usable RAW embedded thumbnails remain preferred, and corrupt RAW thumbnails can still fall back to bounded demosaicing.

### Fixed

- Runtime bootstrap archives now require a valid SHA-256 digest before download reuse, extraction, or launch; stale unchecked cache markers and digest mismatches are rejected, and releases publish a manifest generated from the exact archive files.
- Media preview and source responses now require private revalidation so regenerated files cannot remain stale under the same catalog-ID URL; byte-range delivery is unchanged.
- File discovery now reports unavailable roots and child-directory enumeration errors with the affected path instead of silently returning an empty scan.
- Added regression coverage for empty/unavailable scan behavior and rejected the removed cleanup scope at both cache-clear entry points.
- Missing-entry apply rechecks root accessibility, source absence, scope, review state, and the preview token; stale or unavailable previews return without mutating the catalog.
- Managed previews are removed only for the explicitly confirmed catalog rows, while original files remain untouched.
- Cancellation preserves the existing partial-work behavior while recording truthful failed scan diagnostics and processed counts.
- Completed file mutations remain committed across cancellation or later failures, while operation jobs retain failed/cancelled summaries through their existing status and result endpoints.
- Transfer and delete failures now distinguish missing files from unknown filesystem state, retain both paths and observation errors after uncertain mutations, reconcile catalog failures before move compensation, and preserve every unprocessed ID in the failing batch.
- The folder picker can still choose the explicitly typed local or UNC path when directory listing is unavailable; the later scan reports any access failure with its path.

## [0.3.2] - 2026-07-26

### Fixed

- Fixed Github builds failing due to build test changes

## [0.3.1] - 2026-07-26

### Fixed

- Fixed Github builds failing due to build test changes

## [0.3.0] - 2026-07-25

### Added

- Added support for configuring multiple non-overlapping directory paths as sources in the active library workspace.
- Added a custom directory ignore rules editor supporting exact folder names, relative globs, and wildcards.
- Added an asynchronous, cancellable library preflight check executing lightweight asset counting, source size estimation, and directory/file access permission audits.
- Added metadata-led filters for format groups (JPEG, PNG, TIFF, HEIF, RAW, Other), decimal Megapixel limits, file sizes (in MB), and metadata completeness (valid/unknown).
- Added new Review sorting orders: Resolution (ascending/descending), File Size (ascending/descending), Format Name, Width, and Height.
- Added a visual tag strip in the photo details toolbar displaying format, pixel dimensions, megapixels, computed aspect ratio, and formatted byte size.
- Added opt-in performance diagnostics for catalog overview, Review, and score-row query timing, plus a 60,000-row SQLite baseline and query-plan test that remain disabled during normal test runs.
- Added developer guidance for separating synthetic catalog-query measurements from local real-photo scan, preview, and learned-IQA measurements.

### Changed

- Improved large-catalog Review responsiveness by adding score-order indexes for the default AI-score and score-descending sorts. The Review queue remains paged, and single-photo keep/reject/reset actions now update the visible page without reloading the queue.
- Optimised file scanning traversal by pruning ignored directory names directly from `os.walk` list before descending.
- Made the selected Library folder the persisted default Review scope while retaining one shared catalog for preview and score reuse.
- Displayed separate **This library** and **All cached libraries** totals for discovered, scored, rejected, and selected photos, with a clear global Review scope indicator.
- Scoped rejected-file move and deletion shortcuts to the active library and named that library in destructive deletion confirmations.
- Refactored and modularized all active Python modules, JavaScript components, CSS stylesheets

### Fixed

- Fixed review-state and review-browser bulk delete/move requests so they require an explicit active-library root and cannot widen to the full catalog when the root is missing.
- Fixed direct file_ids delete and export/move operations by requiring page-level selection revision tokens to guard against stale selections after navigation or catalog updates.
- Fixed the Review rejected-actions bar so it stays consistent after repeated reject, move, and delete cycles.
- Fixed RAW image metadata extraction to record full RAW sensor dimensions and megapixels instead of embedded JPEG thumbnail dimensions.
- Resolved all linter warnings, missing type hints (`Sequence`, `Iterable`), and test mock signatures across `src/` and `tests/`.

## [0.2.3] - 2026-05-01

### Added

- Fixed exclude_file_ids must not be empty error on review pane.

## [0.2.2] - 2026-04-26

### Fixed

- Disabled RAW auto-brightening in the full demosaic fallback so monochrome and high-key RAW previews no longer render overexposed and drag learned-IQA scores down.
- Fixed 16-bit grayscale TIFF preview generation so scanned black-and-white images are rescaled to 8-bit tones before JPEG preview export instead of clipping nearly everything to white.
- Reset the review browser back to page 1 after a fresh Analyze run so returning to Review always starts from the beginning of the new result set.
- Clamped review pagination after delete and move operations so counts, page position, and current selection stay in sync when the result set shrinks.
- Switched delete, export/move, and cache-clear actions to async operation jobs with progress reporting and cancellation support in the busy overlay.
- Shortened RAW preview quality option labels so the Auto description fits cleanly inside the selector on tighter layouts.

## [0.2.0] - 2026-04-25

### Added

- Initial release of ShotSieve.
- Local-first desktop workflow for scanning photo folders, scoring images, and reviewing keep/reject decisions on your own machine.
- Runtime-pack and source-install documentation for Windows, Linux, and macOS workflows.
