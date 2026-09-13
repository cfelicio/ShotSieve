# Changelog

All notable changes to this project will be documented in this file.

## [0.3.2] - 2026-07-26

### Fixed

- Fixed Github builds failing due to build test changes

## [0.3.1] - 2026-07-26

### Fixed

- Fixed Github builds failing due to build test changes

## [Unreleased]

### Changed

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
- Restored Q-Align (`qalign`) to the supported learned-IQA product catalog for CUDA and Apple MPS, with accelerator-only preparation, one-image batching, and explicit CPU/DirectML exclusion; other PyIQA names remain disabled for new runs while historical rows stay readable.
- Removed the process-global Torch loader workaround. Runtime discovery now reports empty/unavailable model state when the supported models are not ready, and explicit unavailable accelerators fail with actionable guidance while Auto reports CPU fallback reasons.
- Pinned the Windows DirectML target to Python 3.11–3.12 with `torch==2.4.1`, `torchvision==0.19.1`, and `torch-directml==0.2.5.dev240914` in the target-specific release and sidecar install paths.
- Added selected-model preparation with runtime-specific validation, atomic readiness records, cache-path/version invalidation, sanitized failure diagnostics, and retained cancellation state.
- Made optional AI runtime acquisition explicit: ordinary noninteractive launches no longer install or repair learned-IQA/CUDA sidecars automatically; Settings now provides one job-backed Install / Repair AI support action with target/cache paths, diagnostics, retry, and restart guidance. The existing `SHOTSIEVE_BOOTSTRAP_AUTO_INSTALL_*` environment variables remain available for deliberate startup automation.
- Reused sanitized model diagnostics across preparation, scoring, and comparison jobs, including model/runtime/cache-volume context, offline and recovery classification, orphaned-preparation recovery, and retained API reports when readiness persistence fails.
- Added the optional `--model-cache-dir` startup setting; explicit Hugging Face/Torch cache environment settings remain authoritative.
- Aligned model smoke, release-target, and sidecar installs on the tested learned-IQA dependency stack, with separate non-DirectML Torch and Windows DirectML constraint files plus release-time `pip check` validation.
- Model smoke now records resolved dependency versions and retains sanitized JSON failure reports without uploading caches, weights, or generated images.
- CI now runs on direct `main` pushes, adds a Python 3.14 core suite, smoke-tests an installed wheel outside the checkout, and fails browser coverage when Chromium cannot launch in CI while preserving local skips.
- Pinned the optional learned-IQA integration to `pyiqa==0.1.16` and documented separate package, checkpoint, cache, and Q-Align/base-model licensing boundaries.
- Added offline pull-request CI and a separate weekly/manual CPU smoke workflow for TOPIQ and CLIPIQA; accelerator-specific Q-Align smoke remains a target-host release gate and never publishes weights or photos.
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
