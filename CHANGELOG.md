# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

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