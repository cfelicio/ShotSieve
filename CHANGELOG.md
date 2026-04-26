# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Fixed

- Disabled RAW auto-brightening in the full demosaic fallback so monochrome and high-key RAW previews no longer render overexposed and drag learned-IQA scores down.
- Reset the review browser back to page 1 after a fresh Analyze run so returning to Review always starts from the beginning of the new result set.
- Clamped review pagination after delete and move operations so counts, page position, and current selection stay in sync when the result set shrinks.
- Switched delete, export/move, and cache-clear actions to async operation jobs with progress reporting and cancellation support in the busy overlay.
- Shortened RAW preview quality option labels so the Auto description fits cleanly inside the selector on tighter layouts.

## [0.2.0] - 2026-04-25

### Added

- Initial release of ShotSieve.
- Local-first desktop workflow for scanning photo folders, scoring images, and reviewing keep/reject decisions on your own machine.
- Runtime-pack and source-install documentation for Windows, Linux, and macOS workflows.