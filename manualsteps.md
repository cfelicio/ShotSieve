# ShotSieve manual release steps

Updated 2026-09-13.

This is the remaining candidate, target, hardware, storage, and human-review checklist extracted from `implementation.md`. The code-side implementation work is complete for the current working tree. Do not mark a release candidate ready until the applicable items below have recorded evidence.

## Evidence completed automatically in this pass

- Fixed the narrow Settings layout overflow by allowing the flex column to shrink within the mobile grid track.
- Full offline suite after Q-Align restoration: **584 passed, 52 skipped, 1 warning** in 145.51 seconds.
- Restored Q-Align/runtime/scoring/web focused suite: **96 passed**.
- Focused responsive/accessibility suite: **35 passed**.
- Focused local file-operation/scanner regression suite: **45 passed, 1 warning**.
- Ruff `F,E9`, Python compilation, `git diff --check`, and the release/build tests passed.
- Fresh isolated online and new-process offline CPU smokes passed for TOPIQ and CLIPIQA. Q-Align is restored to the catalog and has offline-capable code paths, but no 7 GB Q-Align weight download was performed locally. Resolved versions for the CPU smokes were `pyiqa 0.1.16`, `timm 1.0.28`, `huggingface-hub 1.24.0`, `transformers 5.14.1`, `openai-clip 1.0.1`, `torch 2.13.0+cu126`, and `torchvision 0.28.0+cu126`; the local environment's `pip check` passed.
- A local Windows CPU bundle was built, its launcher started, and its bundled server scanned one generated JPEG and scored it with `topiq_nr` on CPU: `files_scored=1`, `learned_scored=1`, `files_failed=0`.
- The CPU archive contained 21,552 entries, one launcher, and zero `.safetensors`, `.ckpt`, `.pth`, or `.pt` files.

The local bundle rehearsal used Python 3.14 and a CUDA-enabled Torch install, not the exact Python 3.13 CPU release environment. It is useful evidence, but it does not close the release-target gates.

## R01 — release-candidate model validation

For the exact commit/tag intended for the test release:

1. Prepare fresh isolated caches for `topiq_nr` and `clipiqa` using the selected target constraints. On a compatible CUDA or Apple MPS host, also prepare `qalign` in its own cache; it is approximately 7 GB and uses batch size 1.
2. Run the online smoke, then a new network-disabled process using the complete cache. Capture the sanitized JSON report, exact resolved versions, cache paths, date, commit/tag, and `pip check` result.
3. Exercise a missing or corrupt offline asset for each selected model and confirm the diagnostic is actionable and does not expose credentials, model weights, or private image data.
4. Repeat for each target stack where the cache is claimed to be portable. Do not reuse the local rehearsal as evidence for DirectML or another platform.

## R02 — actual shipped bundles

Build and test every target selected for the test release, using the exact release workflow/environment:

`windows-cpu`, `windows-nvidia`, `windows-dml`, `linux-cpu`, `linux-nvidia`, `macos-cpu`, and `macos-mps`.

For each shipped target:

1. Confirm target Python/Torch constraints resolve, `pip check` passes, and the produced launcher responds to `--help`.
2. Start the extracted bundle with an isolated data directory and externally prepared model cache.
3. Scan and score one disposable JPEG. Confirm the requested and actual runtime in the result or diagnostic.
4. Inspect the staged bundle and archive for model weights, exact package versions, notices, and asset/license terms. Preserve the archive hash and release-manifest evidence.

## R03 — advertised accelerators

The current Windows host is not a passing CUDA host: Torch reports CUDA available, but the installed `2.13.0+cu126` build lacks kernels for the RTX 5060 Ti (`sm_120`), and a one-image CUDA score failed with `no kernel image is available`. DirectML is also untestable in this Python 3.14 environment and is not installed. Q-Align is therefore not locally accelerator-validated.

On supported hosts, run TOPIQ and CLIPIQA one-image checks for every advertised accelerator and Q-Align one-image checks on CUDA/MPS hosts. Record requested runtime, actual runtime, fallback reason, batch size, cache size, and score outcome. Q-Align remains intentionally excluded from DirectML pending a real Windows Python 3.12/Torch-DirectML validation. Required environments not available locally remain explicitly unverified: a compatible Windows CUDA host, Windows DirectML/Python 3.12, Linux CUDA, and macOS MPS.

## R04 — local and network file operations

Using disposable photos only, validate the release's claimed storage matrix:

- Windows same-volume and cross-volume moves.
- Denied and locked files.
- UNC and/or mapped shares.
- Copy-success/delete-failure and disconnect/reconnect during operations.
- Non-ASCII and long paths.
- Cancellation and unavailable-root scans, preserving catalog decisions.
- Linux/macOS local paths and any available mounted-share cases.

For each case, verify bytes on disk, catalog paths, visible result wording, retry eligibility, and decision CSV root/path accuracy. Record untested combinations; do not claim general NAS certification from the automated suite.

## Human visual and accessibility review

Run `docs/accessibility-checklist.md` on a fresh small library before the test release. Check Chromium at approximately 390px, 768px, desktop width, and 125% zoom across Library, Compare, Review, and Settings. Confirm no distracting horizontal scroll, readable contrast, usable dialogs, clear selection/current-photo states, lightbox behavior, and visible primary actions. Pair this with keyboard/focus and any available assistive-technology checks; automated browser tests do not replace this review.

## Test-release publication

1. Review the exact diff, version, changelog, third-party notices, generated release manifest, and archive hashes.
2. Commit the release changes and create the annotated version tag using the documented helpers in `docs/building.md`.
3. After the tag workflow publishes, mark the GitHub release as a pre-release manually if that is the intended test-release channel; the current workflow does not set `-PreRelease` automatically.
4. Attach the recorded R01–R04 evidence to the release notes or build documentation, including explicit untested targets.
