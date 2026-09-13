# Building and developing ShotSieve

This guide is for contributors, local builders, and anyone installing ShotSieve from source.

If you just want to understand what ShotSieve is or which package to download, start with the top-level [README.md](../README.md).

## Python and environment notes

- `requires-python` is `>=3.11`
- For the DirectML extra, prefer Python `3.11` or `3.12`
- Apple Silicon uses the standard `learned-iqa` install and resolves to `mps` automatically when available

## Source install

Basic editable install:

```bash
python -m pip install -e .
```

Optional extras:

- Test dependencies: `python -m pip install -e .[test]`
- Lint dependencies: `python -m pip install -e .[lint]`
- Format loaders for HEIF and RAW workflows: `python -m pip install -e .[format-loaders]`
- Learned IQA support: `python -m pip install -e .[learned-iqa]` (`pyiqa==0.1.16`)
- DirectML learned IQA support on Python 3.11–3.12: `python -m pip install -e .[learned-iqa-directml]` (the supported target pins `torch==2.4.1`, `torchvision==0.19.1`, and `torch-directml==0.2.5.dev240914` together)
- Windows build tooling: `python -m pip install -e .[windows-build]`

## Desktop entry point

ShotSieve is desktop-first. For source installs and editable installs, the main entry point is:

```bash
shotsieve-desktop
```

Downloaded runtime packs use target-specific launcher names instead:

- Windows CPU: `ShotSieve-CPU.exe`
- Windows NVIDIA / CUDA: `ShotSieve-NVIDIA.exe`
- Windows DirectML: `ShotSieve-DML.exe`
- Linux CPU: `ShotSieve-CPU`
- Linux NVIDIA / CUDA: `ShotSieve-NVIDIA`
- macOS CPU: `ShotSieve-CPU`
- macOS Apple Silicon / MPS: `ShotSieve-MPS`

Intel XPU remains a source-only runtime path today; there is no prebuilt XPU runtime-pack target yet.

Useful flags:

```bash
shotsieve-desktop --data-dir ./shot-data
shotsieve-desktop --model-cache-dir ./model-cache
shotsieve-desktop --host 127.0.0.1 --port 9001 --no-browser
```

Startup/runtime notes:

- Frozen portable builds use a writable `data/` folder next to the executable
- Editable source checkouts (`pip install -e .` from this repository) also default to `<repo>/data`
- Installed packages outside a source checkout fall back to `%LOCALAPPDATA%\ShotSieve` on Windows, `%APPDATA%\ShotSieve` as a Windows fallback, or `~/.shotsieve` when platform app-data variables are unavailable
- On any install style, `--data-dir` overrides the default location

### Runtime archive integrity

The bootstrap path accepts only runtime manifests whose acquired archives have a valid, 64-character SHA-256 digest. The digest is checked against downloaded archives and local fallback archives before extraction. Cached installations are reusable only when their `.asset-sha256` marker matches the current manifest digest; an old empty marker, a missing marker, or a malformed marker requires verified reinstallation. A missing, malformed, or mismatched digest stops acquisition and launch with guidance to use a release manifest generated for the same release or a manually verified package.

Release manifests are generated from the exact built archives with `scripts/generate_bootstrap_manifest.py`. The generated `bootstrap-manifest.json` records the release tag, archive URLs, and per-archive SHA-256 values, and is published with the runtime-pack archives. A checksum detects transfer or storage mismatch against the publisher's value; it does not establish publisher identity or replace signing, SBOMs, or broader artifact provenance.

This check covers archives acquired by the bootstrap path. An explicitly supplied or colocated executable in a frozen bundle is a separate trust boundary and is accepted as supplied; the bootstrap does not hash that executable. Verify the containing bundle through its distribution channel when using that path.

Optional AI runtime packages are not downloaded during an ordinary noninteractive `shotsieve-desktop` launch. ShotSieve still opens the catalog and Review UI when learned-IQA support is missing or broken. In Settings, **Install / Repair AI support** runs the existing sidecar installer as an explicit, cancellable best-effort job; it reports the selected target, runtime sidecar path, model cache paths, failure details, and restart guidance. This runtime installation is separate from **Prepare selected model**, which may download model weights and performs the existing first-use CPU validation.

Startup automation remains available when explicitly configured:

- `SHOTSIEVE_BOOTSTRAP_AUTO_INSTALL_TORCH=1` to auto-install CUDA-sidecar PyTorch without prompting
- `SHOTSIEVE_BOOTSTRAP_AUTO_INSTALL_TORCH=0` to skip the CUDA-sidecar install step
- `SHOTSIEVE_BOOTSTRAP_AUTO_INSTALL_LEARNED_IQA=1` to auto-install learned-IQA dependencies without prompting
- `SHOTSIEVE_BOOTSTRAP_AUTO_INSTALL_LEARNED_IQA=0` to skip the learned-IQA install step

Interactive console launches may still ask for consent when neither automation setting is present. Portable and frozen builds prefer the bundled pip-based installer paths from `shotsieve.bootstrap`; if that installer is unavailable or declined, ShotSieve keeps running but learned backends may stay disabled. Cancelling an explicit installation takes effect between runtime install steps; restart ShotSieve if newly installed native packages are not usable in the current process.

The supported learned-model catalog is `topiq_nr` and `clipiqa`, with TOPIQ as the default. PyIQA discovery is not an allowlist: if the two product models are not discoverable or cannot initialize, Settings shows an unavailable/empty model state. Older stored model names remain displayable by their raw name, but retired names cannot be selected for new scoring or comparison runs. No process-wide `torch.load` override is used.

The Settings **Prepare selected model** action uses the ordinary backend on CPU for the selected model and runs a tiny generated-image inference check. It may download the model's upstream assets on first use; TOPIQ can require its ResNet backbone and CFANet checkpoint, while CLIPIQA can require the CLIP RN50 dependency. Preparation records coarse phases, effective cache paths and volume free space, dependency versions/fingerprint, tested CPU runtime, and sanitized error/recovery details in a small atomic JSON file under the app data directory. A `preparing` record owned by a process that has ended is downgraded to an interrupted failure at startup. A later `/api/options` call performs only local record, dependency-metadata, and disk-volume reads; it does not scan caches, download assets, or construct models. Missing or changed cache context invalidates a previous `prepared` state, and scoring still validates its selected runtime at use time: readiness is a last successful check, not a future availability guarantee.

Score and Compare job failures reuse the same diagnostic schema as Prepare. Their status/result payloads retain sanitized exception chains, category and recovery action, model name(s), requested and actual runtime when known, offline flags, effective cache paths, and the free-space view for each cache volume. Known offline-cache, permission, no-space, dependency, runtime, network, and corruption failures receive targeted guidance; unknown or execution failures retain their sanitized causes. The recovery guidance points to **Prepare selected model** before retrying, and URL credentials, Hub tokens, and proxy secrets are redacted before the diagnostic reaches the API or UI.

`--model-cache-dir ROOT` sets default `HF_HOME=ROOT/huggingface`, `HF_HUB_CACHE=ROOT/huggingface/hub`, and `TORCH_HOME=ROOT/torch` before learned-IQA runtime preparation. Explicit environment values win, including Hub endpoints, proxy/certificate, and offline settings, so the resulting cache paths may be split. For an offline portable setup, prepare the selected model into the intended compatible cache tree, shut down the app, copy that tree with the app-data readiness record, and validate it in a fresh offline process. ShotSieve does not migrate or remove caches automatically.

The release and sidecar paths use the tested learned-IQA package set `pyiqa==0.1.16`, `timm==1.0.28`, `huggingface-hub==1.24.0`, `transformers==5.14.1`, and `openai-clip==1.0.1`. CPU, CUDA, and Apple MPS targets use `torch==2.13.0` with `torchvision==0.28.0`; the CUDA path selects the cu126 index, while CPU selects the PyTorch CPU index. The corresponding target constraint file is `scripts/release-constraints-torch.txt`.

For a Windows DirectML sidecar, the embedded installer resolves the pinned Torch/Torchvision/DirectML trio in one install step. Release builds use `scripts/release-constraints-windows-dml.txt` in addition to the common build constraints; the DirectML target remains on Python 3.11-3.12 and does not inherit the non-DirectML Torch pair. Local Windows release builds and CI run `pip check` after resolving the target environment.

Pull requests run the offline test workflow in `.github/workflows/ci.yml`; it sets the learned-model offline flags so an accidental model download fails rather than silently reaching the Hub. The separate `.github/workflows/model-smoke.yml` workflow is manual/weekly and prepares TOPIQ and CLIPIQA in fresh isolated caches, then repeats the CPU check in a new process with socket access disabled. It does not upload caches or generated images.

Each model-smoke invocation records resolved model dependency versions. On failure it writes a sanitized JSON diagnostic containing only model/runtime/cache context and redacted causes; the workflow uploads those JSON reports for troubleshooting and never uploads model caches, weights, or generated images.

## Scan and catalog safety

Ordinary scans only update files discovered during that run. They do not remove cached rows, scores, or review decisions for files outside the selected recursion, extension, or ignore-rule coverage. Root and child-directory enumeration errors are reported with their path and operating-system detail rather than being treated as an empty scan.

Verified missing-entry cleanup is intentionally separate from scanning. The Settings **Review Missing Entries** action previews all missing candidates below each selected root, shows affected review decisions, and requires explicit confirmation before applying. A root that cannot be fully enumerated is reported as unknown, never as an empty or missing-file result; a stale preview must be refreshed before it can apply. Do not use the removed unchecked “missing” cache action in scripts or clients.

Scan failures and cancellations are recorded as failed scan diagnostics even when the catalog transaction rolls back, including the root, timestamps, error text, and counts processed before the failure. Each root in a multi-root job has its own transaction: earlier successful roots remain committed, while failed and not-processed roots are reported in the scan job status. Existing cancellation behavior remains best-effort and retains work completed before cancellation.

Copy, move, and delete workers expose one shared per-file result contract in their synchronous and asynchronous responses. Results retain source/destination paths, the operation stage, OS error details, guarded present/missing/unknown filesystem state, observation errors, retry safety, and partial/unprocessed outcomes; preview cleanup warnings do not inflate transfer failures. When a transfer or unlink may have mutated the filesystem, the result retains both paths and is never marked safe for automatic retry. Catalog failures are rolled back and read back before a move is compensated, and failed bulk jobs retain every untouched ID, including the remainder of the current batch. Completed file mutations are committed independently so cancellation or a later fatal row does not erase earlier work. Failed operation jobs retain their summary for the existing status and result endpoints, while partial or uncertain files are left for manual inspection rather than automatically retried.

The Library UI retains the latest operation result, keeps unsuccessful/unprocessed IDs selected, bounds the on-screen result list, and provides Copy details, Download JSON, safe retry, and Check status actions. Rejected deletion goes through the same tracked async operation path as selected deletion. Settings includes a read-only root-scoped decision CSV export for approved, rejected, or both decisions; it includes all matching rows across Review pages and uses spreadsheet-safe UTF-8 CSV escaping.

The folder browser accepts a full local path or UNC path such as `\\server\share\folder`; press Enter after editing the path to open it. The browser does not enumerate network servers or probe write/delete permissions against user photos.

### Media cache behavior

Preview and original-media endpoints use URLs keyed by catalog file ID, but either file may be regenerated or replaced in place. They therefore return `Cache-Control: private, no-cache`, requiring clients to revalidate rather than reuse stale bytes. Media streaming continues to support single byte ranges for compatible clients.

## Testing and verification

Run the automated suite with:

```bash
python -m pytest -q
```

Validate the dead-import guard with:

```bash
python -m pip install -e .[lint]
python -m ruff check --select F401 src/shotsieve
```

The frontend smoke tests under `tests/test_frontend_accessibility.py` use Playwright for visual usability, layout, and interaction checks. After installing `.[test]`, install Chromium once per environment with:

```bash
python -m playwright install chromium
```

For a quick manual visual QA pass, use [accessibility-checklist.md](./accessibility-checklist.md).

CI runs the offline suite for pull requests and direct pushes to `main`. It also runs a Python 3.14 core suite without browser or accelerator claims, and builds a wheel into a clean virtual environment outside the checkout. Browser tests are marked separately: local environments may skip them when Chromium is unavailable, but CI treats a missing Playwright install or Chromium launch failure as a test failure.

## Performance measurement

Performance diagnostics are opt-in and do not run as part of the ordinary test suite. The synthetic 60,000-row SQLite baseline and the procedure for collecting local real-photo scan, preview, and learned-IQA measurements are documented in [performance-measurement.md](./performance-measurement.md).

Use that guide before changing indexes, pagination, or catalog storage behavior. It explains the required environment notes and separates database-query results from filesystem, preview, model-startup, and inference costs.

## Release builds and portable bundles

Current release automation builds **runtime packs**, not legacy bootstrap helper bundles.

The Windows local build entry point is:

```powershell
./scripts/build_windows_releases.ps1 -PlanOnly
./scripts/build_windows_releases.ps1
```

Useful examples:

```powershell
# JSON plan for the default Windows runtime targets
./scripts/build_windows_releases.ps1 -Mode runtime -PlanOnly -AsJson

# Build only the NVIDIA runtime pack
./scripts/build_windows_releases.ps1 -Mode runtime -TargetIds windows-nvidia
```

Current Windows runtime-pack outputs:

- `ShotSieve-windows-cpu`
- `ShotSieve-windows-nvidia`
- `ShotSieve-windows-dml`

Tier 1 runtime-pack targets are currently defined for:

- Windows CPU, NVIDIA CUDA, and DirectML
- Linux CPU and NVIDIA CUDA
- macOS CPU and Apple Silicon MPS

The target matrix lives in `src/shotsieve/release_targets.py` and is emitted by `scripts/release_target_matrix.py`.

After all runtime-pack build jobs finish, the release workflow runs `scripts/generate_bootstrap_manifest.py` over the downloaded archives and publishes `bootstrap-manifest.json` alongside them. For a local fixture or release rehearsal, use the same generator with `--archive-root`, `--output`, and `--release-tag`; it fails if any target archive is missing or ambiguous.

Prepare release references first, then publish the tag. The prep helper updates the package version files and ensures the changelog has an entry for the release:

```powershell
./scripts/prepare_release.ps1 -Version 0.2.0
```

That script updates `pyproject.toml`, `src/shotsieve/__init__.py`, `CHANGELOG.md`, and (when present) the checked-in `src/shotsieve.egg-info/PKG-INFO`. Review and commit those changes before publishing the tag.

To publish a GitHub release, use the separate tag helper to create and push an annotated version tag:

```powershell
./scripts/create_github_release.ps1 -Version v0.2.0
```

That tag helper performs the local git safety checks, creates the tag, and pushes it to `origin`, but it does **not** edit version files. The actual GitHub release is then published by `.github/workflows/release.yml` after the `v*` tag push reaches GitHub. A dry run is available with:

```powershell
./scripts/create_github_release.ps1 -Version v0.2.0 -DryRun
```

The current tag-push workflow does not support `-PreRelease`; if you need a pre-release, mark it manually in GitHub after the workflow publishes it or extend the release workflow to handle that metadata.
