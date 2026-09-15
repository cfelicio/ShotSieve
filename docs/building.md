# Building and developing ShotSieve

This guide is for contributors, local builders, and anyone installing ShotSieve from source.

If you just want to understand what ShotSieve is or which package to download, start with the top-level [README.md](../README.md).

## Python and environment notes

- `requires-python` is `>=3.11`
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
- Windows build tooling: `python -m pip install -e .[windows-build]`

Intel XPU is a source-install-only track. The pinned Windows/Linux install,
driver prerequisites, runtime probe, and model-evidence commands are in
[intel-xpu.md](./intel-xpu.md). Do not use the CPU/CUDA release constraints
for an XPU environment; use `scripts/source-constraints-xpu.txt` and the
official PyTorch XPU wheel index.

AMD ROCm is also a source-install-only track, Linux first. The exact
ROCm/PyTorch wheels, AMD driver boundary, Windows limitation, runtime probe,
and model-evidence commands are in [amd-rocm.md](./amd-rocm.md). Do not use
the CPU/CUDA release constraints for ROCm; use the matching source constraints
and AMD's current hardware matrix. No ROCm pack is published.

## Desktop entry point

ShotSieve is desktop-first. For source installs and editable installs, the main entry point is:

```bash
shotsieve-desktop
```

Downloaded runtime packs use target-specific launcher names instead:

- Windows CPU: `ShotSieve-CPU.exe`
- Windows NVIDIA / CUDA: `ShotSieve-NVIDIA.exe`
- Linux CPU: `ShotSieve-CPU`
- Linux NVIDIA / CUDA: `ShotSieve-NVIDIA`
- macOS CPU: `ShotSieve-CPU`
- macOS Apple Silicon / MPS: `ShotSieve-MPS`

Intel XPU remains a source-only runtime path today; there is no prebuilt XPU runtime-pack target yet. AMD ROCm is also source-only, with no prebuilt ROCm
runtime-pack target. See [intel-xpu.md](./intel-xpu.md) and
[amd-rocm.md](./amd-rocm.md) for the exact source-install tracks.

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

Optional AI runtime packages are not downloaded during an ordinary noninteractive `shotsieve-desktop` launch. ShotSieve still opens the catalog and Review UI when learned-IQA support is missing or broken. In Settings, **Install / Repair AI support** runs the existing sidecar installer as an explicit, cancellable best-effort job; it reports the selected target, runtime sidecar path, model cache paths, failure details, and restart guidance. This runtime installation is separate from **Prepare selected model**, which may download model weights and performs the existing first-use validation on CPU-compatible models or the selected accelerator for Q-ReAlign Mini.

Startup automation remains available when explicitly configured:

- `SHOTSIEVE_BOOTSTRAP_AUTO_INSTALL_TORCH=1` to auto-install CUDA-sidecar PyTorch without prompting
- `SHOTSIEVE_BOOTSTRAP_AUTO_INSTALL_TORCH=0` to skip the CUDA-sidecar install step
- `SHOTSIEVE_BOOTSTRAP_AUTO_INSTALL_LEARNED_IQA=1` to auto-install learned-IQA dependencies without prompting
- `SHOTSIEVE_BOOTSTRAP_AUTO_INSTALL_LEARNED_IQA=0` to skip the learned-IQA install step

Interactive console launches may still ask for consent when neither automation setting is present. Portable and frozen builds prefer the bundled pip-based installer paths from `shotsieve.bootstrap`; if that installer is unavailable or declined, ShotSieve keeps running but learned backends may stay disabled. Cancelling an explicit installation takes effect between runtime install steps; restart ShotSieve if newly installed native packages are not usable in the current process.

The supported learned-model catalog is `topiq_nr`, `clipiqa`, and `qrealign-mini`, with TOPIQ as the default. Q-ReAlign Mini is the only initial Q-ReAlign size; its published checkpoint is about 2.2 GB and the authors claim CPU support and under 4 GB GPU memory. Runtime availability and release support remain model- and target-specific, so Settings only exposes models discovered and compatible in the selected runtime. PyIQA discovery is not an allowlist: if a product model is not discoverable or cannot initialize, Settings shows an unavailable/empty model state. No process-wide `torch.load` override is used.

The Settings **Prepare selected model** action uses CPU for CPU-compatible models and the selected/Auto runtime for Q-ReAlign Mini, then runs a tiny generated-image inference check. It may download the model's upstream assets on first use; TOPIQ can require its ResNet backbone and CFANet checkpoint, CLIPIQA can require the CLIP RN50 dependency, and Q-ReAlign Mini downloads the `q-future/Q-ReAlign-Mini-0.8B` processor/configuration and approximately 2.2 GB of safetensors. Q-ReAlign Mini uses a maximum batch size of four in ShotSieve. Preparation records coarse phases, effective cache paths and volume free space, upstream checkpoint revision, dependency versions/fingerprint, requested and tested runtime, and sanitized error/recovery details in a small atomic JSON file under the app data directory. A `preparing` record owned by a process that has ended is downgraded to an interrupted failure at startup. A later `/api/options` call performs only local record, dependency-metadata, and disk-volume reads; it does not scan caches, download assets, or construct models. Missing or changed cache context invalidates a previous `prepared` state, and scoring still validates its selected runtime at use time: readiness is a last successful check, not a future availability guarantee.

The preparation implementation keeps those transitions explicit. `model_assets` resolves an immutable attempt context first, then routes all durable records and progress through one private record store. Storage checks, backend construction/runtime compatibility, generated-image validation, success completion, and failure/cancellation persistence are separate helpers. The public preparation signature and record schema remain the compatibility boundary; atomic writes, the `preparing`/`prepared`/failure state sequence, sanitized diagnostics, and backend release ownership stay unchanged.

Score and Compare job failures reuse the same diagnostic schema as Prepare. Their status/result payloads retain sanitized exception chains, category and recovery action, model name(s), requested and actual runtime when known, offline flags, effective cache paths, and the free-space view for each cache volume. Known offline-cache, permission, no-space, dependency, runtime, network, and corruption failures receive targeted guidance; unknown or execution failures retain their sanitized causes. The recovery guidance points to **Prepare selected model** before retrying, and URL credentials, Hub tokens, and proxy secrets are redacted before the diagnostic reaches the API or UI.

`--model-cache-dir ROOT` sets default `HF_HOME=ROOT/huggingface`, `HF_HUB_CACHE=ROOT/huggingface/hub`, and `TORCH_HOME=ROOT/torch` before learned-IQA runtime preparation. Explicit environment values win, including Hub endpoints, proxy/certificate, and offline settings, so the resulting cache paths may be split. For an offline portable setup, prepare the selected model into the intended compatible cache tree, shut down the app, copy that tree with the app-data readiness record, and validate it in a fresh offline process. ShotSieve does not migrate or remove caches automatically.

The release and sidecar paths use the tested learned-IQA package set `pyiqa==0.1.16`, `timm==1.0.29`, `huggingface-hub==1.31.0`, `transformers==5.17.0`, and `openai-clip==1.0.1`, `accelerate==1.15.0`, `sentencepiece==0.2.2`, and `einops==0.8.2`. CPU, CUDA, and Apple MPS targets use `torch==2.14.0` with `torchvision==0.29.0`; the CUDA path selects the cu130 index, while CPU selects the PyTorch CPU index. The corresponding target constraint file is `scripts/release-constraints-torch.txt`. The source-only ROCm track uses AMD's separately validated ROCm 7.2.1 PyTorch wheels and `scripts/source-constraints-rocm.txt` (or the explicit Windows variant); it is not a packaged target.

Windows AMD hardware remains on CPU unless the exact host is in AMD's supported ROCm/PyTorch matrix and passes the separate source-install validation. The retired DirectML package, sidecar path, and Windows-DML release target are absent from new installs and release builds. Local Windows release builds and CI run `pip check` after resolving the target environment.

Pull requests run the offline test workflow in `.github/workflows/ci.yml`; it sets the learned-model offline flags so an accidental model download fails rather than silently reaching the Hub. The separate `.github/workflows/model-smoke.yml` workflow is manual/weekly and prepares TOPIQ and CLIPIQA in fresh isolated caches, then repeats the CPU check in a new process with socket access disabled. Q-ReAlign Mini requires fresh target-specific cache and model smoke evidence; its online/offline checks are described in `manualsteps.md` and are not replaced by the CPU-only GitHub-hosted job. The workflow does not upload caches or generated images.

Each model-smoke invocation records resolved model dependency versions. On failure it writes a sanitized JSON diagnostic containing only model/runtime/cache context and redacted causes; the workflow uploads those JSON reports for troubleshooting and never uploads model caches, weights, or generated images.

## Scan and catalog safety

Review list, count, selection-revision, and bulk-selection queries use the same normalized filter set. This includes path, score, format, metadata, size, resolution, and edge-size filters, so the displayed page, total, revision guard, and materialized bulk-operation IDs cannot silently target different rows.

Ordinary scans only update files discovered during that run. They do not remove cached rows, scores, or review decisions for files outside the selected recursion, extension, or ignore-rule coverage. Root and child-directory enumeration errors are reported with their path and operating-system detail rather than being treated as an empty scan.

Verified missing-entry cleanup is intentionally separate from scanning. The Settings **Review Missing Entries** action previews all missing candidates below each selected root, shows affected review decisions, and requires explicit confirmation before applying. A root that cannot be fully enumerated is reported as unknown, never as an empty or missing-file result; a stale preview must be refreshed before it can apply. Do not use the removed unchecked “missing” cache action in scripts or clients.

Scan failures and cancellations are recorded as failed scan diagnostics even when the catalog transaction rolls back, including the root, timestamps, error text, and counts processed before the failure. Each root in a multi-root job has its own transaction: earlier successful roots remain committed, while failed and not-processed roots are reported in the scan job status. Existing cancellation behavior remains best-effort and retains work completed before cancellation.

The asynchronous scan route keeps the HTTP adapter small and hands a frozen request snapshot to the scan runner. Root execution, job-level progress translation, pagination across roots, and final result/diagnostic construction are separate stages; the scanner dependency, per-root transaction boundary, cancellation callback, and status/result payloads remain the same.

Copy, move, and delete workers expose one shared per-file result contract in their synchronous and asynchronous responses. Results retain source/destination paths, the operation stage, OS error details, guarded present/missing/unknown filesystem state, observation errors, retry safety, and partial/unprocessed outcomes; preview cleanup warnings do not inflate transfer failures. When a transfer or unlink may have mutated the filesystem, the result retains both paths and is never marked safe for automatic retry. Catalog failures are rolled back and read back before a move is compensated, and failed bulk jobs retain every untouched ID, including the remainder of the current batch. Completed file mutations are committed independently so cancellation or a later fatal row does not erase earlier work. Failed operation jobs retain their summary for the existing status and result endpoints, while partial or uncertain files are left for manual inspection rather than automatically retried.

The file-operation implementation is staged by operation: export separates source/target validation, copy or move transfer, catalog update, compensation, and preview cleanup; delete separates trusted-root policy, one-file removal, catalog reconciliation, and preview cleanup; missing-entry maintenance separates root inspection, preview revalidation, and catalog deletion. Private row-outcome and reconciliation state objects retain observed-missing, deleted, not-processed, and catalog-uncertain distinctions while the public summary and exception payloads remain unchanged.

The Library UI retains the latest operation result, keeps unsuccessful/unprocessed IDs selected, bounds the on-screen result list, and provides Copy details, Download JSON, safe retry, and Check status actions. Rejected deletion goes through the same tracked async operation path as selected deletion. Settings includes a read-only root-scoped decision CSV export for approved, rejected, or both decisions; it includes all matching rows across Review pages and uses spreadsheet-safe UTF-8 CSV escaping.

Delete and export use the same bulk-selection boundary: the request is normalized and
root-checked before the database opens, the selection revision is validated inside a
consistent snapshot, and the filtered IDs are materialized before either operation
mutates the catalog or filesystem. Their operation-specific mutation, compensation,
and result contracts remain separate. Delete, export, cache-clear, optional AI-support
installation, and model-preparation jobs share the operation-job launcher for lock
ownership, registry creation, progress publication, cancellation checks, result
retention, failure conversion, and guaranteed lock release; their worker payloads and
diagnostic schemas remain operation-specific.

HTTP route construction keeps the legacy `WebRouteDependencies` injection object for
compatibility, but each route family consumes a read-only dependency view for its own
services. `web.py` now builds that context separately from the request handler class.
The route aggregator supplies explicit response and cross-family callbacks to the
family modules, so common, scan, file, job, review, and media routes do not resolve
`shotsieve.web_routes` through `sys.modules`. Existing aggregator exports and
monkeypatch seams remain available, while import order no longer determines whether a
route family can find a helper.

The folder browser accepts a full local path or UNC path such as `\\server\share\folder`; press Enter after editing the path to open it. The browser does not enumerate network servers or probe write/delete permissions against user photos.

### Media cache behavior

Preview and original-media endpoints use URLs keyed by catalog file ID, but either file may be regenerated or replaced in place. They therefore return `Cache-Control: private, no-cache`, requiring clients to revalidate rather than reuse stale bytes. Media streaming continues to support single byte ranges for compatible clients.

Preview generation and learned-IQA preprocessing share the versioned image conversion helper. Both apply EXIF orientation, normalize palette/alpha data through RGBA, composite transparency onto a white matte, and preserve high-bit grayscale midtones. The preview and score cache records include the conversion version; legacy or changed-version records are regenerated or rescored automatically without clearing review decisions.

Exceptional fallback inputs are checked from their header dimensions before conversion and are refused above the fixed 40-million-pixel decode budget. This protects preview and direct learned-IQA source paths from full-image allocation that a later thumbnail resize cannot undo. A ready bounded preview is still preferred, RAW embedded thumbnails are tried before full demosaicing, and corrupt thumbnails fall back to demosaicing when the sensor dimensions are within budget. Header warnings are captured per file for diagnostics; concurrent worker stderr is not globally redirected.

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

Performance diagnostics are opt-in and do not run as part of the ordinary test suite. The synthetic 100,000-active-row SQLite baseline now measures global and active deep Review navigation, while the local utility records the same list/count/revision timings and query plans alongside scan, preview, and learned-IQA measurements. The procedure is documented in [performance-measurement.md](./performance-measurement.md).

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

Tier 1 runtime-pack targets are currently defined for:

- Windows CPU and NVIDIA CUDA
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

### Model-stack validation follow-up (2026-09-13)

The selected stack was exercised locally on CPU and CUDA 13.0 (RTX 5060 Ti).
The former DirectML experiment is retained only as historical retirement
evidence; it is not a supported stack or release target. Exact shipped bundles
and MPS remain release validation gates. Existing source environments must be
upgraded explicitly; changing release constraints does not modify an existing
virtual environment.

Q-ReAlign Mini model validation is tracked in `manualsteps.md`; no model-specific
hardware support is claimed until its fresh online/offline smoke evidence passes.
