# Building and developing ShotSieve

This guide is for contributors, local builders, and anyone installing
ShotSieve from source. For the user-facing overview, start with the
[README](../README.md).

## Python and source install

The package requires Python 3.11 or newer. A basic editable install is:

```bash
python -m pip install -e .
```

Optional extras:

```bash
python -m pip install -e ".[test]"           # pytest, Playwright, Ruff
python -m pip install -e ".[lint]"           # Ruff only
python -m pip install -e ".[format-loaders]" # HEIF and RAW loaders
python -m pip install -e ".[learned-iqa]"   # pyiqa==0.1.16
python -m pip install -e ".[windows-build]" # PyInstaller and build tooling
```

Keep accelerator environments separate. Intel XPU is a packaged Windows/Linux track
as well as a source-install track; it uses
`scripts/source-constraints-xpu.txt` and the [Intel XPU guide](intel-xpu.md).
AMD ROCm is a packaged Windows/Linux track as well as a source-install track;
it uses the platform-specific source constraints and the [AMD ROCm guide](amd-rocm.md).
Do not mix those wheels with CPU, CUDA, MPS, or another vendor's Torch
installation.

### Tested source install for learned-IQA

The `learned-iqa` extra selects `pyiqa`, but its transitive model stack can
resolve differently without constraints. The tested CPU source setup used by
the model smoke workflow is:

```bash
python -m pip install --upgrade pip setuptools wheel packaging \
  -c scripts/release-constraints.txt \
  -c scripts/release-constraints-torch.txt
python -m pip install torch torchvision \
  --index-url https://download.pytorch.org/whl/cpu \
  -c scripts/release-constraints.txt \
  -c scripts/release-constraints-torch.txt
python -m pip install -e ".[learned-iqa]" \
  -c scripts/release-constraints.txt \
  -c scripts/release-constraints-torch.txt
python -m pip check
```

This pins the common Python model stack and CPU Torch pair used by the
qualification workflow. Accelerator source installs use their platform-specific
constraints and vendor instructions below. Matching these pins does not replace
model preparation or hardware validation.

## Runtime support boundaries

The release matrix describes what ShotSieve builds and ships; it is not a
certification of every device from a vendor.

| Runtime | Current project boundary |
|---|---|
| CPU | Broadest fallback, subject to the platform, Python version, RAM/disk, and selected model. |
| NVIDIA CUDA | PyTorch 2.14.0 from the cu130 index. The active GPU must be covered by the wheel's compiled kernels, and the driver and model VRAM must also be suitable. |
| Intel XPU | PyTorch 2.14.0 + XPU on the Intel GPU/OS/driver combinations listed by the pinned PyTorch guide. |
| AMD ROCm | AMD ROCm 7.2.1 wheels on the exact product/OS/driver/Python combinations in AMD's matrix. Linux is the broader path; Windows is narrower. |
| AMD ROCm 10 `gfx1103` candidate | Stable ROCm 10.0.0 / PyTorch 2.13.0 / TorchVision 0.28.0 on Python 3.12, scoped to the Radeon 780M architecture. Windows/Linux packs are wired, but hardware validation is still outstanding. |
| Apple MPS | Apple Silicon with an MPS-capable PyTorch build and supported macOS. |

Auto mode may fall back to CPU when an accelerator is unavailable. An explicit
accelerator request remains an error. A successful package install or
`pip check` is not model or hardware evidence; use the vendor guides and
model-smoke procedure for that.

## Launching from source

The source entry point is:

```bash
shotsieve-desktop
```

The supported command-line options are:

```bash
shotsieve-desktop --data-dir ./shot-data
shotsieve-desktop --model-cache-dir ./model-cache
shotsieve-desktop --host 127.0.0.1 --port 9001 --no-browser
```

`--data-dir` controls the local database, previews, runtime sidecars, logs,
and readiness records. `--model-cache-dir` supplies defaults for Hugging Face
and Torch cache locations while preserving explicitly set `HF_HOME`,
`HF_HUB_CACHE`, and `TORCH_HOME`.

Source checkouts use `<checkout>/data` by default. Frozen runtime packs use a
`data/` directory next to the launcher. Installed packages outside a checkout
use the platform app-data directory.

## Release builds and portable bundles

### Runtime packs and sidecars

The current release matrix defines ten established runtime-pack targets and
two ROCm 10 `gfx1103` candidates:

- `windows-cpu`, `windows-nvidia-cuda`, `windows-intel-xpu`,
  `windows-amd-rocm`, `windows-amd-rocm10-gfx1103`
- `linux-cpu`, `linux-nvidia-cuda`, `linux-intel-xpu`, `linux-amd-rocm`,
  `linux-amd-rocm10-gfx1103`
- `macos-cpu`, `macos-apple-mps`

The authoritative target metadata is in
[`src/shotsieve/release_targets.py`](../src/shotsieve/release_targets.py) and
can be printed with:

```bash
python scripts/release_target_matrix.py --kind runtime
```

Local Windows builds use the runtime-only release script:

```powershell
./scripts/build_windows_releases.ps1 -PlanOnly
./scripts/build_windows_releases.ps1
```

Useful examples:

```powershell
# Inspect the JSON target plan.
./scripts/build_windows_releases.ps1 -Mode runtime -PlanOnly -AsJson

# Build one target.
./scripts/build_windows_releases.ps1 -Mode runtime -TargetIds windows-nvidia-cuda

# Select a Python interpreter matching the target's declared version.
./scripts/build_windows_releases.ps1 -PythonExe C:\Python313\python.exe `
  -Mode runtime -TargetIds windows-nvidia-cuda
```

The Windows outputs use the `ShotSieve-windows-<vendor-runtime>` naming scheme.
The release and preview workflows build the full twelve-target matrix. Runtime
pack archives are torchless: the build environment installs Torch so
PyInstaller can analyze imports, but the published archive contains neither
Torch nor model weights.

Pushes to `0.5.x` run the `preview-build` workflow. It runs the cross-platform
tests, builds all twelve portable targets, smoke-tests each bundle, and uploads
the archives as Actions artifacts for seven days. It does not create or update a
GitHub Release; do not use the `ci-release` workflow for branch previews because
its manual path publishes a release. Actions artifacts are separate from the
Releases page, but anyone with repository read access can download them.

On first use, the frozen launcher derives its target from its current launcher
name and may install the matching Torch sidecar under
`data/runtime/site-packages/<target-id>`. The install is staged and marked only
after validation. Install logs and state remain under `data/runtime/`.

Ordinary noninteractive launches do not download optional AI packages. An
interactive launch may ask for consent. Deliberate startup automation is
available with:

```text
SHOTSIEVE_BOOTSTRAP_AUTO_INSTALL_TORCH=1
SHOTSIEVE_BOOTSTRAP_AUTO_INSTALL_LEARNED_IQA=1
```

Setting either variable to `0`, declining the prompt, going offline, or
encountering a failed download leaves Catalog and Review usable while learned
models remain unavailable. Runtime installation is separate from Settings >
**Prepare selected model**, which downloads model assets and performs a small
validation run. No model weights are included in a portable archive.

Runtime sidecar selections are target-specific:

- CPU uses the pinned PyTorch CPU index.
- NVIDIA CUDA uses the pinned cu130 index.
- Intel XPU uses the pinned official XPU index.
- AMD Windows/Linux targets use the exact AMD ROCm 7.2.1 package URLs and
  constraints.
- The `*-amd-rocm10-gfx1103` candidate targets use the stable AMD index,
  pinned ROCm 10 / PyTorch package extras, and a locally built selector wheel
  bundled for frozen first-run installation.
- macOS CPU/MPS uses the supported default PyTorch packages.

The learned-model catalog is `topiq_nr`, `clipiqa`, and `qrealign-mini`. The
current common learned-IQA pins are defined in
[`src/shotsieve/dependency_constraints.py`](../src/shotsieve/dependency_constraints.py)
and the release constraint files. The app discovers models from the usable
runtime; catalog membership alone is not hardware certification.

### Archive integrity

The release workflow generates `bootstrap-manifest.json` from the exact archive
files and records a SHA-256 digest for each archive. Bootstrap validates the
digest before reusing, extracting, or launching an acquired runtime asset. It
extracts into a staging directory, validates the expected launcher, writes the
checksum marker, and only then publishes the install. Split release assets are
reassembled and checked against the complete archive digest.

These checks detect transfer or storage mismatches; they do not replace
publisher identity, signing, SBOMs, or broader artifact provenance. An
explicitly supplied executable in a frozen bundle is a separate distribution
trust boundary.

## Testing and verification

Run the ordinary suite with:

```bash
python -m pytest -q
```

Run the configured lint checks with:

```bash
python -m pip install -e ".[lint]"
python -m ruff check src tests
```

The narrower repo-wide dead-import guard is also useful when changing package
imports:

```bash
python -m pip install -e .[lint]
python -m ruff check --select F401 src/shotsieve
```

The browser tests use Playwright. Install Chromium once per environment:

```bash
python -m playwright install chromium
```

For a quick manual visual pass, use the [visual QA checklist](accessibility-checklist.md).
That checklist is intentionally about visual usability, not broad accessibility
conformance.

Pull requests and direct pushes to `main` run the offline suite. CI also runs a
Python 3.14 core suite without browser or accelerator claims and smoke-tests an
installed wheel outside the checkout. CI treats missing Chromium or a failed
browser launch as an error; local runs may skip browser-marked tests when the
browser is unavailable.

The manual/weekly model-smoke workflow prepares TOPIQ, CLIPIQA, and Q-ReAlign
Mini in fresh caches, repeats the checks offline, records resolved dependency
versions, and uploads sanitized JSON reports. It does not upload photos, model
weights, or caches.

The manual `dependency-upgrade-qualification` workflow tests Hub-only,
Transformers-only, and combined candidate pins against all three models
without changing the release constraints. It checks the Q-ReAlign lazy import,
runs online/offline smokes, and uploads reports for comparison with the
current-pin workflow. It also uploads baseline/candidate resolved package
inventories. A Windows CPU lane repeats Q-ReAlign to check the Hub cache's
Windows symlink policy. It is a qualification workflow, not a release or GPU
test.

## Performance measurement

Performance diagnostics are opt-in and are not part of the ordinary test
suite. Use [performance-measurement.md](performance-measurement.md) before
changing indexes, pagination, catalog storage, preview generation, or model
execution. It separates database, filesystem, preview, model startup, and
inference costs.

## Release procedure

Prepare version references and the changelog first:

```powershell
./scripts/prepare_release.ps1 -Version 1.2.3
```

Review and commit those changes. Then create and push the annotated tag:

```powershell
./scripts/create_github_release.ps1 -Version v1.2.3
```

The tag helper does **not** edit version files. GitHub Actions publishes the Python
distributions, twelve runtime packs, checksummed bootstrap manifest, and split
archive parts when required. Use `-DryRun` on the tag helper to inspect its
checks before making the tag.

If artifact publishing is interrupted after a successful build, rerun the
`ci-release` workflow for the existing tag using its `release_tag` input.
