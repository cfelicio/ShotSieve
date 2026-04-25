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
- Learned IQA support: `python -m pip install -e .[learned-iqa]`
- DirectML learned IQA support on supported Windows Python versions: `python -m pip install -e .[learned-iqa-directml]`
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
shotsieve-desktop --host 127.0.0.1 --port 9001 --no-browser
```

Startup/runtime notes:

- Frozen portable builds use a writable `data/` folder next to the executable
- Editable source checkouts (`pip install -e .` from this repository) also default to `<repo>/data`
- Installed packages outside a source checkout fall back to `%LOCALAPPDATA%\ShotSieve` on Windows, `%APPDATA%\ShotSieve` as a Windows fallback, or `~/.shotsieve` when platform app-data variables are unavailable
- On any install style, `--data-dir` overrides the default location

On startup, `shotsieve-desktop` may attempt sidecar installation or repair for missing learned-IQA dependencies and CUDA PyTorch runtimes. Current controls:

- `SHOTSIEVE_BOOTSTRAP_AUTO_INSTALL_TORCH=1` to auto-install CUDA-sidecar PyTorch without prompting
- `SHOTSIEVE_BOOTSTRAP_AUTO_INSTALL_TORCH=0` to skip the CUDA-sidecar install prompt
- `SHOTSIEVE_BOOTSTRAP_AUTO_INSTALL_LEARNED_IQA=1` to auto-install learned-IQA dependencies without prompting
- `SHOTSIEVE_BOOTSTRAP_AUTO_INSTALL_LEARNED_IQA=0` to skip learned-IQA auto-install

Portable and frozen builds prefer the bundled pip-based installer paths from `shotsieve.bootstrap`; if that installer is unavailable, ShotSieve keeps running but learned backends may stay disabled.

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