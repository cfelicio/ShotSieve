# Contributing

Read [architecture.md](architecture.md) before changing a route, persisted
data shape, runtime target, or frontend workflow. This project has several
compatibility boundaries that are more important than a local refactor's
convenience.

## Development setup

Use an isolated Python 3.13+ environment; Python 3.14 is preferred:

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
python -m playwright install chromium
```

Add `format-loaders` for HEIF/RAW support, `learned-iqa` only in its tested or
platform-specific environment, and `windows-build` only for the native
Windows release workflow. Keep CPU, CUDA, XPU, ROCm, and MPS Torch
environments separate. See [building.md](building.md) and
[configuration.md](configuration.md).

## Change workflow

1. Identify the contract being changed: catalog/migration, scan, preview,
   model/runtime, HTTP route, job lifecycle, file operation, or browser
   facade.
2. Read the relevant implementation and tests before editing.
3. Preserve path safety, root scoping, selection revisions, and the
   torchless-runtime boundary unless the change explicitly revises that
   contract.
4. Update focused tests and the corresponding public guide when behavior is
   intentionally changed.
5. Run the focused tests, then the ordinary suite and Ruff before handoff.

Do not commit `data/`, model caches, runtime sidecars, logs, build output,
temporary wheel-smoke environments, or generated release artifacts. These are
local/runtime state, not source inputs.

## Verification commands

```bash
python -m pytest -q
python -m ruff check src tests
python -m ruff check --select F401 src/shotsieve
python -m pip check
```

Install Chromium before running browser-marked tests. The performance baseline
is opt-in and is not part of the ordinary suite:

```bash
SHOTSIEVE_RUN_PERFORMANCE_BASELINE=1 python -m pytest tests/test_performance_baseline.py -s -q
```

On PowerShell:

```powershell
$env:SHOTSIEVE_RUN_PERFORMANCE_BASELINE = "1"
python -m pytest tests/test_performance_baseline.py -s -q
```

The model-smoke workflow is separate from the normal suite because it needs
fresh CPU Torch/model environments and online/offline cache preparation.

## Ownership and test selection

- Catalog, path normalization, migrations, preview ownership, and scanner
  changes: run the catalog/scan/database tests.
- Route, error, job, selection, and file-operation changes: run the matching
  `test_web_*` and job-registry tests, including integration tests.
- Static JavaScript changes: run the Playwright/browser tests and the focused
  frontend contract tests.
- Learned-IQA changes: run runtime/catalog tests and the constrained model
  smoke path when the required environment is available.
- Release/bootstrap changes: inspect the target matrix, run release-script and
  bootstrap tests, and preserve launcher/archive names.

The installed-wheel smoke used by CI builds with:

```bash
python -m pip install build
python -m build --wheel --outdir dist
```

Install the resulting wheel in a temporary environment outside the checkout,
run `python -m pip check`, and run `shotsieve-desktop --help`. Do not use the
checkout's editable import to claim that the wheel is healthy.

## Compatibility surfaces

Treat the CLI flags, loopback security policy, HTTP payloads and error
statuses, selection revisions, persisted/exported data, model IDs,
`release_targets.py`, and `window.ShotSieveWorkflows` as public compatibility
surfaces. The frontend script order and facade names are intentionally
documented in [frontend-workflows.md](frontend-workflows.md).

Comments and docstrings should explain intent, rationale, invariants, or
operational constraints. Do not add comments that only repeat the next line
of code.
