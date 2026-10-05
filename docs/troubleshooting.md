# Troubleshooting

Start with the smallest diagnostic that distinguishes the failure. Do not
delete the catalog or model cache as a first response; those directories
contain useful state and downloaded assets.

## The command or imports fail

Confirm the interpreter and editable install:

```bash
python --version
python -m pip install -e ".[test]"
python -m pip check
shotsieve-desktop --help
```

ShotSieve requires Python 3.13 or newer. The release and tested learned-IQA
paths use Python 3.14. If an optional import fails, install only the relevant
extra and keep accelerator-specific Torch environments isolated; do not mix
CPU, CUDA, XPU, ROCm, and MPS packages in one environment.

## The UI opens but learned models are unavailable

Run:

```bash
shotsieve-desktop --check-runtime
```

This checks native Torch/TorchVision operations and learned imports. A
successful runtime check still does not download model weights. Use Settings >
**Prepare selected model** after the runtime is available. The current model
IDs are `topiq_nr`, `clipiqa`, and `qrealign-mini`.

For a packaged runtime, inspect `data/runtime/` for sidecar state and install
logs. Sidecar installation and model-weight preparation are separate steps.
Catalog and Review remain usable when optional AI installation is declined,
offline, or unsuccessful.

## Runtime installation reports a deeply nested or unwritable folder

ShotSieve checks the final runtime destination before installing or repairing
Torch and learned-IQA dependencies. The check reserves space for deep native
header/library paths and probes the destination filesystem before downloading
packages.

If the path is too deeply nested, move the entire portable ShotSieve folder to
a shorter location, such as `C:\ShotSieve`, and launch it again. If the check
reports a permission or filesystem error, confirm that the runtime folder is
writable and has free space before retrying. A rejected preflight does not
install runtime files or replace the existing sidecar.

## The accelerator is visible but scoring fails

Device visibility alone does not establish compatibility. Intel XPU selection
requires an Alchemist-or-newer architecture reported by PyTorch; missing or
unknown metadata is also rejected. Auto warns and falls back to CPU, while an
explicit XPU request reports a controlled error. ROCm rejects a concrete GPU
architecture that is absent from the installed Torch wheel. See the
[Intel XPU guide](intel-xpu.md) and [AMD ROCm guide](amd-rocm.md) for validation.

Windows ROCm MIOpen can fail to compile BatchNorm with
`HIPRTC_ERROR_COMPILATION` and a missing `type_traits` header. ShotSieve applies
a BatchNorm-only workaround to TOPIQ and CLIPIQA, leaving MIOpen convolutions
enabled; no Visual Studio Build Tools or external compiler installation is
required. See the [workaround details and validation limits](amd-rocm.md#windows-miopen-batchnorm-workaround).

Clear runtime-wide compiler or fatal device failures abort scoring rather than
retrying every image. Choose CPU or repair and validate the accelerator runtime
before retrying. Ordinary batch failures, out-of-memory errors, and a generic
`miopenStatusUnknownError` still receive individual-image retries; that generic
status alone does not prove a compiler failure.

## Downloads fail or offline mode reports a missing asset

Check the effective cache roots and permissions. `--model-cache-dir` provides
defaults but preserves explicitly set `HF_HOME`, `HF_HUB_CACHE`,
`HUGGINGFACE_HUB_CACHE`, and `TORCH_HOME`. `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1` require every needed asset to already be cached.
`HF_TOKEN` can help with Hugging Face rate limits where access is permitted.

For a fresh-process check:

```powershell
$env:HF_HUB_OFFLINE = "1"
shotsieve-desktop --model-cache-dir .\model-cache --check-runtime
```

If the model cache or sidecar path is read-only or full, choose a writable
`--data-dir`/`--model-cache-dir` and retry. Do not copy tokens into an issue
report.

## The browser does not open, or browser tests fail

Use `--no-browser` and open the printed local URL manually:

```bash
shotsieve-desktop --no-browser --port 9001
```

For browser tests, install Chromium in the active environment:

```bash
python -m playwright install chromium
```

CI installs the browser with its platform dependencies. A local test run can
skip browser-marked tests when Chromium is unavailable, but CI treats a
missing browser or failed launch as an error.

## The port is busy or requests are forbidden

Choose an unused port and use the same port in the browser URL:

```bash
shotsieve-desktop --host 127.0.0.1 --port 9001 --no-browser
```

The server validates loopback Host and, when supplied, loopback Origin
headers. An address such as `0.0.0.0` does not provide supported remote/LAN
access. A reverse proxy or hosted deployment is outside this repository's
supported operational model.

## Scans find too much or too little

Check the selected root, recursive toggle, extension list, and ignore rules.
The default extension set and RAW expansion are defined in
`src/shotsieve/config.py`. The scan is root-scoped and does not infer that an
unavailable or unmounted root is empty. Enumeration failures are reported as
errors.

For very large images, lower the source decode budget; the default is 64
megapixels and the accepted range is 1 through 256. Review
`/api/analysis-diagnostics` or the Settings diagnostics before changing
database state.

## Missing-entry cleanup or file operations are surprising

Preview missing entries first, then apply cleanup to the reviewed root:

```text
GET  /api/cache/missing/preview?root=ROOT
POST /api/cache/missing/apply
```

Unavailable roots are treated as unknown and are not a list of missing files.
Copy, move, and delete results retain per-file status. Completed or uncertain
mutations may remain visible after a later failure or cancellation; inspect the
operation result before retrying.

Bulk actions require a current `selection_revision`. Refresh the catalog and
reselect files when the server rejects a stale revision.

## A job ID no longer exists or SQLite is busy

Jobs are process-local and disappear after a process crash or restart. Check
the catalog and affected files manually, then start a new operation only after
confirming the prior operation's per-file outcome.

SQLite contention returns a conflict rather than silently overwriting state.
Wait for the current operation to finish and retry. Keep one active analysis or
file operation at a time from the UI.

## What to include in a report

Include the ShotSieve version, Python version, platform, command shape, route
or workflow, sanitized error text, and whether the problem reproduces with
CPU/online settings. Redact tokens, private paths, cache roots, personal photo
names, model artifacts, and full environment dumps.
