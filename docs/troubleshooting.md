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
