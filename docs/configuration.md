# Configuration

ShotSieve has two distinct configuration roots:

- `--data-dir` controls the local catalog, previews, runtime sidecars,
  installation logs, and model-preparation record.
- `--model-cache-dir` supplies defaults for downloaded model/framework assets.

The application does not load `.env` files itself. If you use
[`.env.example`](../.env.example), load the variables through your shell,
IDE, or another launcher before starting ShotSieve.

## CLI configuration

The supported desktop options are:

| Option | Default or effect |
| --- | --- |
| `--data-dir PATH` | Uses `PATH` instead of the platform/source default. |
| `--model-cache-dir PATH` | Sets default Hugging Face and Torch cache roots. Explicit cache environment variables take precedence. |
| `--host HOST` | `127.0.0.1`; changes the bind address but not the loopback request policy. |
| `--port PORT` | `8765`. |
| `--no-browser` | Starts the server without opening the default browser. |
| `--check-runtime` | Checks native Torch/TorchVision operations and learned imports, then exits without starting the UI. |

Examples:

```bash
shotsieve-desktop --data-dir ./shot-data --no-browser
shotsieve-desktop --model-cache-dir ./model-cache
shotsieve-desktop --port 9001 --no-browser
shotsieve-desktop --check-runtime
```

Source checkouts normally use `<checkout>/data`; frozen launchers use a
`data/` directory beside the launcher; installed packages outside a checkout
use the platform app-data directory. The CLI data directory overrides those
defaults.

## Environment variables

The variables below are read by the implementation. Values such as `1`,
`true`, `yes`, and `on` are enabled for the boolean switches where noted.

| Variable | Purpose |
| --- | --- |
| `HF_TOKEN` | Optional Hugging Face access token for authenticated or higher-rate downloads. |
| `HF_HOME` | Hugging Face cache root. |
| `HF_HUB_CACHE` or `HUGGINGFACE_HUB_CACHE` | Hugging Face Hub cache root. |
| `TORCH_HOME` | Torch/PyTorch model cache root. |
| `XDG_CACHE_HOME` | Base cache root used when more specific cache roots are not set. |
| `HF_HUB_OFFLINE` and `TRANSFORMERS_OFFLINE` | Make model loading/download behavior offline for a fresh process. Offline mode cannot satisfy a missing cache. |
| `SHOTSIEVE_BOOTSTRAP_AUTO_INSTALL_TORCH` | Permit automatic target-specific Torch sidecar installation at startup. |
| `SHOTSIEVE_BOOTSTRAP_AUTO_INSTALL_LEARNED_IQA` | Permit automatic installation of the optional learned-IQA package at startup. |
| `SHOTSIEVE_BOOTSTRAP_MANIFEST_URL` | Override the bootstrap manifest URL or local manifest path. The `--manifest-url` bootstrap option takes precedence. |
| `SHOTSIEVE_GITHUB_TOKEN` or `GITHUB_TOKEN` | Optional token for private GitHub release assets used by bootstrap. |
| `SHOTSIEVE_PERFORMANCE_LOGGING` | Emit opt-in aggregate performance JSON diagnostics. |
| `SHOTSIEVE_RUN_PERFORMANCE_BASELINE` | Opt in to the performance-baseline test module; it is not part of the ordinary suite. |

`--model-cache-dir` uses setdefault-style behavior: it supplies
`<root>/huggingface`, `<root>/huggingface/hub`, and `<root>/torch` only when
the corresponding environment variables are not already set. An explicitly
configured `HF_HOME`, `HF_HUB_CACHE`, `HUGGINGFACE_HUB_CACHE`, or `TORCH_HOME`
therefore remains authoritative.

POSIX shell:

```bash
export HF_HUB_OFFLINE=1
export SHOTSIEVE_PERFORMANCE_LOGGING=1
shotsieve-desktop --model-cache-dir ./model-cache
```

PowerShell:

```powershell
$env:HF_HUB_OFFLINE = "1"
$env:SHOTSIEVE_PERFORMANCE_LOGGING = "1"
shotsieve-desktop --model-cache-dir .\model-cache
```

Do not include tokens, full private paths, or cache contents in issue reports.
Diagnostic output is designed to redact sensitive environment names, but
captured environment files and command transcripts still require review.

## UI analysis settings

The web UI sends scan/score settings with each operation. The current defaults
and supported values are:

- default extensions cover common JPEG/PNG/TIFF images, HEIF, and the RAW
  extensions defined in `src/shotsieve/config.py`; `raw` and `.raw` expand to
  the supported RAW set;
- recursive scanning is enabled by default;
- ignore rules are newline-separated patterns;
- RAW preview mode is `fast`, `auto` (the default), or `high-quality`;
- source decode is 64 megapixels by default and accepts 1 through 256
  megapixels;
- device policy defaults to `auto`, with explicit `cpu`, `cuda`, `rocm`,
  `xpu`, or `mps` targets when the installed runtime supports them;
- the learned-model catalog is `topiq_nr`, `clipiqa`, and `qrealign-mini`;
  each currently defaults to batch size 4, subject to runtime availability;
- the Review page requests 60 files per page; the resource profiles are
  `low`, `normal`, and `aggressive`.

These are operation/UI settings, not shell environment variables. Use the
Settings and Analyze controls or the documented HTTP payloads in
[api.md](api.md).

## Data and cache separation

Keep the data directory and model cache separate when possible:

```text
data/
  shotsieve.db
  previews/
  runtime/
  model-preparation.json
model-cache/
  huggingface/
  torch/
```

Runtime sidecars and model weights are not the same artifact. A successful
sidecar installation does not mean that a selected model is prepared, and
preparing a model does not add Torch to a portable release archive.
