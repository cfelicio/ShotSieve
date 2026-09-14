# Intel XPU source-install track

ShotSieve can use native PyTorch XPU through the `xpu` runtime target. This is
an isolated source-install track, not a packaged runtime target. The release
matrix intentionally continues to publish CPU, CUDA, and Apple MPS packs only.

The commands below pin the Torch pair used by this checkout and must be run in
a fresh virtual environment. Do not mix the XPU wheels with the CPU, CUDA, or
ROCm Torch wheels in an existing environment.

## Prerequisites

- A supported Intel Arc GPU or Intel Core Ultra system with an Intel GPU
  exposed to the operating system. A model name alone is not sufficient;
  confirm the device is in Intel's current driver support list.
- Python 3.13 is the reproducible ShotSieve validation interpreter for this
  track. The published Torch 2.14.0 XPU wheels currently include CPython 3.11,
  3.12, 3.13, and 3.14 variants for Windows and Linux; each interpreter still
  needs its own `pip check` and model smoke evidence.
- The current Intel graphics driver for Windows, or the Intel GPU/Level Zero
  driver stack for Linux. Use the driver installation instructions for the
  exact operating system and record the installed driver version.
- Enough writable disk space for the learned-IQA caches and model assets. Put
  them on a local SSD when possible; the `--model-cache-dir` path below keeps
  model downloads separate from the repository.

Official references:

- [PyTorch Intel GPU/XPU documentation](https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html)
- [PyTorch XPU wheel index](https://download.pytorch.org/whl/xpu/torch/)
- [Intel Arc graphics Windows drivers](https://www.intel.com/content/www/us/en/download/785597/intel-arc-graphics-windows.html)
- [Intel GPU driver documentation](https://dgpu-docs.intel.com/)

## Install on Windows

Use PowerShell from the ShotSieve checkout:

```powershell
py -3.13 -m venv .venv-xpu
.\.venv-xpu\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv-xpu\Scripts\python.exe -m pip install `
  "torch==2.14.0+xpu" "torchvision==0.29.0+xpu" `
  --index-url https://download.pytorch.org/whl/xpu `
  --extra-index-url https://pypi.org/simple `
  -c scripts/source-constraints-xpu.txt
.\.venv-xpu\Scripts\python.exe -m pip install -e ".[learned-iqa]" `
  -c scripts/source-constraints-xpu.txt
.\.venv-xpu\Scripts\python.exe -m pip check
```

Record the Intel driver version from Device Manager or `dxdiag`. If more than
one Intel adapter is present, record the adapter name alongside the version.

## Install on Linux

Use a distribution-supported Python 3.13 and install the Intel GPU/Level Zero
driver prerequisites before creating the environment:

```bash
python3.13 -m venv .venv-xpu
. .venv-xpu/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install \
  "torch==2.14.0+xpu" "torchvision==0.29.0+xpu" \
  --index-url https://download.pytorch.org/whl/xpu \
  --extra-index-url https://pypi.org/simple \
  -c scripts/source-constraints-xpu.txt
python -m pip install -e '.[learned-iqa]' \
  -c scripts/source-constraints-xpu.txt
python -m pip check
```

Record the kernel, Intel GPU driver/Level Zero package versions, GPU name,
Python version, and Torch wheel version. Do not infer Linux support from a
Windows run or from a CPU-only Torch installation.

## Verify the native XPU runtime

Run this in the same fresh environment before starting ShotSieve:

```bash
python -c "import torch; assert torch.xpu.is_available(), 'PyTorch XPU is unavailable'; print({'torch': torch.__version__, 'xpu': torch.xpu.is_available(), 'devices': torch.xpu.device_count(), 'name': torch.xpu.get_device_name(0)}); x=torch.randn((64, 64), device='xpu'); print(x @ x, x.device); torch.xpu.synchronize()"
```

This checks an actual XPU tensor operation. If it fails, keep ShotSieve on
CPU and retain the failure output; do not record the accelerator as available.

## Run ShotSieve model evidence

Start the source entry point with a disposable data directory and an explicit
cache root:

```bash
python -m shotsieve.desktop \
  --data-dir ./build/xpu-evidence-data \
  --model-cache-dir ./build/xpu-evidence-cache
```

For each model, run the online command and then a new process with `--offline`.
The smoke report records the exact Python/Torch package versions, resolved
runtime, one-image raw/normalized score, model backend version, cache paths,
and peak accelerator memory when the runtime exposes that counter. Supply the
driver version captured from the operating system with `--driver-version`.

```bash
python scripts/model_smoke.py \
  --model topiq_nr --device xpu \
  --cache-dir ./build/xpu-evidence-cache/topiq_nr \
  --data-dir ./build/xpu-evidence-data/topiq_nr \
  --driver-version '<Intel driver version>' \
  --report-path ./build/audit-reports/xpu-topiq_nr-online.json

python scripts/model_smoke.py \
  --model topiq_nr --device xpu --offline \
  --cache-dir ./build/xpu-evidence-cache/topiq_nr \
  --data-dir ./build/xpu-evidence-data/topiq_nr \
  --driver-version '<Intel driver version>' \
  --report-path ./build/audit-reports/xpu-topiq_nr-offline.json
```

Repeat those two commands with `clipiqa`. Repeat with
`qrealign-mini` only after W04 adds the model to the product catalog; its
absence before W04 is expected and is not an XPU failure. Use a fresh cache
and data directory for every model. Never put source photos, model weights, or
private paths in a report intended for sharing.

Record a failed explicit-XPU run as unavailable rather than silently rerunning
it on CPU. Separately run one `--device cpu` smoke when documenting the
fallback behavior. A successful source-install smoke does not authorize a
packaged XPU target; PyInstaller collection and a dedicated release target
remain intentionally out of scope until separately tested.
