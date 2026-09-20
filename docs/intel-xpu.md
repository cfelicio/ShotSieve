# Intel XPU install and release track

ShotSieve can use native PyTorch XPU through the `xpu` runtime target. Windows
and Linux runtime packs include this track, and the same pinned instructions
can be used for source installs.

The commands below pin the Torch pair used by this checkout and must be run in
a fresh virtual environment. Do not mix the XPU wheels with the CPU, CUDA, or
ROCm Torch wheels in an existing environment.

## Supported boundary and prerequisites

This is a target-specific PyTorch XPU path, not a promise that every Intel
GPU or integrated graphics device is supported. PyTorch's current 2.14 XPU
guide lists these validated hardware families:

- **Data Center GPU Max:** RHEL 9.2, SLES 15 SP5, or Ubuntu Server 22.04 with
  the listed Intel GPU driver stack.
- **Client GPUs:** Arc A-Series, Arc B-Series, Core Ultra with Arc graphics
  (Meteor Lake-H), Core Ultra Series 2 with Arc graphics (Arrow Lake-H), and
  Core Ultra Mobile Series 2 with Arc graphics (Lunar Lake), on the operating
  systems listed by PyTorch.
- **Panther Lake:** listed with narrower current OS requirements than the
  other client families; do not infer support for every Core Ultra generation.

See [PyTorch's 2.14 Intel GPU guide](https://docs.pytorch.org/docs/2.14/notes/get_start_xpu.html)
for the exact OS/driver table. Older Intel GPUs, unsupported integrated
graphics, an unsupported OS/driver combination, or a CPU-only Torch install
must use CPU unless a new target-specific validation has been recorded.

**Intel Iris Xe warning:** “Iris Xe” is a product label used across multiple
generations. The current PyTorch 2.14 Windows validation table names Core Ultra
systems with **Arc** graphics, not older laptop systems marketed as Intel Iris
Xe. Do not assume that an Iris Xe laptop is an XPU-supported device; record the
exact adapter and CPU generation first. For an older Iris Xe system, CPU is the
supported ShotSieve fallback unless a separate native XPU smoke test succeeds.

- Python 3.13 is the reproducible ShotSieve validation interpreter for this
  track. The current Torch 2.14.0 XPU index publishes several CPython
  variants, but this project targets 3.13; each interpreter still needs its
  own `pip check` and model smoke evidence.
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
- [Intel PyTorch 2.14 prerequisites](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu/2-14.html)
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
The bundle's presence and a successful import do not establish XPU support;
the native tensor check below is required. If the device or driver is outside
the PyTorch matrix, select CPU.

Portable XPU bundles install Torch and its native Intel dependencies on first
use. The launcher also exposes the sidecar's `Library/bin` and Torch library
directories to Windows DLL loading. A fixed 0.5.x build invalidates an older
incomplete sidecar and installs the dependency-complete set again. If the
runtime still reports `WinError 126` after that repair, the missing component
is outside the application bundle—most commonly the Intel graphics/Level Zero
driver, Microsoft Visual C++ runtime, or unsupported hardware—and the native
check below should be run directly to distinguish those cases.

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

Repeat those two commands with `clipiqa` and, if Q-ReAlign Mini is included in
the target claim, with `qrealign-mini`. Use a fresh cache
and data directory for every model. Never put source photos, model weights, or
private paths in a report intended for sharing.

Record a failed explicit-XPU run as unavailable rather than silently rerunning
it on CPU. Separately run one `--device cpu` smoke when documenting the
fallback behavior. The packaged Windows/Linux XPU targets use the same pinned
Torch pair and learned-model catalog; source-install evidence does not replace
the target-specific packaged-bundle and hardware checks required for a release
claim.
