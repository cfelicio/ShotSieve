# AMD ROCm install and release track

ShotSieve can use native AMD ROCm through the logical `rocm` runtime target
(`amd` is its vendor alias). Windows and Linux runtime packs include the legacy
ROCm 7.2.1 track. Separate ROCm 10.0 `gfx1103` candidate packs target Radeon
780M-class hardware; they are not a replacement for the existing targets
until native validation is complete.

This guide documents both the legacy ROCm 7.2.1 / PyTorch 2.9.1 track and a
separate ROCm 10.0 / PyTorch 2.13 candidate for the Radeon 780M (`gfx1103`).
AMD's ROCm 10.0 matrix lists the 780M for specific Windows/Linux configurations,
but that does not make the legacy 7.2.1 ShotSieve pack compatible with it.
Use a fresh environment and do not mix the 7.2.1 wheels with the common
CPU/CUDA/MPS or Intel XPU Torch environments. The exact GPU, OS, driver, Python,
Torch, and ROCm combination must appear in AMD's current compatibility matrix
before a run is treated as evidence.

## Supported boundary and prerequisites

- **Linux first:** ROCm 7.2.1's Radeon PyTorch matrix currently lists selected
  RX 7000/9000 products (including RX 7700/7800/7900 and RX 9060/9070 family
  entries), plus the listed Radeon PRO and AI PRO products, on AMD-supported
  Linux distributions and kernels. This is a selected list, not all Radeon
  cards or all `gfx` architectures.
- **Windows is optional and narrower:** the documented 7.2.1 PyTorch path is
  for explicitly listed Radeon/Ryzen hardware, Python 3.12, and the required
  AMD graphics driver. AMD documents PyTorch on Windows for this path, not the
  complete Linux ROCm framework stack; do not infer Windows support for other
  ROCm frameworks or arbitrary AMD GPUs.
- **Ryzen is selective:** AMD's 7.2.1 PyTorch documentation includes selected
  Ryzen AI Max 300, AI 400, and AI 300 products. A generic AMD APU, older
  Radeon, Instinct/data-center card, or unlisted `gfx` target is not covered
  by this ShotSieve pack unless it appears in the exact matrix and passes the
  native and model checks below.
- Keep enough system memory and local disk for the model caches. Put the
  disposable data and cache roots outside the repository when possible.
- Record the GPU marketing name, `gfx` architecture, OS/kernel, Python,
  PyTorch wheel, ROCm/HIP version, AMD driver, cache paths, and `pip check`
  output for every validation host.

Official references:

- [AMD Radeon Linux compatibility matrix](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/native_linux/native_linux_compatibility.html)
- [AMD Radeon Windows compatibility matrix](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/windows/windows_compatibility.html)
- [AMD ROCm on Radeon and Ryzen overview](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/)
- [AMD Linux ROCm installation](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/native_linux/install-radeon.html)
- [AMD Linux PyTorch wheels](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/native_linux/install-pytorch.html)
- [AMD Windows PyTorch wheels](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/windows/install-pytorch.html)
- [ROCm license and disclaimers](https://rocm.docs.amd.com/en/latest/about/license.html)

## ROCm 10.0 `gfx1103` candidate (Radeon 780M)

This is a separate candidate for the Radeon 780M's `gfx1103` architecture. It
uses AMD's stable multi-architecture Python index with the published ROCm
`10.0.0`, PyTorch `2.13.0`, and TorchVision `0.28.0` packages. The AMD index
lists Windows and Linux wheels for Python 3.12 and the `device-gfx1103`
packages. This is not the old ROCm 7.2.1 package set; do not mix the two
environments.

As of the review date, AMD's compatibility matrix requires Windows 11 25H2
and lists Adrenalin driver 26.8.1 for this APU family. Linux entries list
Ubuntu 24.04.4 with the OEM 6.17 kernel or Ubuntu 26.04 with GA kernel 7.0.
Check AMD's live matrix before testing because driver and OS support can
change. The code and build workflow are integrated, but no 780M inference has
yet been run in this workspace.

For a source-install validation, create a clean Python 3.12 environment:

```powershell
py -3.12 -m venv .venv-rocm10
.\.venv-rocm10\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv-rocm10\Scripts\python.exe -m pip install `
  "torch[device-gfx1103]==2.13.0+rocm10.0.0" `
  "torchvision[device-gfx1103]==0.28.0+rocm10.0.0" `
  "rocm==10.0.0" `
  --index-url https://stable.repo.amd.com/rocm/whl-next/ `
  --extra-index-url https://pypi.org/simple `
  -c scripts/source-constraints-rocm10-gfx1103.txt
.\.venv-rocm10\Scripts\python.exe -m pip install -e ".[learned-iqa]" `
  -c scripts/source-constraints-rocm10-gfx1103.txt
.\.venv-rocm10\Scripts\python.exe -m pip check
```

On Linux, use the same requirements and indexes in a Python 3.12 environment
on an OS/kernel listed by AMD. For frozen runtime packs, the build creates a
local wheel from AMD's pinned ROCm selector source distribution using normal
Python, then includes it in the archive. The app's embedded installer uses
that wheel to avoid launching a PEP 517 subprocess through the frozen
executable; pip resolves the remaining SDK and `gfx1103` packages from AMD's
stable index.

To build the Windows candidate locally:

```powershell
./scripts/build_windows_releases.ps1 -TargetIds windows-amd-rocm10-gfx1103
```

Before calling the target usable, test the clean Python install and frozen
candidate on actual `gfx1103` hardware: `pip check`, Torch import, HIP/device
enumeration, a synchronized GPU tensor operation, then ShotSieve's online and
offline model smoke tests. GitHub Actions can validate dependency resolution
and bundling, but a hosted build runner cannot prove the laptop's
driver/device path.

## Install on supported Linux

First install the AMDGPU/ROCm system components for the exact distribution and
GPU from AMD's guide. The following is the currently documented Ubuntu 24.04
ROCm 7.2.1 setup; use the matching AMD commands for another listed system.

```bash
sudo apt update
sudo apt install python3-setuptools python3-wheel
wget https://repo.radeon.com/amdgpu-install/7.2.1/ubuntu/noble/amdgpu-install_7.2.1.70201-1_all.deb
sudo apt install ./amdgpu-install_7.2.1.70201-1_all.deb
sudo amdgpu-install -y --usecase=graphics,rocm
sudo usermod -a -G render,video "$USER"
# Reboot so the driver and group membership are active.
```

Verify the host before installing ShotSieve:

```bash
rocminfo
clinfo
```

Create a clean Python 3.12 environment in the ShotSieve checkout. These are
the exact AMD-published Linux wheels documented for ROCm 7.2.1/Ubuntu 24.04.

```bash
python3.12 -m venv .venv-rocm
. .venv-rocm/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install \
  "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/torch-2.9.1%2Brocm7.2.1.lw.gitff65f5bc-cp312-cp312-linux_x86_64.whl" \
  "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/torchvision-0.24.0%2Brocm7.2.1.gitb919bd0c-cp312-cp312-linux_x86_64.whl" \
  "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/torchaudio-2.9.0%2Brocm7.2.1.gite3c6ee2b-cp312-cp312-linux_x86_64.whl" \
  "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/triton-3.5.1%2Brocm7.2.1.gita272dfa8-cp312-cp312-linux_x86_64.whl"
python -m pip install -e '.[learned-iqa]' -c scripts/source-constraints-rocm.txt
python -m pip check
```

The wheel filenames and hashes can change when AMD publishes a new ROCm
release. Update the constraints and this guide together when selecting a new
validated pair; do not silently substitute an untested PyPI/nightly wheel.

## Optional supported Windows path

Use this path only when the exact host appears in AMD's Windows compatibility
matrix. For the current 7.2.1 path, AMD documents Python 3.12 and graphics
driver 26.2.2. The commands below install the AMD SDK components and the
published PyTorch wheels; they do not claim general Windows ROCm support.

```powershell
py -3.12 -m venv .venv-rocm
.\.venv-rocm\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv-rocm\Scripts\python.exe -m pip install --no-cache-dir `
  "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_core-7.2.1-py3-none-win_amd64.whl" `
  "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl" `
  "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl" `
  "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm-7.2.1.tar.gz"
.\.venv-rocm\Scripts\python.exe -m pip install --no-cache-dir `
  "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl" `
  "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torchvision-0.24.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl" `
  "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torchaudio-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl"
.\.venv-rocm\Scripts\python.exe -m pip install -e ".[learned-iqa]" `
  -c scripts/source-constraints-rocm-windows.txt
.\.venv-rocm\Scripts\python.exe -m pip check
```

If the host, driver, Python, or wheel is outside AMD's listed combination,
leave AMD on CPU and retain the failed/unavailable evidence. Do not infer
Windows support from a Linux run.

## Verify the native ROCm runtime

ROCm PyTorch uses the CUDA API surface for tensor devices. The HIP version and
the logical runtime are recorded separately so the app never reports AMD as
NVIDIA CUDA:

```bash
python -c "import torch; assert torch.version.hip, 'Torch is not a ROCm build'; assert torch.cuda.is_available(), 'ROCm GPU is unavailable'; p=torch.cuda.get_device_properties(0); print({'torch': torch.__version__, 'hip': torch.version.hip, 'devices': torch.cuda.device_count(), 'name': torch.cuda.get_device_name(0), 'architecture': getattr(p, 'gcnArchName', None)}); x=torch.randn((64, 64), device='cuda'); print(x @ x, x.device); torch.cuda.synchronize()"
python -m torch.utils.collect_env
```

The tensor operation must pass. If it does not, record the output as an
unavailable ROCm check and keep ShotSieve on CPU. A `torch.cuda.is_available()`
result without a HIP build or supported AMD device is not ROCm evidence.

## Run ShotSieve model evidence

Use `rocm` for an explicit provider request; `amd` remains an alias. Keep a
fresh cache and disposable data directory for each model. Run the online
process first, then repeat in a new process with the complete cache and
network disabled:

```bash
python scripts/model_smoke.py \
  --model topiq_nr --device rocm \
  --cache-dir ./build/rocm-evidence-cache/topiq_nr \
  --data-dir ./build/rocm-evidence-data/topiq_nr \
  --driver-version '<AMD driver version>' \
  --report-path ./build/audit-reports/rocm-topiq_nr-online.json

python scripts/model_smoke.py \
  --model topiq_nr --device rocm --offline \
  --cache-dir ./build/rocm-evidence-cache/topiq_nr \
  --data-dir ./build/rocm-evidence-data/topiq_nr \
  --driver-version '<AMD driver version>' \
  --report-path ./build/audit-reports/rocm-topiq_nr-offline.json
```

Repeat both commands with `clipiqa`. If Q-ReAlign Mini is included in the
target claim, repeat them with `qrealign-mini`. The report records the raw and
normalized score, model revision, resolved dependency versions, cache paths,
logical runtime, HIP/ROCm version, GPU name/architecture, elapsed time, peak
memory, and supplied driver version. Never share source photos, weights, or
private paths in an evidence report.

Also run one separate `--device cpu` smoke when documenting fallback behavior.
An unavailable explicit ROCm request must remain a failure; do not rerun it as
CPU and count that as an accelerator pass. Successful source-install smokes do
not certify every packaged host. Q-ReAlign Mini and any Windows AMD claim
require their own model/host evidence before a user-facing hardware claim is
expanded.
