# AMD ROCm install and release track

ShotSieve can use native AMD ROCm through the logical `rocm` runtime target
(`amd` is its vendor alias). Windows and Linux runtime packs include this
track, and the same pinned instructions can be used for source installs.

The commands below follow AMD's currently documented ROCm 7.2.1 Radeon
PyTorch pair: PyTorch 2.9.1, Python 3.12, and the AMD-published wheels. Use a
fresh environment and do not mix these wheels with the common CPU/CUDA/MPS or
Intel XPU Torch environments. The exact GPU, OS, driver, Python, Torch, and
ROCm combination must appear in AMD's current compatibility matrix before a
run is treated as evidence.

## Supported boundary and prerequisites

- **Linux first:** use a Radeon GPU listed in AMD's current Linux matrix, such
  as the supported RX 7000/9000 and Radeon PRO entries for the selected ROCm
  release, on an AMD-supported Linux distribution and kernel.
- **Windows is optional and narrower:** the documented 7.2.1 PyTorch path is
  for explicitly listed Radeon/Ryzen hardware, Python 3.12, and the required
  AMD graphics driver. The full ROCm stack is not supported on Windows; only
  the PyTorch path is covered here.
- Keep enough system memory and local disk for the model caches. Put the
  disposable data and cache roots outside the repository when possible.
- Record the GPU marketing name, `gfx` architecture, OS/kernel, Python,
  PyTorch wheel, ROCm/HIP version, AMD driver, cache paths, and `pip check`
  output for every validation host.

Official references:

- [AMD Radeon Linux compatibility matrix](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/native_linux/native_linux_compatibility.html)
- [AMD Radeon Windows compatibility matrix](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/windows/windows_compatibility.html)
- [AMD Linux ROCm installation](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/native_linux/install-radeon.html)
- [AMD Linux PyTorch wheels](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/native_linux/install-pytorch.html)
- [AMD Windows PyTorch wheels](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/windows/install-pytorch.html)
- [ROCm license and disclaimers](https://rocm.docs.amd.com/en/latest/about/license.html)

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
  "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torchvision-0.24.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl"
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
