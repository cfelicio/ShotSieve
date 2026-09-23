# AMD ROCm install and release track

ShotSieve's `rocm` runtime is AMD ROCm 10.0.0 on Windows and Linux. Its
release targets are `windows-amd-rocm` and `linux-amd-rocm`. Both use AMD's
stable multi-architecture index with PyTorch 2.13.0 and TorchVision 0.28.0;
the current target selects `gfx1103` kernels for Radeon 780M-class devices.
Python 3.14 is supported by AMD's ROCm 10 PyTorch packages and is used by the
release targets.

Check AMD's live compatibility matrix before installing. It lists the exact
GPU, operating system, driver, and framework combinations; a package build
does not establish that a specific machine is supported.

Official references:

- [AMD ROCm 10 PyTorch installation](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html)
- [AMD ROCm 10 compatibility matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html)
- [AMD ROCm 10 release notes](https://rocm.docs.amd.com/en/latest/about/release-notes.html)
- [ROCm license and disclaimers](https://rocm.docs.amd.com/en/latest/about/license.html)

## Install from source

Use a clean Python 3.14 environment. The exact package pair and AMD index are
also recorded in `scripts/source-constraints-rocm.txt`.

```powershell
py -3.14 -m venv .venv-rocm
.\.venv-rocm\Scripts\python.exe -m pip install --upgrade pip setuptools wheel packaging
.\.venv-rocm\Scripts\python.exe -m pip install `
  "rocm==10.0.0" `
  "torch[device-gfx1103]==2.13.0+rocm10.0.0" `
  "torchvision[device-gfx1103]==0.28.0+rocm10.0.0" `
  --index-url https://stable.repo.amd.com/rocm/whl-next/ `
  --extra-index-url https://pypi.org/simple `
  -c scripts/source-constraints-rocm.txt
.\.venv-rocm\Scripts\python.exe -m pip install -e ".[learned-iqa]" `
  -c scripts/release-constraints.txt -c scripts/source-constraints-rocm.txt
.\.venv-rocm\Scripts\python.exe -m pip check
```

Linux uses the same requirements and index in a fresh Python 3.14 environment
on an AMD-listed distribution. Set up the AMD driver and operating-system
prerequisites from the live ROCm guide before installing the Python packages.

## Verify the runtime and models

Run a synchronized tensor operation to confirm the native runtime. AMD ROCm
PyTorch exposes its GPU through the CUDA API while `torch.version.hip`
identifies the AMD build:

```bash
python -c "import torch; assert torch.version.hip; assert torch.cuda.is_available(); x=torch.randn((64, 64), device='cuda'); (x @ x).sum().item(); torch.cuda.synchronize(); print(torch.__version__, torch.version.hip, torch.cuda.get_device_name(0))"
python -m torch.utils.collect_env
```

Then qualify each model with a fresh cache, followed by a new offline process:

```bash
for model in topiq_nr clipiqa qrealign-mini; do
  python scripts/model_smoke.py --model "$model" --device rocm \
    --cache-dir "./build/rocm-model-cache/$model" \
    --data-dir "./build/rocm-model-data/$model" \
    --driver-version '<AMD driver version>'
  python scripts/model_smoke.py --model "$model" --device rocm --offline \
    --cache-dir "./build/rocm-model-cache/$model" \
    --data-dir "./build/rocm-model-data/$model" \
    --driver-version '<AMD driver version>'
done
```

The preview build checks dependency resolution and bundles on hosted runners.
It cannot verify a physical AMD GPU, driver, or model inference path; keep
hardware claims within the current AMD matrix and record native test evidence
for the exact system.
