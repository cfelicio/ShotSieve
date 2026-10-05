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

## Windows MIOpen BatchNorm workaround

Some Windows ROCm runtimes fail while MIOpen JIT-compiles BatchNorm, reporting
`MIOpenBatchNormFwdInferSpatial.cpp`, `HIPRTC_ERROR_COMPILATION`, and a missing
`type_traits` header. This is a runtime compiler failure, not a bad photo.

For Windows ROCm only, ShotSieve wraps TOPIQ and CLIPIQA's BatchNorm forwards
with `torch.backends.cudnn.enabled = False` and restores the previous value
in `finally`. PyTorch uses this flag for
[MIOpen BatchNorm dispatch](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/Normalization.cpp);
disabling it selects native HIP BatchNorm. The original forward, parameters,
running statistics, and subclass behavior are preserved. CLIPIQA's
[unregistered CLIP backbone](https://github.com/chaofengc/IQA-PyTorch/blob/v0.1.16/pyiqa/archs/clipiqa_arch.py)
is included in the wrapper traversal.

MIOpen remains enabled for convolutions. ROCm acceleration remains active and
no Visual Studio Build Tools, SDK, or external compiler installation is
required by this workaround. Native BatchNorm can have different throughput;
the impact depends on the GPU and model and has not been benchmarked. Because
the dispatch flag is process-wide, Windows ROCm model forwards share a lock
to keep concurrent scoring from observing the temporary change. Q-ReAlign
receives no BatchNorm wrapper. BatchNorm dispatch on other runtimes and Linux
ROCm is unchanged.

Unmistakable HIPRTC compilation errors and MIOpen code-object compilation
failures abort the scoring operation immediately, including when first
encountered during an individual-image fallback. Generic
`miopenStatusUnknownError`, out-of-memory, and image-specific errors retain
the existing individual retry behavior.

Mocked regression tests verify dispatch scoping and retries without an AMD
GPU. Full TOPIQ/CLIPIQA inference with disabled MIOpen must still be qualified
on a reproducing Windows ROCm machine using the model smoke commands above;
source inspection and mocked tests do not establish hardware success.
