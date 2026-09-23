# Third-Party Licenses

ShotSieve is licensed under the [GNU Affero General Public License v3.0 or later](LICENSE).

The learned-IQA path is optional. ShotSieve does not bundle model weights or
private photographs. When a supported model is prepared, its upstream assets
may be retrieved into the configured caches. The caches are deliberately split:
`HF_HOME`/`HF_HUB_CACHE` are used for Hugging Face assets and `TORCH_HOME` is
used by Torch Hub and the OpenAI CLIP loader. A cache root supplied with
`--model-cache-dir` only supplies defaults for those locations; it does not
move existing caches.

The entries below identify the current supported product boundary. They are
not a blanket permission for every upstream asset: package code, checkpoint
files, and any base model can have different terms. The exact package versions
used by a release must be taken from that release's target constraints and
audited before publication. The current learned-IQA pins include
`pyiqa==0.1.16`, `facexlib==0.3.0`, `timm==1.0.30`,
`huggingface-hub==1.32.0`, `transformers==5.17.0`, `openai-clip==1.0.1`,
`accelerate==1.15.0`, `sentencepiece==0.2.2`, `einops==0.8.2`,
`icecream==2.2.0`, and `setuptools==81.0.0` to supply the `pkg_resources` API
used by OpenAI CLIP.
Supported CPU, CUDA, and MPS targets use `torch==2.14.0` and
`torchvision==0.29.0`; XPU uses its target-specific 2.14.0 pair; AMD ROCm uses
PyTorch 2.13.0 and TorchVision 0.28.0 with ROCm 10.0.0. The target-specific
selections are recorded in the release constraints.

---

## pyiqa (IQA-PyTorch)

- **Pinned project version:** `pyiqa==0.1.16` in the learned-IQA extras
- **Repository:** https://github.com/chaofengc/IQA-PyTorch
- **License:** [PolyForm Noncommercial License 1.0.0](https://github.com/chaofengc/IQA-PyTorch/blob/v0.1.16/LICENSE), with the repository's additional `LICENSE-S-Lab` notice file
- **Required notice:** retain the upstream license and notice files when distributing a bundle containing pyiqa code
- **Authors:** Chaofeng Chen et al.

The pyiqa license changed over time. Do not copy the terms from an older
release into a current package audit.

## TOPIQ (`topiq_nr`)

- **Paper:** [TOPIQ: A Top-down Approach from Semantics to Distortions for Image Quality Assessment](https://arxiv.org/abs/2308.03060)
- **Implementation:** pyiqa's `CFANet` configuration `cfanet_nr_koniq_res50`
- **Checkpoint identifier:** `cfanet_nr_koniq_res50-9a73138b.pth`
- **Upstream source:** [IQA-PyTorch-Weights](https://huggingface.co/chaofengc/IQA-PyTorch-Weights)
- **Terms:** the checkpoint and ResNet-50 semantic backbone must be audited under their own upstream terms in the exact release environment; they are not automatically licensed by ShotSieve or by the pyiqa package metadata

## CLIPIQA (`clipiqa`)

- **Paper:** [Exploring CLIP for Assessing the Look and Feel of Images](https://github.com/IceClear/CLIP-IQA)
- **Implementation:** pyiqa's plain `CLIPIQA` configuration, using the OpenAI CLIP `RN50` backbone and packaged prompt pairs
- **Checkpoint identifier:** `RN50.pt`
- **Reference URL and embedded SHA-256:** `https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt`
- **Terms:** review the [CLIP source license](https://github.com/openai/CLIP/blob/main/LICENSE) and the applicable checkpoint/model terms before redistribution or commercial use

The supported plain `clipiqa` model does not download a separate learned
CLIPIQA prompt checkpoint. The optional `clipiqa+` variants are outside the
ShotSieve product catalog.

## Q-ReAlign Mini (`qrealign-mini`)

- **Model card:** [Q-Future Q-ReAlign Mini 0.8B](https://huggingface.co/q-future/Q-ReAlign-Mini-0.8B)
- **Repository:** [Q-Future/Q-ReAlign](https://github.com/Q-Future/Q-ReAlign)
- **PyIQA metric:** `qrealign-mini` in `pyiqa==0.1.16`; ShotSieve exposes
  only the catalog name `qrealign-mini` for new runs
- **Checkpoint:** `q-future/Q-ReAlign-Mini-0.8B`, about 2.21 GB of safetensors
- **Pinned model revision:** `fe1f45a7574c9e9d908875af9f7e90cb946aa19f` (model-card commit)
- **Model-card license:** Apache-2.0, as declared by the checkpoint repository
- **Base model:** Qwen3.5-VL (`qwen3_5`); review the [Qwen3.5 source and license](https://github.com/QwenLM/Qwen3.5) and the exact base-model terms before redistribution or commercial use
- **Terms boundary:** the Q-ReAlign checkpoint, Qwen3.5-VL base/model terms, Q-ReAlign implementation, PyIQA code, and downloaded tokenizer/processor assets are separate review boundaries. ShotSieve does not bundle any of them.
- **Runtime boundary:** the initial product size is Mini only, with a ShotSieve batch maximum of four. CPU and accelerator execution are exposed through the model/runtime compatibility catalog, but each target requires fresh online/offline model-smoke evidence using the relevant runtime guide before it is claimed as validated.

## AMD ROCm 10 track

- **Pinned package family:** AMD ROCm 10.0.0, PyTorch 2.13.0, and TorchVision
  0.28.0 with `gfx1103` device packages; Python 3.14;
  see `scripts/source-constraints-rocm.txt` and `docs/amd-rocm.md`
- **Distribution:** AMD-published wheels and system ROCm/AMDGPU components. The
  Windows/Linux runtime packs advertise this track and install the matching
  wheels as target-specific sidecars; the wheels and system components are not
  inside the ShotSieve archive.
- **Terms:** review the [ROCm license and disclaimers](https://rocm.docs.amd.com/en/latest/about/license.html), AMD driver terms, and the exact wheel metadata before redistribution or commercial use
- **Support boundary:** use AMD's live GPU/OS/driver/Python compatibility
  matrix for the exact host; ShotSieve's selected device package is `gfx1103`

---

## Release audit

The source project intentionally does not claim that a lower-bound transitive
dependency set is an exact license inventory. For each release target, record
the resolved versions and license metadata from the isolated build environment,
inspect all license/notice files in the staged bundle, and verify that no
weights are present. A minimal metadata check is:

```bash
python -m pip show pyiqa facexlib torch torchvision timm huggingface-hub transformers openai-clip accelerate sentencepiece einops Pillow numpy
```

Handle missing optional packages explicitly; the command reports warnings for
packages not selected by a target. The audit must also review the
model-specific sources above, because package license metadata alone is
insufficient.

| Component | Current status | Commercial-use decision |
|---|---|---|
| ShotSieve | AGPL-3.0-or-later | Follow AGPL obligations |
| pyiqa 0.1.16 | PolyForm Noncommercial 1.0.0 plus included notices | Do not assume commercial permission |
| TOPIQ assets | Package, checkpoint, and backbone terms are separate | Verify each asset before use |
| CLIPIQA assets | pyiqa code plus OpenAI CLIP RN50 terms | Verify both code and checkpoint terms |
| Q-ReAlign Mini assets | Downloaded on demand under Q-ReAlign, Qwen3.5-VL, and processor/base-model terms | Verify exact checkpoint and base-model rights before use |
