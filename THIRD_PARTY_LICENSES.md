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
audited before publication. The current tested model integration selection is
`pyiqa==0.1.16`, `timm==1.0.29`, `huggingface-hub==1.31.0`,
`transformers==5.17.0`, and `openai-clip==1.0.1`; supported targets use
`torch==2.14.0` and `torchvision==0.29.0`. The retired DirectML package and
target are not part of the supported dependency or release matrix. The
source-only AMD track uses AMD-published ROCm 7.2.1 Torch wheels documented in
`docs/amd-rocm.md`; those wheels and the ROCm runtime are not bundled in a
release pack and must be audited under AMD's applicable terms.

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

## Q-Align (`qalign`)

- **Paper:** [Q-Align: Teaching LMMs for Visual Scoring via Discrete Text-Defined Levels](https://arxiv.org/abs/2312.17091)
- **Implementation:** pyiqa's Q-Align integration, which downloads the Q-Future `one-align` model from Hugging Face
- **Repository:** [Q-Future/Q-Align](https://github.com/Q-Future/Q-Align)
- **Terms:** subject to the licenses of the Q-Align implementation, its model card, and the underlying mPLUG-Owl2/LLaMA base models. Users must treat the downloaded weights as research-use only unless the upstream terms clearly grant broader rights.
- **ShotSieve runtime boundary:** supported for CUDA and Apple MPS only; not supported on CPU or XPU. Q-Align weights are never bundled and are downloaded only during explicit model preparation or scoring.
- **Authors:** Haoning Wu et al.

## AMD ROCm source track

- **Pinned validation family:** AMD ROCm 7.2.1 PyTorch wheels, Python 3.12;
  see `scripts/source-constraints-rocm.txt` and `docs/amd-rocm.md`
- **Distribution:** AMD-published wheels and system ROCm/AMDGPU components;
  not included in ShotSieve runtime packs
- **Terms:** review the [ROCm license and disclaimers](https://rocm.docs.amd.com/en/latest/about/license.html), AMD driver terms, and the exact wheel metadata before redistribution or commercial use
- **Support boundary:** Linux-first, exact GPU/OS/driver/Python matrix only;
  Windows PyTorch support is optional and narrower than the Linux stack

---

## Release audit

The source project intentionally does not claim that a lower-bound transitive
dependency set is an exact license inventory. For each release target, record
the resolved versions and license metadata from the isolated build environment,
inspect all license/notice files in the staged bundle, and verify that no
weights are present. A minimal metadata check is:

```bash
python -m pip show pyiqa torch torchvision timm huggingface-hub transformers Pillow numpy
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
| Q-Align assets | Downloaded on demand under upstream model/base-model terms | Verify research/commercial rights before use |
