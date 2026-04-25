# Third-Party Licenses

ShotSieve is licensed under the [GNU Affero General Public License v3.0 or later](LICENSE).

ShotSieve depends on third-party libraries and AI model weights that are
distributed under their own licenses. **These licenses restrict usage to
non-commercial purposes.** By using ShotSieve with learned IQA scoring, you
agree to comply with the terms of each upstream license listed below.

No model weights are bundled with ShotSieve. Weights are downloaded on
first use from Hugging Face and cached locally in `~/.cache/torch/hub/pyiqa/`.

---

## pyiqa (IQA-PyTorch)

- **Repository:** https://github.com/chaofengc/IQA-PyTorch
- **License:** NTU S-Lab License 1.0 + Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
- **Usage:** Non-commercial only. Attribution required. Derivative works must use the same license.
- **Authors:** Chaofeng Chen et al.

## TOPIQ (topiq_nr)

- **Paper:** "TOPIQ: A Top-down Approach from Semantics to Distortions for Image Quality Assessment" (2024)
- **Repository:** https://github.com/chaofengc/IQA-PyTorch
- **License:** NTU S-Lab License 1.0 / CC BY-NC-SA 4.0 (via pyiqa)
- **Usage:** Non-commercial only.
- **Authors:** Chaofeng Chen, Jiadi Mo, Jingwen Hou, Haoning Wu, Liang Liao, Wenxiu Sun, Qiong Yan, Weisi Lin

## CLIP-IQA (clipiqa)

- **Paper:** "Exploring CLIP for Assessing the Look and Feel of Images" (AAAI 2023)
- **Repository:** https://github.com/IceClear/CLIP-IQA
- **License:** NTU S-Lab License 1.0
- **Usage:** Non-commercial only. Contact authors for commercial use.
- **Authors:** Jianyi Wang, Kelvin C.K. Chan, Chen Change Loy

## Q-Align (qalign)

- **Paper:** "Q-Align: Teaching LMMs for Visual Scoring via Discrete Text-Defined Levels" (ICML 2024)
- **Repository:** https://github.com/Q-Future/Q-Align
- **License:** Subject to the licenses of the underlying base models (LLaVA). No explicit open-source license in the Q-Align repository. Users should treat the model weights as research-use only unless the authors clarify otherwise.
- **Authors:** Haoning Wu, Zicheng Zhang, Weixia Zhang, Chaofeng Chen, Liang Liao, Chunyi Li, Yixuan Gao, Annan Wang, Erli Zhang, Wenxiu Sun, Qiong Yan, Xiongkuo Min, Guangtao Zhai, Weisi Lin

---

## Summary

| Component | License | Commercial use |
|---|---|---|
| ShotSieve | AGPL-3.0-or-later | Allowed (copyleft) |
| pyiqa | NTU S-Lab + CC BY-NC-SA 4.0 | Non-commercial only |
| TOPIQ weights | NTU S-Lab + CC BY-NC-SA 4.0 | Non-commercial only |
| CLIP-IQA weights | NTU S-Lab License 1.0 | Non-commercial only |
| Q-Align weights | No explicit OSS license | Research use only |

If you intend to use ShotSieve or its AI scoring components in a commercial
context, you must obtain separate commercial licenses from the respective
authors of pyiqa and each model listed above.
