<div align="center">

# DermaFusion-AI

### A dual-branch Vision Transformer + ConvNeXt pipeline for multi-source dermoscopic skin cancer classification

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Muhammad Abdullah**<sup>1</sup> · **Amna Shifa**<sup>1,2</sup>

<sup>1</sup> Department of Artificial Intelligence, The Islamia University of Bahawalpur, Pakistan
<sup>2</sup> University of Galway, Ireland

</div>

---

## Overview

**DermaFusion-AI** is a two-stage deep-learning system for seven-class dermoscopic skin-lesion classification, trained on a de-duplicated, patient-level split of four public archives. A Swin-Transformer U-Net first produces a soft lesion mask; a dual-branch classifier then combines an **EVA-02 Large Vision Transformer** (global context, from the original image) with a **ConvNeXt V2 Base** network (local texture, from the masked image).

Beyond the classifier, the project is a **controlled study of class imbalance in multi-source dermoscopy**. Its central, reproducible result is that **logit adjustment, the standard prescription for class imbalance, is counter-productive for rare lesion classes, while a simple decoupled rebalancing preserves them.** The repository also reports honest negative results: elaborate cross-attention fusion does not outperform a linear probe on frozen features, so representation quality, not fusion complexity, drives performance here.

---

## Key results

Deployed model on the held-out, patient-level test set (seven classes, five-view test-time augmentation):

| Metric | Value |
|---|---|
| Accuracy | 0.866 |
| Balanced accuracy | 0.857 |
| Macro-F1 | 0.833 |
| **Macro-AUC** | **0.983** |
| Expected Calibration Error (↓) | 0.067 |
| Partial AUC @ 80% TPR (melanoma) | 0.896 |
| Melanoma sensitivity (default / screening point) | 0.859 / 0.939 |
| Segmentation Dice (Swin-U-Net) | 0.949 |

### Per-class results

| Class | F1 | Sensitivity | Specificity |
|---|---|---|---|
| akiec | 0.768 | 0.773 | 0.988 |
| bcc | 0.910 | 0.952 | 0.981 |
| bkl | 0.780 | 0.733 | 0.984 |
| df | 0.806 | 0.879 | 0.998 |
| mel | 0.781 | 0.859 | 0.925 |
| nv | 0.914 | 0.882 | 0.943 |
| vasc | 0.873 | 0.923 | 0.999 |

### The central finding: imbalance handling

Frozen-feature study, rare-class mean F1 (mean ± s.d. over 5 seeds):

| Strategy | Rare-mean F1 |
|---|---|
| Class-weighted cross-entropy | 0.803 ± 0.043 |
| Per-source logit adjustment (SALA) | 0.706 ± 0.072 |
| Global logit adjustment | 0.678 ± 0.054 |
| **Decoupled rebalancing (ours)** | **0.813 ± 0.030** |

Both forms of logit adjustment substantially reduce rare-class F1; decoupled class-weighted rebalancing preserves it.

---

## Architecture

```
Input dermoscopy image (448x448)
        |
        v
  Swin-Transformer U-Net  -->  soft lesion mask  -->  masked image
  (learnable Tversky loss)
        |                                               |
   original image                                  masked image
        |                                               |
        v                                               v
  EVA-02 Large ViT                               ConvNeXt V2 Base
  (global context)                               (local texture)
        |                                               |
        +----------------->  Feature fusion  <----------+
                                  |
                                  v
                     Classification head (softmax)
                                  |
                                  v
        7 classes: akiec, bcc, bkl, df, mel, nv, vasc
```

| Component | Architecture | Parameters |
|---|---|---|
| Swin-Transformer U-Net (segmentation) | Hierarchical ViT + U-Net decoder | 95.5M |
| Branch A: EVA-02 Large | ViT-L, patch 14, 448x448 | 304.1M |
| Branch B: ConvNeXt V2 Base | Fully convolutional, 384x384 | 88.5M |
| Fusion + classification head | Feature fusion | ~9M |

- **Stage 1, segmentation:** Swin-Transformer U-Net trained with a *learnable* Tversky loss, so the false-positive / false-negative trade-off is fitted to the data rather than fixed.
- **Stage 2, classification:** dual-branch EVA-02 + ConvNeXt V2, trained with a label-smoothed Focal loss combined with Symmetric Cross-Entropy; EMA weights and 5-view test-time augmentation at evaluation.
- **Imbalance study:** the backbone is frozen and only the classifier head is re-trained, so the rebalancing strategy is the only variable.

---

## Datasets

Four public dermoscopy sources are merged into a unified 7-class taxonomy, de-duplicated by image identity and split at the **patient level** (70/15/15) so that no patient appears in more than one partition.

| Source | Records (after de-dup) |
|---|---|
| HAM10000 | 10,015 |
| ISIC 2019 | 15,316 |
| ISIC 2020 | 584 |
| ISIC 2024 (SLICE-3D) | 401,059 |
| **Total** | **426,974** |

Splitting uses `GroupShuffleSplit` on patient IDs. All datasets are public and anonymised.

| Code | Full name | Type |
|---|---|---|
| `mel` | Melanoma | Malignant |
| `bcc` | Basal cell carcinoma | Malignant |
| `akiec` | Actinic keratosis / intraepithelial carcinoma | Pre-malignant |
| `bkl` | Benign keratosis-like lesions | Benign |
| `nv` | Melanocytic nevi | Benign |
| `df` | Dermatofibroma | Benign |
| `vasc` | Vascular lesions | Vascular |

---

## Repository structure

```
DermaFusion-AI/
├── configs/            # central configuration (config.py)
├── datasets/           # unified multi-source dataset, de-dup, patient split
├── models/             # Swin-U-Net, EVA-02 + ConvNeXt dual-branch, losses
├── training/           # training loops, EMA, schedulers, TTA
├── evaluation/         # metrics, calibration, decoupled study, thresholds,
│                       # ablations, Grad-CAM++, statistical tests
├── train_segmentation.py   # Stage 1: train the Swin-U-Net
├── train_classifier.py     # Stage 2: train the dual-branch classifier
├── evaluate.py             # full evaluation of the deployed model
├── app.py                  # local Gradio demo
├── train_on_kaggle.ipynb   # Kaggle training notebook
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/ai-with-abdullah/DermaFusion-AI.git
cd DermaFusion-AI
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Place the datasets under `data/` and adjust paths in `configs/config.py`.

### Pretrained weights

| Weight file | Description | Link |
|---|---|---|
| `best_dual_branch_fusion.pth` | Dual-branch classifier | [Kaggle](https://kaggle.com/datasets/aiwithAbdullah71/weights1) |
| `best_unet.pth` | Swin-U-Net segmentation | [Kaggle](https://kaggle.com/datasets/aiwithAbdullah71/weights) |

Place both files in `outputs/weights/` or update the paths in `configs/config.py`.

---

## Usage

Train the two stages, then evaluate:

```bash
# Stage 1 - segmentation
python train_segmentation.py

# Stage 2 - dual-branch classifier
python train_classifier.py

# Full evaluation of the deployed model
python evaluate.py
```

Reproduce the imbalance study and analyses (run as modules from the repository root):

```bash
PYTHONPATH=. python -m evaluation.decoupled_study        # 4-strategy imbalance comparison
PYTHONPATH=. python -m evaluation.optimize_thresholds    # per-class + melanoma-safety thresholds
PYTHONPATH=. python -m evaluation.run_novelty_ablation   # architecture ablation
```

Run the local demo (uploads a dermoscopy image and returns a 7-class prediction with a Grad-CAM++ heatmap):

```bash
python app.py
```

### Loss functions

- **Segmentation:** `0.5 x BCE + 0.5 x Learnable-Tversky`, where the Tversky (alpha, beta) are learned via a softmax parameter.
- **Classification:** `0.7 x Focal + 0.3 x Symmetric-CE`, with label smoothing (epsilon = 0.1), focusing gamma = 2, and class weighting. Symmetric Cross-Entropy adds robustness to label noise when merging heterogeneous datasets.

---

## Research paper

This repository accompanies the manuscript *"A dual-branch vision transformer and ConvNeXt pipeline for multi-source dermoscopic skin cancer classification"* (in preparation for **Scientific Reports**). The paper reports the full methodology, calibration analysis, a clinical melanoma-safety operating point, and the imbalance study summarised above.

---

## Limitations

- The two rarest classes (dermatofibroma, vascular lesion) have fewer than 260 training images each, which caps their absolute performance.
- De-duplication reduces, but cannot fully guarantee the absence of, cross-archive overlap.
- All training data are dermoscopic and public and skew toward lighter skin phototypes; external validation on smartphone (e.g. PAD-UFES-20) and diverse-skin cohorts, with a Fitzpatrick-stratified fairness analysis, is future work.
- The system is large (~500M parameters) and needs a GPU for practical inference; knowledge distillation to a compact model is planned.

---

## Citation

```bibtex
@article{abdullah_dermafusion,
  title   = {A dual-branch vision transformer and ConvNeXt pipeline for
             multi-source dermoscopic skin cancer classification},
  author  = {Abdullah, Muhammad and Shifa, Amna},
  note    = {Manuscript in preparation for Scientific Reports},
  year    = {2026},
  url     = {https://github.com/ai-with-abdullah/DermaFusion-AI}
}
```

---

## License and acknowledgements

Released under the [MIT License](LICENSE). We thank the International Skin Imaging Collaboration (ISIC) and the HAM10000 contributors for making the datasets publicly available.
