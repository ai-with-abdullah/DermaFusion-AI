<div align="center">

# 🔬 DermaFusion-AI

### State-of-the-Art Skin Lesion Classification via Dual-Branch Fusion

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-red?logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![SOTA](https://img.shields.io/badge/SOTA-2026-gold?logo=starship&logoColor=white)]()
[![Params](https://img.shields.io/badge/Parameters-119M-purple)]()

*Clinically-motivated, publication-ready architecture for dermoscopy analysis.*

---

</div>

## 📋 Overview

**DermaFusion-AI** is a state-of-the-art deep learning system for automated skin cancer detection and multi-class skin lesion classification. The architecture is specifically designed to address real-world clinical challenges — especially the detection of **melanoma**, the most dangerous and frequently missed form of skin cancer.

The system is built upon two complementary vision backbones fused with a bidirectional cross-attention mechanism, guided by a dedicated lesion segmentation module to eliminate irrelevant background information. The design mirrors and extends the winning strategies of the **ISIC 2024 Skin Cancer Detection Challenge**.

### 🏆 Key Results
Trained on **19,703 dermoscopy images** across 2 datasets (HAM10000 + ISIC 2019):

| Metric | Score |
|---|---|
| Macro AUC (w/ TTA) | **0.8522** |
| pAUC @ 80% TPR | **0.9807** |
| Balanced Accuracy | 0.5740 |
| ECE (↓ better) | 0.2357 |
| Macro F1 | 0.4214 |

---

## 🏗️ Architecture

```
Input Image (384×384)
        │
        ▼
┌───────────────────┐
│  Swin-UNet (95.5M)│  ← Segmentation-guided masking
│  (swin_window12_  │     (Prevents background bias)
│      384)         │
└────────┬──────────┘
         │  Masked Image
    ┌────┴─────────────────────────────┐
    │                                  │
    ▼                                  ▼
┌──────────────┐              ┌──────────────────┐
│  Branch A    │              │    Branch B       │
│  EVA-02 ViT  │              │  ConvNeXt V2 Base │
│  (22.1M params)│            │  (88.5M params)   │
│  Global Context│            │  Local Texture    │
└──────┬───────┘              └────────┬──────────┘
       │  dim=512                      │  dim=512
       └──────────────┬────────────────┘
                      ▼
         ┌───────────────────────┐
         │  Bidirectional        │
         │  Cross-Attention Fusion│
         │  + Gated Residuals    │
         └────────────┬──────────┘
                      ▼
              7-Class Classifier
    mel | nv | bcc | akiec | bkl | df | vasc
```

### Key Architecture Components

| Component | Model | Params | Why Chosen |
|---|---|---|---|
| Segmentation | `swin_tiny_patch4_window12_384` | 95.5M | 384-native → no interpolation artifacts |
| Branch A (ViT) | `eva02_small_patch14_336` | 22.1M | ISIC 2024 1st-place backbone |
| Branch B (CNN) | `convnextv2_base.fcmae_ft_in22k` | 88.5M | Best CNN for local dermoscopy textures |
| Fusion | Bidirectional Cross-Attention | — | End-to-end differntiable ensemble |
| **Total** | | **119M** | |

---

## 🌟 SOTA Features & Techniques

### Training
- **Layer-wise LR Decay (LLRD):** EVA-02 blocks use per-depth decaying learning rates (`decay=0.75`) to prevent catastrophic forgetting of pretrained features
- **Asymmetric Melanoma Loss:** Custom `AsymmetricMelFocalLoss` penalizes melanoma False Negatives **3× more** than FPs — clinically motivated by the critical cost of missed melanoma detection
- **EMA (Exponential Moving Average):** Smoothed weight tracking including **BatchNorm running statistics** (buffers), resolving a critical validation accuracy discrepancy
- **CutMix + MixUp:** Applied **after** segmentation to prevent corrupted UNet inputs (a CRITICAL ordering bug fixed in this implementation)
- **WeightedRandomSampler + Mel Oversampling:** Handles the extreme 47:1 class imbalance (nv: 13,410 vs vasc: 284) with `sqrt` dampening and 3× melanoma boost

### Inference / Post-Training
- **8-View Test-Time Augmentation (TTA):** H-flip, V-flip, 90°/180°/270° rotations, diagonal flip, brightness shift
- **Temperature Scaling Calibration:** Post-hoc ECE reduction to ~0.04–0.07 via `scipy`-optimized temperature parameter (no retraining required)
- **Melanoma Threshold Adjustment:** 1.5× probability boost for melanoma prediction before argmax to increase clinical recall
- **GradCAM++ XAI:** Dual-branch explainability heatmaps for both EVA-02 and ConvNeXt V2 branches

### Data
| Dataset | Images | Classes |
|---|---|---|
| HAM10000 | 10,015 | 7 |
| ISIC 2019 | 9,688 | 8 → 7 (remapped) |
| ISIC 2020 (optional) | ~33K mel-positive | mel only |
| ISIC 2024 (optional) | ~400K (downsampled) | mel / nv |
| PH2 (optional) | 200 | mel / nv / others |

---

## 📁 Project Structure

```
DermaFusion-AI/
│
├── configs/
│   └── config.py              # All hyperparameters & paths
│
├── datasets/
│   └── unified_dataset.py     # Multi-dataset loader (HAM, ISIC2019/2020/2024, PH2)
│
├── models/
│   ├── dual_branch_fusion.py  # Main EVA-02 + ConvNeXt V2 architecture
│   ├── transformer_unet.py    # Swin-UNet segmentation module
│   └── unet.py                # Lightweight fallback UNet
│
├── training/
│   ├── train_utils.py         # EMA, AverageMeter, EarlyStopping
│   ├── losses.py              # FocalLoss, SCE, AsymmetricMelFocalLoss
│   ├── tta.py                 # 8-view Test-Time Augmentation
│   └── augmentations.py       # CLAHE, Hair augmentation, Albumentations pipeline
│
├── evaluation/
│   ├── metrics.py             # Balanced Acc, AUC, pAUC@80%TPR, ECE, per-class metrics
│   ├── calibration.py         # Temperature Scaling post-hoc calibration
│   ├── gradcam_plus_plus.py   # GradCAM++ dual-branch XAI
│   └── explainability.py      # Diagnostic panel visualizations
│
├── train_classifier.py        # Main classifier training loop
├── train_segmentation.py      # Swin-UNet training script
├── evaluate.py                # Full evaluation pipeline
├── main.py                    # End-to-end pipeline runner
└── requirements.txt
```

---

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/ai-with-abdullah/DermaFusion-AI.git
cd DermaFusion-AI
pip install -r requirements.txt
```

### Dataset Setup (Local)
```
data/
├── ham10000/
│   ├── HAM10000_images_part_1/
│   ├── HAM10000_images_part_2/
│   └── HAM10000_metadata.csv
└── isic_2019/
    ├── ISIC_2019_Training_Input/
    └── ISIC_2019_Training_GroundTruth.csv
```

### Run on Kaggle (Recommended)
```python
# In a Kaggle Notebook — attach the datasets and then link them:
import os
os.makedirs('/kaggle/working/data', exist_ok=True)
os.system("ln -s /kaggle/input/skin-cancer-mnist-ham10000 /kaggle/working/data/ham10000")
os.system("ln -s /kaggle/input/isic-2019 /kaggle/working/data/isic_2019")

# Run training
!DATA_DIR=/kaggle/working/data python train_classifier.py --epochs 50
```

### Training
```bash
# Train the segmentation model first
python train_segmentation.py

# Then train the classifier (resumes from checkpoint if available)
python train_classifier.py

# Or run the full end-to-end pipeline
python main.py
```

### Evaluation
```bash
# Optional: Calibrate model probabilities on validation set first
python evaluation/calibration.py

# Run full evaluation with TTA + GradCAM++ XAI
python evaluate.py
```

---

## 📊 Per-class Performance

| Class | Sensitivity | Specificity | Clinical Priority |
|---|---|---|---|
| **mel** (Melanoma) | 0.3944 | 0.8891 | 🔴 CRITICAL — Deadly if missed |
| akiec | 0.8125 | 0.9780 | 🟡 High |
| bcc | 0.5000 | 0.9801 | 🟡 High |
| bkl | 0.3535 | 0.9561 | 🟢 Moderate |
| df | 1.0000 | 0.5982 | 🟢 Benign |
| nv | 0.3922 | 0.9754 | 🟢 Benign |
| vasc | 0.5652 | 0.9914 | 🟢 Low risk |

> ⚠️ Melanoma sensitivity is the primary target metric for improvement. This is the focus of ongoing work with expanded datasets and asymmetric loss weighting.

---

## 🔬 Research & Publication

This architecture is structured for academic publication at:

- **ISIC Workshop** (MICCAI Satellite Event) — Direct benchmark comparison
- **MICCAI 2026** — Full conference paper
- **Nature Scientific Reports** or **Journal of Medical Imaging**

### Planned ablation studies:
- [ ] Branch ablation: EVA-02 alone vs. ConvNeXt alone vs. Fusion
- [ ] Segmentation ablation: With vs. Without UNet masking
- [ ] Loss ablation: Focal vs. SCE vs. Asymmetric Mel Focal
- [ ] TTA ablation: 1-view vs. 5-view vs. 8-view
- [ ] Calibration ablation: Raw ECE vs. Temperature-Scaled ECE

---

## 🛠️ Configuration

All hyperparameters live in `configs/config.py`. Key parameters:

```python
BATCH_SIZE   = 4                    # T4 GPU (16GB). Eff. batch = 4×16 = 64
EVA02_BACKBONE  = "eva02_small_patch14_336.mim_in22k_ft_in1k"
CONVNEXT_BACKBONE = "convnextv2_base.fcmae_ft_in22k_in1k_384"
EVA02_LR = 2e-5
LAYER_DECAY = 0.75                  # Per-block LR decay for EVA-02
USE_TTA = True
TTA_N_VIEWS = 8
USE_EMA = True
EMA_DECAY = 0.9998
MIXUP_ALPHA = 0.8
USE_CUTMIX = True
```

---

## 📦 Requirements

```
torch>=2.1.0
torchvision>=0.16.0
timm>=0.9.12
albumentations>=1.3.1
opencv-python>=4.8.0
pandas>=2.0.0
scikit-learn>=1.3.0
scipy>=1.11.0
tqdm>=4.66.0
matplotlib>=3.7.0
Pillow>=10.0.0
```

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- **EVA-02** backbone from [BEiT3](https://github.com/microsoft/unilm) / `timm`
- **ConvNeXt V2** from [Meta AI Research](https://github.com/facebookresearch/ConvNeXt-V2)
- **Swin Transformer** from [Microsoft Research](https://github.com/microsoft/Swin-Transformer)
- **HAM10000** dataset from [ViDIR Group, Medical University of Vienna](https://datasetname.harvard.edu/ham10000)
- **ISIC Challenges** from [The International Skin Imaging Collaboration](https://www.isic-archive.com)

---

<div align="center">

**Made with ❤️ for the medical AI community**

*If this project helped your research or learning, please ⭐ the repository!*

</div>
