<div align="center">

# 🔬 DermaFusion-AI

### Dual-Branch Fusion Architecture for Skin Lesion Classification

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-red?logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![SOTA](https://img.shields.io/badge/SOTA-2026-gold?logo=starship&logoColor=white)]()
[![Params](https://img.shields.io/badge/Parameters-401.6M-purple)]()

*Clinically-motivated, publication-ready architecture for dermoscopy analysis.*

**Research Paper →** [RESEARCH_PAPER.md](RESEARCH_PAPER.md)

---

</div>

## 📋 Overview

**DermaFusion-AI** is a state-of-the-art deep learning system for automated skin cancer detection and 7-class skin lesion classification. It fuses two complementary vision backbones — **EVA-02 Large** (Vision Transformer) and **ConvNeXt V2 Base** (CNN) — via a cross-attention fusion mechanism, guided by a dedicated Swin-UNet lesion segmentation module. The architecture is trained on a unified multi-dataset pipeline (HAM10000 + ISIC 2020 + ISIC 2024) and evaluated with Test-Time Augmentation (TTA, N=5 views).

### 🏆 Key Results (EVA-02 Large — Final Model)

> Evaluated on a patient-aware held-out test set with TTA (N=5 views), ISIC 2024 downsampled at 50:1 neg:pos ratio.

| Metric | Score |
|---|---|
| **Macro AUC** | **0.9908** |
| **pAUC @ 80% TPR** | **1.0000** ✅ |
| **Balanced Accuracy** | **0.8559** |
| **Melanoma Sensitivity** | **0.9216** |
| **ECE (↓ better)** | **0.0662** |
| Accuracy | 0.8882 |
| Macro F1 | 0.7991 |
| Weighted F1 | 0.9007 |

---

## 📈 Model Evolution — Ablation

| Configuration | Datasets | Params | Val BalAcc | Test AUC | Mel Sensitivity |
|---|---|---|---|---|---|
| EVA-02 Small (baseline) | HAM10000 + ISIC 2019 + 2020 | ~22M | ~0.72–0.74 | ~0.94–0.96 | ~0.84–0.88 |
| **EVA-02 Large + ConvNeXt V2 (final)** | **HAM10000 + ISIC 2020 + 2024** | **401.6M** | **0.775** | **0.9908** | **0.9216** |

The jump from Small to Large model, combined with ISIC 2024 data (+400K images), accounts for the significant improvement in melanoma sensitivity (+4–8%) and AUC (+3–5%).

---

## 🏗️ Architecture

```
Input Image (448×448)
        │
   ┌────┴────┐
   │ SwinUNet│  ← Segmentation pre-processor (95.5M params, Swin-Tiny encoder)
   └────┬────┘
        │ Soft lesion mask applied to image
        │
   ┌────┴─────────────────────────────┐
   │                                  │
   ▼                                  ▼
┌──────────────────┐       ┌────────────────────┐
│  Branch A        │       │  Branch B           │
│  EVA-02 Large    │       │  ConvNeXt V2 Base   │
│  ~307M params    │       │  88.5M params       │
│  Original image  │       │  Segmented image    │
│  Global context  │       │  Local textures     │
└──────┬───────────┘       └──────────┬──────────┘
       │  dim → 512                   │  dim → 512
       └──────────────┬───────────────┘
                      ▼
         ┌─────────────────────────┐
         │  Cross-Attention Fusion │
         │  (512-dim, 8 heads)     │
         └────────────┬────────────┘
                      ▼
             7-Class Classifier
   mel | nv | bcc | akiec | bkl | df | vasc
```

### Components

| Component | Model | Pre-trained | Params |
|---|---|---|---|
| Segmentation | `swin_tiny_patch4_window7_224` | ✅ ImageNet | 95.5M |
| Branch A (ViT) | `eva02_large_patch14_448.mim_in22k_ft_in22k_in1k` | ✅ ImageNet-22K + 1K | ~307M |
| Branch B (CNN) | `convnextv2_base.fcmae_ft_in22k_in1k_384` | ✅ ImageNet-22K + 1K | 88.5M |
| Fusion | Cross-Attention (512-dim) | — | ~6M |
| **Total** | | | **401.6M** |

---

## 📊 Datasets

### Final Model Training Data

| Dataset | Records | Classes | Notes |
|---|---|---|---|
| HAM10000 | 10,015 | 7 | Standard dermoscopic dataset |
| ISIC 2020 | 584 (mel+ only) | mel | Confirmed positives; negatives excluded (label noise) |
| ISIC 2024 | 401,059 | mel / nv | 3D Total Body Photography; 10:1 train downsampling |
| **Total** | **411,658** | **7** | Patient-aware train/val/test split |

### Baseline (EVA-02 Small) Training Data
HAM10000 + ISIC 2019 (partial) + ISIC 2020 — no ISIC 2024.

### Test Set Evaluation

| Split | HAM10000 | ISIC 2020 | ISIC 2024 |
|---|---|---|---|
| Train | ~80% | ~80% | 10:1 neg:pos |
| Val | ~10% | ~10% | 20:1 neg:pos |
| **Test** | **~10%** | **~10%** | **50:1 neg:pos** |

All splits are **patient-aware** — no patient's images appear in both training and test.

---

## 📉 Per-class Performance (Final Model)

| Class | F1 | Sensitivity | Specificity | Clinical Priority |
|---|---|---|---|---|
| **mel** (Melanoma) | 0.6281 | **0.9216** | 0.9063 | 🔴 CRITICAL |
| nv (Nevus) | 0.9398 | 0.8938 | 0.9603 | 🟢 Benign |
| bcc | 0.8403 | 0.8475 | 0.9972 | 🟡 High |
| bkl | 0.7882 | 0.8033 | 0.9875 | 🟡 Moderate |
| df | 0.8649 | 0.8000 | 0.9997 | 🟢 Low risk |
| akiec | 0.7629 | 0.7255 | 0.9975 | 🟡 High |
| vasc | 0.7692 | **1.0000** | 0.9983 | 🟡 Moderate |

> Melanoma F1 is lower because the model intentionally accepts more false positives (lower precision) to achieve high recall — missing melanoma is far more dangerous than a false alarm.

---

## 🌟 Training Strategy

| Technique | Details |
|---|---|
| **Loss** | LabelSmoothingFocalLoss (α=0.25, γ=2) + SymmetricCrossEntropy |
| **Optimizer** | AdamW, weight decay=0.05 |
| **LR Schedule** | 7-epoch linear warmup → cosine annealing |
| **LLRD** | EVA-02 layer decay=0.75 |
| **Effective batch** | 64 (batch=2 × gradient accumulation=32) |
| **EMA** | decay=0.9998, stored on CPU (saves 1.6GB GPU memory) |
| **Augmentation** | MixUp (α=0.8), CutMix (α=1.0), RandomResizedCrop, Flips, ColorJitter |
| **Class imbalance** | WeightedRandomSampler + ISIC 2024 downsampling + Focal Loss |
| **TTA at eval** | N=5 views, post-softmax probability averaging |
| **Gradient checkpointing** | Enabled on both backbones (saves ~3GB activation memory) |

---

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/ai-with-abdullah/DermaFusion-AI.git
cd DermaFusion-AI
pip install -r requirements.txt
```

### Kaggle (Recommended — 2× T4 GPU)
```python
# Step 1: Attach these datasets in Kaggle:
#   - skin-cancer-mnist-ham10000
#   - siim-isic-melanoma-classification (ISIC 2020)
#   - isic-2024-challenge (ISIC 2024)

# Step 2: Link datasets
import os, shutil
data_dir = '/kaggle/working/DermaFusion-AI/data'
os.makedirs(data_dir, exist_ok=True)
os.symlink('/kaggle/input/skin-cancer-mnist-ham10000', f'{data_dir}/ham10000')
os.symlink('/kaggle/input/siim-isic-melanoma-classification', f'{data_dir}/isic_2020')
os.symlink('/kaggle/input/isic-2024-challenge', f'{data_dir}/isic_2024')

# Step 3: Train segmentation
!PYTHONPATH=. python train_segmentation.py

# Step 4: Train classifier (auto-resumes from checkpoint)
!PYTHONPATH=. python train_classifier.py

# Step 5: Evaluate
!PYTHONPATH=. python evaluate.py
```

### Local Training
```bash
python train_segmentation.py   # Train UNet first
python train_classifier.py     # Train dual-branch classifier
python evaluate.py             # Run full evaluation with TTA + GradCAM++
```

---

## 📁 Project Structure

```
DermaFusion-AI/
├── configs/
│   └── config.py                  # All hyperparameters and paths
├── datasets/
│   └── unified_dataset.py         # Multi-dataset unified loader
├── models/
│   ├── dual_branch_fusion.py      # Main architecture (EVA-02 + ConvNeXt V2)
│   ├── transformer_unet.py        # Swin-UNet segmentation module
│   ├── sota25_backbone.py         # EVA-02 backbone wrapper
│   └── convnextv3_backbone.py     # ConvNeXt V2 backbone wrapper
├── training/
│   ├── train_utils.py             # EMA, AverageMeter, EarlyStopping
│   ├── losses.py                  # FocalLoss, SCE, CombinedClassLoss
│   └── tta.py                     # Test-Time Augmentation
├── evaluation/
│   ├── metrics.py                 # BalAcc, AUC, pAUC@80%TPR, ECE
│   ├── calibration.py             # Temperature Scaling calibration
│   ├── gradcam_plus_plus.py       # GradCAM++ for EVA-02 + ConvNeXt V2
│   └── explainability.py          # Diagnostic panel visualizations
├── train_classifier.py            # Main classifier training loop
├── train_segmentation.py          # Swin-UNet training script
├── evaluate.py                    # Full evaluation pipeline
├── RESEARCH_PAPER.md              # Full research paper with methods & results
└── requirements.txt
```

---

## 🔬 Explainability (XAI)

GradCAM++ visualizations are generated for both branches:

- **EVA-02 Attention Rollout:** Uses `attn_drop` hooks to capture post-softmax attention matrices from all 24 transformer blocks → aggregated via attention rollout (Abnar & Zuidema, 2020)
- **ConvNeXt GradCAM++:** Second-order gradient-weighted activations from the final Conv2d layer (Chattopadhay et al., 2018)
- **Fusion Attention Map:** Direct cross-attention weights from the fusion module — consistently the most spatially focused, correctly centering on lesion boundaries

---

## ⚙️ Key Configuration

```python
# configs/config.py (highlights)
BATCH_SIZE                 = 2        # + accumulation=32 → eff. batch 64
GRADIENT_ACCUMULATION_STEPS = 32
EVA02_BACKBONE             = "eva02_large_patch14_448.mim_in22k_ft_in22k_in1k"
CONVNEXT_BACKBONE          = "convnextv2_base.fcmae_ft_in22k_in1k_384"
USE_EMA                    = True
EMA_DECAY                  = 0.9998
EMA_DEVICE                 = 'cpu'    # EMA on CPU → saves 1.6GB GPU 0
GRADIENT_CHECKPOINTING     = True     # Saves ~3GB activation memory
USE_TTA                    = True
TTA_N_VIEWS                = 5
ISIC2024_NEG_TO_POS_RATIO  = 10       # Train downsampling
```

---

## 📦 Requirements

```
torch>=2.2.0
torchvision>=0.17.0
timm>=0.9.12
albumentations>=1.3.1
opencv-python>=4.8.0
pandas>=2.0.0
scikit-learn>=1.3.0
tqdm>=4.66.0
matplotlib>=3.7.0
Pillow>=10.0.0
```

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- **EVA-02** — [BAAI / timm](https://github.com/baaivision/EVA)
- **ConvNeXt V2** — [Meta AI Research](https://github.com/facebookresearch/ConvNeXt-V2)
- **Swin Transformer** — [Microsoft Research](https://github.com/microsoft/Swin-Transformer)
- **HAM10000** — [ViDIR Group, Medical University of Vienna](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)
- **ISIC Archive** — [International Skin Imaging Collaboration](https://www.isic-archive.com)

---

<div align="center">

**Made with ❤️ for the medical AI and research community**

*For the full methodology, results, and literature comparison — read the [Research Paper](RESEARCH_PAPER.md)*

*If this project helped your research or learning, please ⭐ the repository!*

</div>
