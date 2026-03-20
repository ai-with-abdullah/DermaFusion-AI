<div align="center">

# DermaFusion-AI

### A Dual-Branch EVA-02 + ConvNeXt Fusion Architecture with Segmentation-Guided Attention  
### for Multi-Class Skin Lesion Classification

[![Paper](https://img.shields.io/badge/DermaFusion-AI.pdf-blue?logo=adobeacrobatreader)](./DermaFusion_AI_Paper.pdf)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Muhammad Abdullah**  
Department of Artificial Intelligence, The Islamia University of Bahawalpur, Pakistan  
*Supervisor: Dr. Amna Shifa*

</div>

---

## Overview

**DermaFusion-AI** is a research-grade skin cancer detection system that combines two complementary deep learning backbones — an EVA-02 Large Vision Transformer (global context) and a ConvNeXt V2 Base CNN (local texture) — fused through **bidirectional cross-attention**. A Swin-Transformer U-Net provides segmentation-guided masking so the CNN branch specialises entirely in lesion-internal texture patterns.

The model is trained on **460,000+ dermoscopy images** across four public ISIC datasets with patient-aware splitting to prevent data leakage, and evaluated on two held-out external datasets (PAD-UFES-20, DERM7PT) for cross-domain benchmarking.

> **Key Result:** AUC **0.9908** | MEL Sensitivity **92.2%** — surpassing the average dermatologist benchmark of 86%.

---

## Architecture

```
Input Dermoscopy Image
         │
         ▼
┌─────────────────────┐
│  Swin-Transformer   │  ← 95.5M params
│  U-Net Segmentation │    Produces binary lesion mask
└────────┬────────────┘
         │
   ┌─────┴────────────────────────────────────┐
   │                                           │
   ▼                                           ▼
Original Image (448×448)         Masked Image (384×384)
(lesion + background)            (lesion only, background=0)
   │                                           │
   ▼                                           ▼
┌──────────────────┐             ┌──────────────────────┐
│  Branch A:       │  307M       │  Branch B:           │  88.5M
│  EVA-02 Large    │  params     │  ConvNeXt V2 Base    │  params
│  Global Context  │             │  Local Texture       │
└────────┬─────────┘             └──────────┬───────────┘
         │                                  │
         └──────────────┬───────────────────┘
                        │
                        ▼
         ┌──────────────────────────────┐
         │  Bidirectional Cross-Attention│  8 heads, 512-dim
         │  + Gated Residual Fusion     │
         └──────────────┬───────────────┘
                        │
                        ▼
              7-Class Softmax Output
    {mel, nv, bcc, akiec, bkl, df, vasc}
```

| Component | Architecture | Parameters | GFLOPs |
|---|---|---|---|
| Branch A — EVA-02 Large | ViT-L, patch 14, 448×448 | 304.1M | 296.8 |
| Branch B — ConvNeXt V2 Base | Fully Convolutional, 384×384 | 88.5M | 59.1 |
| Fusion + Classifier Head | Bidirectional Cross-Attention | 9.0M | — |
| Swin-UNet Segmentation | Hierarchical ViT + UNet Decoder | 95.5M | 59.1 |
| **Total (End-to-End)** | — | **497.1M** | **415.0** |

---

## Results

### Primary Benchmark — ISIC Multi-Dataset Test Split (n=2,260)

| Metric | Value |
|---|---|
| Macro AUC | **0.9908** |
| Balanced Accuracy | **85.59%** |
| MEL Sensitivity | **92.16%** *(avg. dermatologist: 86%)* |
| Macro F1 | **0.7991** |
| Weighted F1 | **0.9007** |
| ECE (after temperature scaling) | **0.0390** *(T = 1.460)* |
| GPU Latency (T4) | **371 ms/image** |

### Per-Class Results

| Class | F1 | Sensitivity | Specificity | AUC |
|---|---|---|---|---|
| akiec | 0.7629 | 72.55% | 99.75% | 0.997 |
| bcc | 0.8403 | 84.75% | 99.72% | 0.999 |
| bkl | 0.7882 | 80.33% | 98.75% | 0.989 |
| df | 0.8649 | 80.00% | 99.97% | 0.999 |
| **mel** | **0.6281** | **92.16%** | 90.63% | 0.972 |
| nv | 0.9398 | 89.38% | 96.03% | 0.980 |
| vasc | 0.7692 | 100.0% | 99.83% | 1.000 |

### Cross-Domain Evaluation (External, Never Seen During Training)

| Dataset | Modality | AUC | Balanced Acc |
|---|---|---|---|
| PAD-UFES-20 (zero-shot) | Smartphone | 0.642 | 28.9% |
| PAD-UFES-20 (head fine-tuned) | Smartphone | **0.873** | **67.4%** |
| DERM7PT (original weights) | Dermoscope | **0.872** | 50.8% |

### Statistical Validation

| Test | Result |
|---|---|
| Bootstrap 95% CI on AUC (N=2000) | [0.9939, 0.9977] |
| McNemar's test vs EVA-02 alone | χ²=12.96, **p < 0.001** |
| 5-fold CV AUC (EVA-02 Small) | 0.9512 ± 0.0093 |

### Ablation Study — Backbone Scale

| Model | AUC | Balanced Acc | MEL Sensitivity |
|---|---|---|---|
| EVA-02 Small (22M) | 0.9839 | 77.71% | ~39% |
| **EVA-02 Large (307M) + ConvNeXt** | **0.9908** | **85.59%** | **92.16%** |
| **Gain** | **+0.69%** | **+7.88 pp** | **+53 pp** |

---

## Datasets

| Dataset | Images | Split | Role |
|---|---|---|---|
| HAM10000 | 10,015 | Patient-aware 80/10/10 | Training |
| ISIC 2019 | 25,331 | Patient-aware 80/10/10 | Training |
| ISIC 2024 SLICE-3D | ~401,059 | 50:1 neg:pos downsampled | Training |
| PAD-UFES-20 | 2,298 | Held-out | Cross-domain evaluation |
| DERM7PT | 1,011 | Held-out | Cross-domain evaluation |

All training splits use `GroupShuffleSplit` on patient IDs to prevent data leakage.  
Label scheme: `{akiec, bcc, bkl, df, mel, nv, vasc}` — HAM10000 7-class taxonomy.

---

## Project Structure

```
DermaFusion-AI/
│
├── configs/
│   └── config.py                  # All hyperparameters, paths, flags
│
├── models/
│   ├── dual_branch_fusion.py      # ⭐ Main DermaFusion-AI model
│   ├── transformer_unet.py        # Swin-UNet segmentation model
│   ├── sota25_backbone.py         # EVA-02 Large wrapper
│   └── convnextv3_backbone.py     # ConvNeXt V2 Base wrapper
│
├── datasets/
│   ├── unified_dataset.py         # Multi-dataset loader (HAM10000+ISIC)
│   └── augmentations.py           # CutMix, MixUp, TTA transforms
│
├── training/
│   ├── losses.py                  # Focal + SCE combined loss
│   ├── train_utils.py             # EMA, LR scheduler, gradient clipping
│   └── tta.py                     # Test-Time Augmentation
│
├── evaluation/
│   ├── metrics.py                 # AUC, BalAcc, F1, ECE computation
│   ├── calibration.py             # Temperature scaling
│   ├── explainability.py          # GradCAM++ pipeline
│   ├── gradcam_plus_plus.py       # GradCAM++ core implementation
│   ├── run_temperature_scaling.py # Post-hoc calibration
│   ├── run_confidence_intervals.py# Bootstrap 95% CIs
│   ├── run_statistical_tests.py   # McNemar's test
│   ├── run_inference_benchmark.py # GPU/CPU latency benchmarking
│   └── run_cross_validation.py    # 5-fold patient-aware CV
│
├── utils/
│   ├── logger.py                  # Structured logging
│   └── seed.py                    # Reproducibility seeding
│
├── outputs/                       # Generated results (gitignored)
│   ├── weights/                   # Saved model checkpoints
│   └── *.png / *.csv              # Figures and metrics outputs
│
├── huggingface_space/             # Hugging Face demo app
│   └── app.py                     # Gradio interface
│
├── train_classifier.py            # Main classifier training script
├── train_segmentation.py          # Swin-UNet training script
├── finetune_padufes.py            # PAD-UFES-20 head fine-tuning
├── evaluate.py                    # Full evaluation pipeline
├── test_both_weights.py           # Cross-domain test (DERM7PT)
├── test_padufes.py                # PAD-UFES-20 evaluation
├── app.py                         # Local Gradio demo
├── train_on_kaggle.ipynb          # Kaggle training notebook
├── train_on_colab.ipynb           # Google Colab training notebook
├── DermaFusion_AI_Paper.pdf       # Research paper (PDF)
├── requirements.txt               # Python dependencies
└── README.md
```

---

## Setup

### 1. Clone Repository
```bash
git clone https://github.com/ai-with-abdullah/DermaFusion-AI.git
cd DermaFusion-AI
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Weights

Model weights (~3.3 GB total) are hosted on Kaggle:

| Weight File | Description | Link |
|---|---|---|
| `best_dual_branch_fusion.pth` | DermaFusion-AI classifier (401M) | [Kaggle — weights1](https://kaggle.com/datasets/aiwithAbdullah71/weights1) |
| `best_unet.pth` | Swin-UNet segmentation (95.5M) | [Kaggle — weights](https://kaggle.com/datasets/aiwithAbdullah71/weights) |

Place both files in `outputs/weights/` or update paths in `configs/config.py`.

---

## Usage

### Run Local Demo (Gradio)
```bash
python app.py
```
Launches a web interface for uploading dermoscopy images and receiving 7-class predictions with confidence scores and GradCAM++ heatmaps.

> **Note:** Requires GPU for real-time inference (~371 ms/image on T4). CPU inference is supported but slow (~5 seconds/image).

### Train the Classifier
```bash
# Local (GPU required)
python train_classifier.py

# On Kaggle (recommended — free T4×2 GPU)
# Open train_on_kaggle.ipynb and run all cells

# On Google Colab
# Open train_on_colab.ipynb and run all cells
```

### Train the Swin-UNet Segmentation Model
```bash
python train_segmentation.py
```

### Evaluate on ISIC Test Set
```bash
python evaluate.py
```

### Cross-Domain Evaluation
```bash
# PAD-UFES-20 (smartphone images)
python test_padufes.py

# DERM7PT (dermoscopy + smartphone)
python test_both_weights.py
```

### Run Full Evaluation Suite
```bash
# Temperature scaling calibration
python -m evaluation.run_temperature_scaling

# Bootstrap 95% confidence intervals (N=2000)
python -m evaluation.run_confidence_intervals

# McNemar's statistical test vs baseline
python -m evaluation.run_statistical_tests

# GPU/CPU inference benchmark
python -m evaluation.run_inference_benchmark

# 5-fold patient-aware cross-validation
python -m evaluation.run_cross_validation
```

### Fine-Tune for Smartphone Images (PAD-UFES-20)
```bash
python finetune_padufes.py
```
Fine-tunes only the classifier head (~1.18M params) on PAD-UFES-20. All backbone and fusion weights remain frozen.

---

## Loss Function

The training loss combines Focal Loss (for class imbalance) and Symmetric Cross-Entropy (for label noise across heterogeneous datasets):

```
L_total = 0.7 × L_FocalLoss + 0.3 × L_SCE
```

- **Focal Loss** with label smoothing (ε=0.1, γ=2.0), per-class √inverse-frequency weights, 2× melanoma boost
- **Symmetric Cross-Entropy** for robustness to mislabelled samples when merging across datasets

---

## Limitations

- **Zero-shot smartphone performance** (AUC 0.642) is insufficient for clinical use — domain gap is a known challenge in the field
- **Catastrophic forgetting** occurs under head-only fine-tuning (MEL sensitivity drops 92.3% → 77.4% on DERM7PT after PAD-UFES adaptation) — LoRA fine-tuning is proposed as future work
- **Dataset bias** — training data is predominantly Fitzpatrick phototype I–II; PAD-UFES-20 (phototypes III–VI) reveals systematic degradation for darker skin
- **Model size** — 497M parameters require a GPU for practical deployment; knowledge distillation to a compact model is planned

---

## Skin Cancer Classes

| Code | Full Name | Type |
|---|---|---|
| `mel` | Melanoma | 🔴 Malignant |
| `bcc` | Basal Cell Carcinoma | 🔴 Malignant |
| `akiec` | Actinic Keratosis / Intraepithelial Carcinoma | 🟠 Pre-malignant |
| `bkl` | Benign Keratosis-like Lesions | 🟢 Benign |
| `nv` | Melanocytic Nevi | 🟢 Benign |
| `df` | Dermatofibroma | 🟢 Benign |
| `vasc` | Vascular Lesions | 🟣 Vascular |

---

## Citation

If you use this work, please cite:

```bibtex
@article{abdullah2026dermafusion,
  title     = {DermaFusion-AI: A Dual-Branch EVA-02 and ConvNeXt Fusion Architecture
               with Segmentation-Guided Attention for Multi-Class Skin Lesion Classification},
  author    = {Abdullah, Muhammad},
  year      = {2026},
  note      = {Targeting MIDL 2026 / MDPI Diagnostics.
               AUC 0.9908, MEL Sensitivity 92.2\%, HAM10000 + ISIC 2019/2024 benchmark.},
  url       = {https://github.com/ai-with-abdullah/DermaFusion-AI}
}
```

---

## License

This project is licensed under the **MIT License**. See [`LICENSE`](LICENSE) for full details.

---

<div align="center">

**Department of Artificial Intelligence, The Islamia University of Bahawalpur, Pakistan**  
*March 2026*

</div>
