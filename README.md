# DermaFusion-AI 🔬

**A Dual-Branch Vision Transformer for Dermatological Image Classification**

DermaFusion-AI combines EVA-02 Large (global attention) and ConvNeXt V2 Base (local texture) through bidirectional cross-attention fusion and Swin-UNet lesion segmentation for 7-class skin cancer classification on HAM10000/ISIC benchmarks.

---

## Architecture

```
Input Image (448×448)
       │
       ├──► Swin-UNet Segmentation ──► Segmented Image
       │                                      │
       ▼                                      ▼
EVA-02 Large (304M)              ConvNeXt V2 Base (88M)
  Global Context                  Local Texture Features
       │                                      │
       └──────── Cross-Attention Fusion ───────┘
                         │
                    Classifier Head
                         │
               7-Class Logits (skin cancer)
```

| Component | Backbone | Params | GFLOPs |
|---|---|---|---|
| Branch A (Global) | EVA-02 Large 448px | 304.1M | 296.8 |
| Branch B (Texture) | ConvNeXt V2 Base 384px | 88.5M | 59.1 |
| Fusion + Head | Cross-Attention | 9.0M | — |
| Segmentation | Swin-UNet | 95.5M | 59.1 |
| **Total** | — | **497.1M** | **415.0** |

---

## Results

### HAM10000 + ISIC 2020/2024 Test Set

| Metric | Value |
|---|---|
| Macro AUC | **0.9908** |
| Balanced Accuracy | **85.59%** |
| Macro F1 | **0.7991** |
| MEL Sensitivity | **92.16%** |
| ECE (after temperature scaling) | **0.0390** (T=1.460) |

### 5-Fold Cross-Validation (EVA-02 Small, HAM10000)

| Metric | Mean ± SD |
|---|---|
| AUC | 0.9512 ± 0.0093 |
| Balanced Accuracy | 0.7698 ± 0.0152 |
| Macro F1 | 0.7161 ± 0.0391 |
| MEL Sensitivity | 0.7553 ± 0.0414 |

### Statistical Validation
- **McNemar's test** vs EVA-02 alone: χ²=12.96, df=1, **p < 0.001**
- **Bootstrap 95% CI** (N=2000): AUC [0.9939, 0.9977]
- **Temperature scaling**: ECE reduced from 0.0661 → 0.0390

### Inference (NVIDIA Tesla T4)
| Configuration | Latency |
|---|---|
| Swin-UNet segmentation | 44.7 ± 1.3 ms |
| DermaFusion classifier | 326.7 ± 5.7 ms |
| **End-to-end** | **371.4 ms/image** |
| Batch throughput (batch=8) | 3.3 images/sec |

---

## Skin Cancer Classes

| Class | Full Name |
|---|---|
| `mel` | Melanoma |
| `nv` | Melanocytic Nevi |
| `bcc` | Basal Cell Carcinoma |
| `akiec` | Actinic Keratosis / Intraepithelial Carcinoma |
| `bkl` | Benign Keratosis |
| `df` | Dermatofibroma |
| `vasc` | Vascular Lesions |

---

## Setup

```bash
git clone https://github.com/ai-with-abdullah/DermaFusion-AI.git
cd DermaFusion-AI
pip install -r requirements.txt
```

### Weights

Weights are hosted on Kaggle (too large for GitHub):
- **Classifier** (`best_dual_branch_fusion.pth`): [kaggle.com/datasets/aiwithAbdullah71/weights1](https://kaggle.com/datasets/aiwithAbdullah71/weights1)
- **Swin-UNet** (`best_unet.pth`): [kaggle.com/datasets/aiwithAbdullah71/weights](https://kaggle.com/datasets/aiwithAbdullah71/weights)

Place weights in `outputs/weights/` or update paths in `configs/config.py`.

---

## Project Structure

```
DermaFusion-AI/
├── configs/                  # Config & hyperparameters
│   └── config.py
├── models/                   # Model definitions
│   ├── dual_branch_fusion.py # Main DermaFusion-AI model
│   ├── transformer_unet.py   # Swin-UNet segmentation
│   ├── sota25_backbone.py    # EVA-02 wrapper
│   └── convnextv3_backbone.py# ConvNeXt V2 wrapper
├── datasets/                 # Dataset loaders
│   ├── unified_dataset.py    # HAM10000 + ISIC multi-dataset
│   └── augmentations.py      # Training augmentations
├── training/                 # Training utilities
├── evaluation/               # Evaluation scripts
│   ├── run_temperature_scaling.py
│   ├── run_confidence_intervals.py
│   ├── run_statistical_tests.py
│   ├── run_inference_benchmark.py
│   └── run_cross_validation.py
├── utils/                    # Logging, seeding, metrics
├── outputs/                  # Results, weights (gitignored)
├── train_classifier.py       # Main training script
├── train_segmentation.py     # Swin-UNet training
├── evaluate.py               # Full evaluation pipeline
├── app.py                    # Gradio demo app
├── train_on_kaggle.ipynb     # Kaggle training notebook
├── train_on_colab.ipynb      # Colab training notebook
├── input.md                  # Research paper (full)
├── cv_summary.csv            # 5-fold CV results
└── requirements.txt
```

---

## Training

### Classifier (Kaggle T4×2, ~40 epochs)
```bash
python train_classifier.py
```

### Segmentation (Swin-UNet)
```bash
python train_segmentation.py
```

### 5-Fold Cross-Validation
```bash
python -m evaluation.run_cross_validation
```

### Evaluation Suite
```bash
# Temperature scaling
python -m evaluation.run_temperature_scaling

# Bootstrap confidence intervals
python -m evaluation.run_confidence_intervals

# McNemar's statistical test
python -m evaluation.run_statistical_tests

# Inference benchmark (GPU recommended)
python -m evaluation.run_inference_benchmark
```

---

## Demo

```bash
python app.py
```

Opens a Gradio interface for uploading dermoscopy images and getting 7-class predictions with confidence scores.

---

## Citation

If you use this work, please cite:

```bibtex
@article{dermafusion2025,
  title   = {DermaFusion-AI: Dual-Branch Vision Transformer for Dermatological Image Classification},
  author  = {Abdullah},
  journal = {MDPI Diagnostics / MIDL 2026},
  year    = {2025},
  note    = {EVA-02 Large + ConvNeXt V2 + Swin-UNet, HAM10000/ISIC benchmark}
}
```

---

## License

MIT License — see `LICENSE` for details.
