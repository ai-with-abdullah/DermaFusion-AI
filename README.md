# DermaFusion-AI 🔬

**State-of-the-Art Dual-Branch Fusion Classifier for Melanoma Detection and Skin Lesion Classification**

![Dual-Branch Framework](https://img.shields.io/badge/Architecture-Dual--Branch%20Fusion-blue)
![Dependencies](https://img.shields.io/badge/PyTorch-2.0+-red)
![SOTA](https://img.shields.io/badge/Performance-2026%20SOTA-success)

## 📌 Overview
DermaFusion-AI is an advanced medical image classification framework designed to achieve State-of-the-Art (SOTA) accuracy in dermatologist-level skin cancer detection. By harnessing the global contextual understanding of Vision Transformers (**EVA-02**) and the local textural feature extraction of Modern Convolutional Networks (**ConvNeXt V2**), this model is highly effective at identifying subtle lesion boundaries and morphological patterns.

This pipeline natively supports multi-dataset training across the **ISIC** and **HAM10000** benchmarks and incorporates clinical priorities (such as a heavy asymmetric penalty for melanoma false negatives) directly into the loss landscape.

## ✨ Key Features
- **Swin-UNet Segmentation Masking**: Forces the CNN branch to focus strictly on the lesion by dynamically masking out background skin and artifacts (hair, dermatoscope gel bubbles).
- **Dual-Branch Fusion Model**: End-to-end differentiable cross-attention fusion of `eva02_small_patch14_336` and `convnextv2_base`.
- **Layer-wise Learning Rate Decay (LLRD)**: Optimized fine-tuning methodology to prevent catastrophic forgetting in massive pre-trained ViTs.
- **Test-Time Augmentation (TTA)**: 8-view deterministic multi-crop eval (flips, rotations, brightness shifts).
- **Asymmetric Melanoma Loss**: Custom focal loss weighted heavily (3x) against Melanoma (MEL) false negatives, matching clinical imperatives.
- **Temperature Scaling Calibration**: Post-hoc Expected Calibration Error (ECE) reduction to yield deeply trustworthy, calibrated clinical probabilities.
- **Explainable AI (GradCAM++)**: Integrated dual-backbone visual heatmaps for transparent clinical predictions.

## 📂 Supported Datasets
The `UnifiedDataset` dataset loader natively integrates:
- Kaggle SLICE-3D (ISIC 2024)
- ISIC 2020 (Safely heavily curated for MEL positives)
- ISIC 2019
- HAM10000
- PH2

## 🚀 Getting Started

### 1. Requirements
Install via standard pip installation:
```bash
pip install -r requirements.txt
```

### 2. Training
Run the training script (supports multi-GPU and single-GPU setups):
```bash
python train_classifier.py --epochs 30
```

### 3. Evaluation & Calibration
Evaluate performance, generate GradCAM++ heatmaps, and produce standard multi-class ROCs:
```bash
# Optional: Calibrate model ECE on a validation set first
python evaluation/calibration.py

# Run standard Test-Time inference and metrics calculation
python evaluate.py
```

## 📝 Citation
*This project was developed for advanced MS AI research focusing on bridging the gap between clinical requirements and State-of-the-Art Vision Foundation Models.*
