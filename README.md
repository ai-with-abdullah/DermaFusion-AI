# DermaFusion-AI: State-of-the-Art Skin Lesion Classification

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)

**DermaFusion-AI** is a bleeding-edge, dual-branch deep learning pipeline designed for high-accuracy skin cancer detection and melanoma classification. It leverages a fusion of global contextual understanding from Vision Transformers (EVA-02) and local texture extraction from modern Convolutional Networks (ConvNeXt V2), guided by a Swin-Transformer UNet segmentation mask.

Designed to align with 2024/2026 State-of-the-Art (SOTA) ISIC competition architectures, this repository is optimized for both academic research and robust clinical deployment.

---

## 🌟 Key Features

*   **Dual-Branch Architecture:**
    *   **Branch A (Global Context):** EVA-02 Small (`eva02_small_patch14_336.mim_in22k_ft_in1k`)
    *   **Branch B (Local Texture):** ConvNeXt V2 Base (`convnextv2_base.fcmae_ft_in22k_in1k_384`)
*   **Segmentation-Guided Attention:** A Swin-UNet (384-native) generates precise lesion masks, forcing the ConvNeXt branch to focus purely on the lesion characteristics, mitigating lighting and skin-type domain shifts.
*   **Asymmetric Melanoma Focal Loss:** Clinically motivated custom loss function that heavily penalizes Melanoma False Negatives (3× penalty), maximizing sensitivity (recall) for the most dangerous skin cancer.
*   **Advanced Training Paradigm:**
    *   Layer-wise Learning Rate Decay (LLRD) for optimal ViT fine-tuning.
    *   Mixed-precision training (`torch.amp.autocast`).
    *   Weighted Random Sampling + Inverse Frequency Damping (handles extreme 60:1 imbalances).
    *   CutMix & MixUp applied *after* segmentation masking to preserve label integrity.
*   **Clinical-Grade Inference:**
    *   **8-View Test-Time Augmentation (TTA):** Flips, full rotations, and lighting shifts for robust predictions.
    *   **Temperature Scaling Calibration:** Scipy-optimized global temperature scaling reduces Expected Calibration Error (ECE), providing trustworthy probabilities.
    *   **Explainable AI (XAI):** Native Dual-Branch GradCAM++ generates visual heatmaps of the model's decision-making process.
*   **Unified Multi-Dataset Loader:** Automatically parses, merges, and balances HAM10000, ISIC 2019, ISIC 2020, ISIC 2024 (SLICE-3D), and PH2 datasets.

---

## 📊 Supported Datasets
The `UnifiedSkinDataset` loader seamlessly handles:
1.  **HAM10000** (10,015 images, 7 classes)
2.  **ISIC 2019** (25,331 images, 8 classes)
3.  **ISIC 2020** (Melanoma positives only to prevent label noise)
4.  **ISIC 2024 SLICE-3D** (Auto-balanced negative downsampling)
5.  **PH2** (High-quality dermoscopy dataset)

---

## 🛠️ Installation

**1. Clone the repository:**
```bash
git clone https://github.com/ai-with-abdullah/DermaFusion-AI.git
cd DermaFusion-AI
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Training
Configure your hyper-parameters in `configs/config.py` and run the main training script. The script automatically handles multi-dataset merging if the data is present in the `DATA_DIR`.

```bash
python train_classifier.py --epochs 30
```

### 2. Calibration (Post-Training)
Before final evaluation, calculate the optimal Temperature scaling factor on your validation set to fix the Expected Calibration Error (ECE).

```bash
python evaluation/calibration.py
```

### 3. Evaluation & XAI
Run the comprehensive evaluation pipeline. This handles TTA, applies the calibrated temperature scale, calculates standard and threshold-boosted metrics, and generates GradCAM++ heatmaps.

```bash
python evaluate.py
```

---

## 📈 State-of-the-Art Alignment
This architecture is heavily inspired by the winning solutions of the **ISIC 2024 Skin Cancer Detection Challenge**, specifically adopting:
*   **EVA-02 & ConvNeXt Extractor Blend:** The documented optimal combination for this domain.
*   **Extensive TTA Strategy:** 8+ deterministic views.
*   **Clinical FN Penalty Focus:** Optimizing for Melanoma Sensitivity over pure accuracy.

---

## 🤝 Contributing
Contributions, issues, and feature requests are welcome!

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.
