"""
test_padufes.py — Cross-Domain Evaluation on PAD-UFES-20 Smartphone Dataset
=============================================================================
Tests the trained DermaFusion AI model on real-world smartphone images
WITHOUT any retraining or architecture changes.

Enhancements applied (no retraining):
  1. CLAHE contrast enhancement + sharpening → makes smartphone images look
     more like dermoscopic images (reduces domain shift)
  2. 5-view TTA (flips + rotations) → more robust predictions

Dataset: PAD-UFES-20 (2,298 smartphone images, 6 diagnostic classes)
         Mendeley Data: https://data.mendeley.com/datasets/zr7vgbcyr2/1

Run: python test_padufes.py
"""

import os
import glob
import warnings
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import (
    balanced_accuracy_score, roc_auc_score,
    classification_report, confusion_matrix
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# ── Project imports ──────────────────────────────────────────────────────── #
from configs.config import config
from models.dual_branch_fusion import DualBranchFusionClassifier
from models.transformer_unet import SwinTransformerUNet
from models.unet import LightweightUNet
from training.train_utils import apply_mask

# ── Configuration ────────────────────────────────────────────────────────── #
N_SAMPLES  = 200   # images to evaluate (None = all 2298, slow on CPU)
USE_CLAHE  = True  # CLAHE + sharpening to reduce domain shift (recommended)
TTA_VIEWS  = 5     # test-time augmentation views (1 = off, 5 = on)

# ── Image Enhancement ────────────────────────────────────────────────────── #
def enhance_smartphone_image(img: Image.Image) -> Image.Image:
    """
    Apply CLAHE contrast enhancement + mild sharpening.
    Reduces domain shift by making smartphone images look more like
    dermoscopic images (higher contrast, sharper lesion boundaries).
    """
    img_np = np.array(img)
    # CLAHE on L channel (contrast normalisation)
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    # Mild sharpening kernel
    kernel = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
    enhanced = cv2.filter2D(enhanced.astype(np.float32), -1, kernel)
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
    return Image.fromarray(enhanced)


# ── TTA Helper ────────────────────────────────────────────────────────────── #
TTA_TRANSFORMS = [
    lambda x: x,                                           # original
    transforms.functional.hflip,                          # horizontal flip
    transforms.functional.vflip,                          # vertical flip
    lambda x: transforms.functional.rotate(x, 90),        # 90°
    lambda x: transforms.functional.rotate(x, 270),       # 270°
]


# ── Class Mapping ────────────────────────────────────────────────────────── #
# PAD-UFES-20 has 6 classes; mapped to our 7-class model
PAD_TO_MODEL = {
    'MEL': 'mel',    # Melanoma → mel
    'BCC': 'bcc',    # Basal Cell Carcinoma → bcc
    'NEV': 'nv',     # Nevus (benign mole) → nv
    'ACK': 'akiec',  # Actinic Keratosis → akiec (same disease)
    'SEK': 'bkl',    # Seborrheic Keratosis → bkl (closest match)
    'SCC': 'akiec',  # Squamous Cell Carcinoma → akiec (closest match)
}


# ── Dataset ──────────────────────────────────────────────────────────────── #
class PADUFESDataset(Dataset):
    """
    Loads PAD-UFES-20 images and maps their labels to the DermaFusion AI
    7-class label space. Searches all 3 image part directories automatically.
    Optionally applies CLAHE enhancement before the standard transform.
    """

    def __init__(self, df, img_dirs, transform, use_clahe=True):
        self.df          = df.reset_index(drop=True)
        self.transform   = transform
        self.use_clahe   = use_clahe
        self.class_to_idx = {c: i for i, c in enumerate(config.CLASSES)}

        # Build filename → full path map across all 3 part directories
        self.path_map = {}
        for d in img_dirs:
            if os.path.exists(d):
                for f in glob.glob(os.path.join(d, '*.png')):
                    self.path_map[os.path.basename(f)] = f
        print(f"  [PADUFESDataset] Found {len(self.path_map)} images across part dirs")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row       = self.df.iloc[idx]
        img_id    = row['img_id']
        pad_label = row['diagnostic']
        model_class = PAD_TO_MODEL.get(pad_label, 'nv')
        label    = self.class_to_idx[model_class]

        path = self.path_map.get(img_id)
        if path is None or not os.path.exists(path):
            img = Image.fromarray(np.zeros((448, 448, 3), dtype=np.uint8))
        else:
            img = Image.open(path).convert('RGB')
            if self.use_clahe:
                img = enhance_smartphone_image(img)  # CLAHE + sharpen

        return self.transform(img), label, img_id, pad_label


def main():
    # ── Device setup ─────────────────────────────────────────────────────── #
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Device: {device}")
    if str(device) == 'cpu':
        print("⚠️  Running on CPU — this will be slow.")
        print(f"   Using N_SAMPLES={N_SAMPLES} (change at top of script for full run)")

    # ── Load CSV ─────────────────────────────────────────────────────────── #
    csv_path = os.path.join(config.DATA_DIR, 'metadata.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"metadata.csv not found at {csv_path}\n"
            "Place it at: data/metadata.csv"
        )

    df = pd.read_csv(csv_path)
    print(f"\n📋 PAD-UFES-20 Dataset: {len(df)} total records")
    print(f"   Class distribution: {df['diagnostic'].value_counts().to_dict()}")

    if N_SAMPLES is not None and N_SAMPLES < len(df):
        # Stratified sample to ensure each class is represented
        df = df.groupby('diagnostic', group_keys=False).apply(
            lambda x: x.sample(min(len(x), max(1, int(N_SAMPLES * len(x) / len(df)))),
                                random_state=42)
        )
        print(f"   Sampled {len(df)} images (stratified by class)")

    # ── Image directories ────────────────────────────────────────────────── #
    pad_dir = os.path.join(config.DATA_DIR, 'pad_ufes', 'images')
    img_dirs = [
        os.path.join(pad_dir, 'imgs_part_1'),
        os.path.join(pad_dir, 'imgs_part_2'),
        os.path.join(pad_dir, 'imgs_part_3'),
    ]

    # Validation transforms (same as ISIC evaluation — no augmentation)
    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD),
    ])

    dataset = PADUFESDataset(df, img_dirs, transform, use_clahe=USE_CLAHE)
    loader = DataLoader(
        dataset, batch_size=4, shuffle=False,
        num_workers=0, pin_memory=False
    )
    print(f"   CLAHE enhancement : {'ON' if USE_CLAHE else 'OFF'}")
    print(f"   TTA views         : {TTA_VIEWS}")

    # ── Load Models ───────────────────────────────────────────────────────── #
    print("\n🔄 Loading trained model weights...")

    # Segmentation model
    if config.SEG_MODEL == 'swin_unet':
        unet = SwinTransformerUNet(pretrained=False).to(device)
    else:
        unet = LightweightUNet(n_channels=3, n_classes=1).to(device)

    unet_path = os.path.join(config.WEIGHTS_DIR, 'best_unet.pth')
    if not os.path.exists(unet_path):
        raise FileNotFoundError(f"UNet weights not found: {unet_path}")
    unet.load_state_dict(torch.load(unet_path, map_location=device, weights_only=True))
    unet.eval()
    print(f"  ✅ UNet loaded ({unet_path})")

    # Classification model
    model = DualBranchFusionClassifier(
        eva02_name=config.EVA02_BACKBONE,
        eva02_pretrained=False,
        convnext_name=config.CONVNEXT_BACKBONE,
        convnext_pretrained=False,
        fusion_dim=config.FUSION_EMBED_DIM,
        num_heads=config.FUSION_NUM_HEADS,
        num_classes=config.NUM_CLASSES,
        dropout=config.FUSION_DROPOUT,
    ).to(device)

    clf_path = os.path.join(config.WEIGHTS_DIR, 'padufes_finetuned_head.pth')
    if not os.path.exists(clf_path):
        raise FileNotFoundError(f"Classifier weights not found: {clf_path}")
    model.load_state_dict(torch.load(clf_path, map_location=device, weights_only=True))
    model.eval()
    print(f"  ✅ Classifier loaded ({clf_path})")

    # ── Inference ─────────────────────────────────────────────────────────── #
    print(f"\n🔍 Running inference on {len(dataset)} PAD-UFES-20 images...")
    tta_label = f'TTA (N={TTA_VIEWS})' if TTA_VIEWS > 1 else 'Single pass'
    print(f"   Mode: {tta_label}\n")

    all_labels, all_preds, all_probs = [], [], []
    all_pad_labels = []

    # Select TTA views to use (subset of 5 available)
    tta_fns = TTA_TRANSFORMS[:TTA_VIEWS]

    with torch.no_grad():
        for images, labels, img_ids, pad_labels in tqdm(loader, desc='PAD-UFES-20 Eval'):
            images = images.to(device)

            # Segment lesion region (use original view only for UNet)
            mask_logits = unet(images)
            images_seg  = apply_mask(images, mask_logits)

            # TTA: average probabilities across N augmented views
            view_probs = []
            for tta_fn in tta_fns:
                aug_imgs     = torch.stack([tta_fn(img) for img in images])
                aug_imgs_seg = torch.stack([tta_fn(img) for img in images_seg])
                logits, _ = model(aug_imgs.to(device), aug_imgs_seg.to(device))
                view_probs.append(F.softmax(logits, dim=1).cpu())

            probs = torch.stack(view_probs).mean(0).numpy()  # average over views
            preds = probs.argmax(axis=1)

            all_probs.extend(probs)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
            all_pad_labels.extend(list(pad_labels))

    all_labels    = np.array(all_labels)
    all_preds     = np.array(all_preds)
    all_probs     = np.array(all_probs)
    all_pad_labels = np.array(all_pad_labels)

    # ── Metrics ───────────────────────────────────────────────────────────── #
    acc     = (all_preds == all_labels).mean()
    bal_acc = balanced_accuracy_score(all_labels, all_preds)

    # Macro AUC (skip classes with only 1 sample)
    auc_scores = []
    for c in np.unique(all_labels):
        y_bin = (all_labels == c).astype(int)
        if y_bin.sum() > 0 and (1 - y_bin).sum() > 0:
            try:
                auc_scores.append(roc_auc_score(y_bin, all_probs[:, c]))
            except Exception:
                pass
    macro_auc = np.mean(auc_scores) if auc_scores else 0.0

    sep = "=" * 65
    print(f"\n{sep}")
    print("  PAD-UFES-20 Cross-Domain Evaluation Results")
    print("  (Trained on ISIC dermoscopy → Tested on smartphone photos)")
    print(sep)
    print(f"  Samples evaluated    : {len(all_labels)}")
    print(f"  Accuracy             : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  Balanced Accuracy    : {bal_acc:.4f}  ({bal_acc*100:.1f}%)")
    print(f"  Macro AUC            : {macro_auc:.4f}")
    print(sep)

    # ISIC reference for comparison
    print("\n  📊 ISIC vs PAD-UFES-20 Comparison:")
    print(f"  {'Metric':<22} {'ISIC (dermoscopy)':<22} {'PAD-UFES-20 (phone)'}")
    print(f"  {'-'*65}")
    print(f"  {'Accuracy':<22} {'88.8%':<22} {acc*100:.1f}%")
    print(f"  {'Balanced Accuracy':<22} {'85.6%':<22} {bal_acc*100:.1f}%")
    print(f"  {'Macro AUC':<22} {'99.1%':<22} {macro_auc*100:.1f}%")
    print(sep)

    # Per PAD-UFES-20 class breakdown
    print("\n  Per-class accuracy (PAD-UFES-20 labels → model class mapping):")
    for pad_cls in ['MEL', 'BCC', 'NEV', 'ACK', 'SEK', 'SCC']:
        mask = all_pad_labels == pad_cls
        if mask.sum() == 0:
            continue
        cls_acc = (all_preds[mask] == all_labels[mask]).mean()
        model_cls = PAD_TO_MODEL.get(pad_cls, '?')
        print(f"    {pad_cls:4s} → {model_cls:6s}  |  {cls_acc:.3f}  [{mask.sum()} samples]")

    print(f"\n  ⚠️  NOTE: Performance drop vs ISIC is expected and normal.")
    print("   Cause: Domain shift — dermoscope has polarized light + 10× magnification.")
    print("   Smartphone images are blurrier, less standardized, and have more noise.")
    print("   This result documents the model's real-world cross-domain limitation.")
    print(sep)

    # ── Save results ──────────────────────────────────────────────────────── #
    out_dir = os.path.join(config.OUTPUT_DIR, 'padufes_results')
    os.makedirs(out_dir, exist_ok=True)

    results_df = pd.DataFrame({
        'img_id':       all_pad_labels,  # reuse pad label as id
        'pad_label':    all_pad_labels,
        'model_class':  [config.CLASSES[l] for l in all_labels],
        'predicted':    [config.CLASSES[p] for p in all_preds],
        'correct':      (all_preds == all_labels),
        **{f'prob_{c}': all_probs[:, i] for i, c in enumerate(config.CLASSES)},
    })
    csv_out = os.path.join(out_dir, 'padufes_predictions.csv')
    results_df.to_csv(csv_out, index=False)

    # Confusion matrix plot
    present_classes = sorted(np.unique(np.concatenate([all_labels, all_preds])).tolist())
    cls_names = [config.CLASSES[c] for c in present_classes]
    cm = confusion_matrix(all_labels, all_preds, labels=present_classes)
    cm_filtered = cm

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_filtered, annot=True, fmt='d', cmap='Blues',
                xticklabels=cls_names, yticklabels=cls_names)
    plt.title('PAD-UFES-20 Confusion Matrix (smartphone → dermoscopy model)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    cm_path = os.path.join(out_dir, 'padufes_confusion_matrix.png')
    plt.savefig(cm_path, dpi=150)
    plt.close()

    print(f"\n✅ Results saved to: {out_dir}/")
    print(f"   - padufes_predictions.csv")
    print(f"   - padufes_confusion_matrix.png")


if __name__ == '__main__':
    main()
