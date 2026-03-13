"""
test_both_weights.py — DERM7PT Cross-Domain Evaluation
=======================================================
Tests BOTH weight files on the DERM7PT dataset in a single run.

DERM7PT has two image types per case:
  - 'clinic' = clinical/smartphone photo
  - 'derm'   = dermoscopic photo

This script tests all 4 combinations:
  Original weights  × smartphone images
  Original weights  × dermoscopic images
  Fine-tuned weights × smartphone images
  Fine-tuned weights × dermoscopic images

Run: python test_both_weights.py
"""

import os, glob, warnings, cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

warnings.filterwarnings('ignore')

from configs.config import config
from models.dual_branch_fusion import DualBranchFusionClassifier
from models.transformer_unet import SwinTransformerUNet
from models.unet import LightweightUNet
from training.train_utils import apply_mask

# ══════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════
BASE_DIR   = 'release_v0'
CSV_PATH   = os.path.join(BASE_DIR, 'meta', 'meta.csv')
IMAGES_DIR = os.path.join(BASE_DIR, 'images')   # subfolders are inside here
USE_CLAHE  = True
TTA_VIEWS  = 1        # keep 1 for speed; set to 3 for paper-quality results
BATCH_SIZE = 4

# DERM7PT diagnosis → model class name
LABEL_MAP = {
    'melanoma':                         'mel',
    'melanoma (in situ)':               'mel',
    'melanoma (less than 0.76 mm)':     'mel',
    'melanoma (0.76 to 1.5 mm)':        'mel',
    'melanoma (more than 1.5 mm)':      'mel',
    'basal cell carcinoma':             'bcc',
    'clark nevus':                      'nv',
    'blue nevus':                       'nv',
    'combined nevus':                   'nv',
    'congenital nevus':                 'nv',
    'dermal nevus':                     'nv',
    'seborrheic keratosis':             'bkl',
    'lentigo':                          'bkl',
    'dermatofibroma':                   'df',
    'vascular lesion':                  'vasc',
    'squamous cell carcinoma':          'akiec',
}

# Both weight files to compare
WEIGHT_FILES = {
    'Original (ISIC trained)':        'best_dual_branch_fusion.pth',
    'Fine-tuned (PAD-UFES adapted)':  'padufes_finetuned_head.pth',
}

# Image types to test  ('clinic' = smartphone,  'derm' = dermoscope)
IMAGE_COLS = {
    'Smartphone (clinic)':   'clinic',
    'Dermoscope (derm)':     'derm',
}
# ══════════════════════════════════════════════════════════════════

CLASS_TO_IDX = {c: i for i, c in enumerate(config.CLASSES)}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def enhance_image(img: Image.Image) -> Image.Image:
    img_np = np.array(img)
    lab    = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    clahe  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    kernel   = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
    enhanced = cv2.filter2D(enhanced.astype(np.float32), -1, kernel)
    return Image.fromarray(np.clip(enhanced, 0, 255).astype(np.uint8))


class DERM7PTDataset(Dataset):
    """Dataset for DERM7PT — paths in CSV are 'subfolder/filename.jpg'"""
    def __init__(self, df, img_col, images_dir, transform, use_clahe=True):
        self.df         = df.reset_index(drop=True)
        self.img_col    = img_col
        self.images_dir = images_dir
        self.transform  = transform
        self.use_clahe  = use_clahe

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        rel_path = str(row[self.img_col]).strip()   # e.g.  'NEL/NEL025.JPG'
        diagnosis = str(row['diagnosis']).strip().lower()
        model_cls = LABEL_MAP.get(diagnosis, 'nv')
        label     = CLASS_TO_IDX.get(model_cls, CLASS_TO_IDX['nv'])

        # Build absolute path
        full_path = os.path.join(self.images_dir, rel_path)
        # Try case-insensitive match if exact path doesn't exist
        if not os.path.exists(full_path):
            folder = os.path.dirname(full_path)
            fname  = os.path.basename(full_path)
            for f in glob.glob(os.path.join(folder, '*')):
                if os.path.basename(f).lower() == fname.lower():
                    full_path = f
                    break

        if os.path.exists(full_path):
            img = Image.open(full_path).convert('RGB')
            if self.use_clahe:
                img = enhance_image(img)
        else:
            img = Image.fromarray(np.zeros((448, 448, 3), dtype=np.uint8))

        return self.transform(img), label


def load_unet():
    if config.SEG_MODEL == 'swin_unet':
        unet = SwinTransformerUNet(pretrained=False).to(device)
    else:
        unet = LightweightUNet(n_channels=3, n_classes=1).to(device)
    unet_path = os.path.join(config.WEIGHTS_DIR, 'best_unet.pth')
    unet.load_state_dict(torch.load(unet_path, map_location=device, weights_only=True))
    for p in unet.parameters(): p.requires_grad = False
    unet.eval()
    return unet


def load_classifier(weight_file):
    model = DualBranchFusionClassifier(
        eva02_name=config.EVA02_BACKBONE, eva02_pretrained=False,
        convnext_name=config.CONVNEXT_BACKBONE, convnext_pretrained=False,
        fusion_dim=config.FUSION_EMBED_DIM, num_heads=config.FUSION_NUM_HEADS,
        num_classes=config.NUM_CLASSES, dropout=config.FUSION_DROPOUT,
    ).to(device)
    path = os.path.join(config.WEIGHTS_DIR, weight_file)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model


def _apply_tta_view(images: torch.Tensor, view_idx: int) -> torch.Tensor:
    """
    Deterministic in-tensor TTA transform. Inlined here so this script
    works on both local and Google Colab without needing 'training.tta'.
      0: Original  1: H-flip  2: V-flip  3: 90° rot  4: 180° rot
    """
    if view_idx == 0:
        return images
    elif view_idx == 1:
        return torch.flip(images, dims=[-1])
    elif view_idx == 2:
        return torch.flip(images, dims=[-2])
    elif view_idx == 3:
        return torch.rot90(images, k=1, dims=[-2, -1])
    elif view_idx == 4:
        return torch.rot90(images, k=2, dims=[-2, -1])
    else:
        return images


@torch.no_grad()
def run_tta(model, unet, loader, n_views=3):
    """
    TRUE Test-Time Augmentation: runs N forward passes per batch with
    different deterministic transforms and averages softmax probabilities.

    FIXED: Previously probs_list had only 1 element (plain evaluation).
    Now applies _apply_tta_view() for each view — real TTA.

    UNet segments ONCE on the original image; the same spatial transform
    is applied to both branches per view.
    """
    all_probs, all_labels = [], []

    for images, labels in tqdm(loader, desc='  Evaluating', leave=False):
        images = images.to(device)

        # Segment once on original image (UNet is deterministic)
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            mask_logits = unet(images)
        images_seg = apply_mask(images, mask_logits)

        # Run N augmented forward passes and average
        view_probs = []
        for view_idx in range(n_views):
            aug_images     = _apply_tta_view(images,     view_idx)
            aug_images_seg = _apply_tta_view(images_seg, view_idx)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                logits, _ = model(aug_images, aug_images_seg)
            view_probs.append(F.softmax(logits, dim=1).cpu().numpy())

        avg_probs = np.mean(view_probs, axis=0)  # (B, C) averaged over N views
        all_probs.extend(avg_probs)
        all_labels.extend(labels.numpy())

    return np.array(all_probs), np.array(all_labels)


def compute_metrics(all_probs, all_labels, label_name):
    preds   = all_probs.argmax(1)
    acc     = (preds == all_labels).mean()
    bal_acc = balanced_accuracy_score(all_labels, preds)
    mel_idx = CLASS_TO_IDX['mel']
    mel_mask = all_labels == mel_idx
    mel_sens = (preds[mel_mask] == mel_idx).mean() if mel_mask.sum() > 0 else float('nan')

    auc_scores = []
    for c in np.unique(all_labels):
        y_bin = (all_labels == c).astype(int)
        if y_bin.sum() > 0 and (1 - y_bin).sum() > 0:
            try: auc_scores.append(roc_auc_score(y_bin, all_probs[:, c]))
            except: pass
    macro_auc = np.mean(auc_scores) if auc_scores else float('nan')

    per_class = {}
    for cls_name, c_idx in CLASS_TO_IDX.items():
        mask = all_labels == c_idx
        if mask.sum() > 0:
            per_class[cls_name] = (preds[mask] == c_idx).mean()

    return {
        'label':     label_name,
        'n':         len(all_labels),
        'acc':       acc,
        'bal_acc':   bal_acc,
        'auc':       macro_auc,
        'mel_sens':  mel_sens,
        'per_class': per_class,
    }


def print_final_table(all_results):
    """Print 4-row table: 2 weights × 2 image types."""
    sep = '═' * 90
    print(f'\n{sep}')
    print('  DERM7PT CROSS-DOMAIN EVALUATION — All 4 Combinations')
    print(f'  CSV: {CSV_PATH}')
    print(sep)
    header = f'  {"Combination":<42} {"Acc":>7} {"BalAcc":>8} {"AUC":>7} {"MELsens":>8}'
    print(header)
    print(f'  {"-"*86}')
    for r in all_results:
        mel = f'{r["mel_sens"]:.1%}' if not np.isnan(r['mel_sens']) else ' N/A '
        print(f'  {r["label"]:<42} {r["acc"]:>7.1%} {r["bal_acc"]:>8.1%} '
              f'{r["auc"]:>7.1%} {mel:>8}')
    print(sep)
    print('\n  Per-class accuracy:')
    all_classes = set()
    for r in all_results: all_classes |= set(r['per_class'].keys())
    col_w = 14
    header2 = f'  {"Class":<10}' + ''.join(f'{r["label"][:col_w-1]:<{col_w}}' for r in all_results)
    print(header2)
    print(f'  {"-"*86}')
    for cls in sorted(all_classes):
        row = f'  {cls:<10}'
        for r in all_results:
            v = r['per_class'].get(cls, float('nan'))
            val_str = f'{v:.1%}' if not np.isnan(v) else 'N/A'
            row += f'{val_str:<{col_w}}'
        print(row)
    print(sep)


def main():
    print(f'\n🖥️  Device: {device}')
    if device.type == 'cpu':
        print('⚠️  Running on CPU — consider using GPU for speed.')

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f'CSV not found: {CSV_PATH}')
    df = pd.read_csv(CSV_PATH)
    print(f'\n📋 DERM7PT: {len(df)} cases')
    print(f'   Diagnosis distribution: {df["diagnosis"].value_counts().to_dict()}')

    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD),
    ])

    print('\n🔄 Loading UNet (shared across all tests)...')
    unet = load_unet()
    print('  ✅ UNet loaded')

    all_results = []
    # Loop: image_type (smartphone / derm) × weight_file (original / fine-tuned)
    for img_type_name, img_col in IMAGE_COLS.items():
        dataset = DERM7PTDataset(df, img_col, IMAGES_DIR, transform, use_clahe=USE_CLAHE)
        loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        print(f'\n── Image type: {img_type_name} ({len(dataset)} images) ──')

        for weight_name, weight_file in WEIGHT_FILES.items():
            label = f'{weight_name} | {img_type_name}'
            print(f'  🔍 {weight_name}')
            model  = load_classifier(weight_file)
            probs, labels = run_tta(model, unet, loader, n_views=TTA_VIEWS)
            metrics = compute_metrics(probs, labels, label)
            all_results.append(metrics)
            mel = f'{metrics["mel_sens"]:.1%}' if not np.isnan(metrics['mel_sens']) else 'N/A'
            print(f'     BalAcc={metrics["bal_acc"]:.1%}  AUC={metrics["auc"]:.1%}  MEL={mel}')
            del model

    print_final_table(all_results)
    print('\n✅ Done.')


if __name__ == '__main__':
    main()
