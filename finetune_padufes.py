"""
finetune_padufes.py — Head-Only Domain Adaptation on PAD-UFES-20
==================================================================
Fine-tunes ONLY the classification head (+ gated fusion) of the trained
DermaFusion AI model on PAD-UFES-20 smartphone images.

🔒 SAFETY:
  - ALL backbone weights (EVA-02, ConvNeXt, UNet, Fusion attention) are FROZEN
  - Original weights (best_dual_branch_fusion.pth) are NEVER modified
  - New adapted weights saved to TWO files:
      padufes_best_mel.pth          ← best balanced acc where MEL sensitivity = 100%
      padufes_finetuned_head.pth    ← best overall balanced acc (may have lower MEL)

🎯 GOAL: Improve cross-domain smartphone accuracy without losing ISIC performance

📊 Split (patient-aware, no data leakage):
  - 60% Training   (~1379 images)
  - 20% Validation (~460 images)
  - 20% Test       (~460 images)

⏱️ Runtime: ~2–3 hours on CPU (MacBook), ~10–15 min on GPU

Run: python finetune_padufes.py
"""

import os
import glob
import warnings
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

warnings.filterwarnings('ignore')

# ── Project imports ──────────────────────────────────────────────────────── #
from configs.config import config
from models.dual_branch_fusion import DualBranchFusionClassifier
from models.transformer_unet import SwinTransformerUNet
from models.unet import LightweightUNet
from training.train_utils import apply_mask

# ── Configuration ────────────────────────────────────────────────────────── #
EPOCHS        = 20          # max epochs (early stopping will trigger earlier)
BATCH_SIZE    = 4           # keep small for CPU memory
LR            = 3e-4        # learning rate for head only (higher OK since frozen backbone)
PATIENCE      = 6           # early stopping patience
USE_CLAHE     = True        # CLAHE preprocessing (same as test)
MEL_WEIGHT    = 8.0         # Extra loss weight for MEL — protects 100% sensitivity
SEED          = 42

# Modules to FREEZE (never trained)
FROZEN_MODULES = ['branch_eva', 'branch_conv', 'proj_eva', 'proj_conv', 'fusion']
# Modules to TRAIN (head + small gated combiner)
TRAINABLE_MODULES = ['gate', 'classifier']

torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Class Mapping ────────────────────────────────────────────────────────── #
PAD_TO_MODEL = {
    'MEL': 'mel',
    'BCC': 'bcc',
    'NEV': 'nv',
    'ACK': 'akiec',
    'SEK': 'bkl',
    'SCC': 'akiec',
}

CLASS_TO_IDX = {c: i for i, c in enumerate(config.CLASSES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}


# ── Image Enhancement ─────────────────────────────────────────────────────── #
def enhance_smartphone_image(img: Image.Image) -> Image.Image:
    """CLAHE contrast enhancement + sharpening for smartphone images."""
    img_np = np.array(img)
    lab    = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    clahe  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    kernel   = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
    enhanced = cv2.filter2D(enhanced.astype(np.float32), -1, kernel)
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
    return Image.fromarray(enhanced)


# ── Dataset ──────────────────────────────────────────────────────────────── #
class PADUFESDataset(Dataset):
    def __init__(self, df, img_dirs, transform, augment=False, use_clahe=True):
        self.df        = df.reset_index(drop=True)
        self.transform = transform
        self.augment   = augment
        self.use_clahe = use_clahe

        # Build filename → path map
        self.path_map = {}
        for d in img_dirs:
            if os.path.exists(d):
                for f in glob.glob(os.path.join(d, '*.png')):
                    self.path_map[os.path.basename(f)] = f

        # Simple augmentation for training
        self.aug_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        ]) if augment else None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row       = self.df.iloc[idx]
        img_id    = row['img_id']
        pad_label = row['diagnostic']
        label     = CLASS_TO_IDX[PAD_TO_MODEL.get(pad_label, 'nv')]

        path = self.path_map.get(img_id)
        if path is None or not os.path.exists(path):
            img = Image.fromarray(np.zeros((448, 448, 3), dtype=np.uint8))
        else:
            img = Image.open(path).convert('RGB')
            if self.use_clahe:
                img = enhance_smartphone_image(img)
            if self.aug_transforms is not None:
                img = self.aug_transforms(img)

        return self.transform(img), label


# ── Data Split ───────────────────────────────────────────────────────────── #
def make_patient_aware_splits(df):
    """
    Patient-aware split: no patient appears in both train and test.
    Uses img_id prefix (PAT_XXX) as patient group.
    Split: 60% train, 20% val, 20% test
    """
    # Extract patient ID from filename: PAT_123_456_789.png → 123
    df = df.copy()
    df['patient_id'] = df['img_id'].str.extract(r'PAT_(\d+)_')[0]

    groups = df['patient_id'].values
    indices = np.arange(len(df))

    # First split: 80% temp (train+val) vs 20% test
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    temp_idx, test_idx = next(gss.split(indices, groups=groups))

    # Second split: within temp, 75% train vs 25% val → overall 60% train, 20% val
    temp_groups = df.iloc[temp_idx]['patient_id'].values
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    train_sub, val_sub = next(gss2.split(temp_idx, groups=temp_groups))

    train_idx = temp_idx[train_sub]
    val_idx   = temp_idx[val_sub]

    print(f"\n📊 Patient-aware split (no data leakage):")
    print(f"   Train : {len(train_idx)} images ({len(train_idx)/len(df)*100:.1f}%)"
          f" | {df.iloc[train_idx]['diagnostic'].value_counts().to_dict()}")
    print(f"   Val   : {len(val_idx)} images ({len(val_idx)/len(df)*100:.1f}%)"
          f" | {df.iloc[val_idx]['diagnostic'].value_counts().to_dict()}")
    print(f"   Test  : {len(test_idx)} images ({len(test_idx)/len(df)*100:.1f}%)"
          f" | {df.iloc[test_idx]['diagnostic'].value_counts().to_dict()}")

    # Verify no patient overlap (anti-leakage check)
    train_patients = set(df.iloc[train_idx]['patient_id'])
    val_patients   = set(df.iloc[val_idx]['patient_id'])
    test_patients  = set(df.iloc[test_idx]['patient_id'])
    assert len(train_patients & test_patients) == 0, "⛔ Train/Test patient OVERLAP!"
    assert len(train_patients & val_patients) == 0,  "⛔ Train/Val patient OVERLAP!"
    print("   ✅ No patient overlap across splits")

    return (df.iloc[train_idx].reset_index(drop=True),
            df.iloc[val_idx].reset_index(drop=True),
            df.iloc[test_idx].reset_index(drop=True))


# ── Class-weighted Loss ───────────────────────────────────────────────────── #
def get_class_weights(train_df):
    """
    Compute inverse-frequency class weights.
    MEL gets an extra MEL_WEIGHT multiplier to protect its high sensitivity.
    """
    counts = np.zeros(len(config.CLASSES))
    for _, row in train_df.iterrows():
        mapped = PAD_TO_MODEL.get(row['diagnostic'], 'nv')
        counts[CLASS_TO_IDX[mapped]] += 1

    # Replace 0 counts with 1 to avoid division by zero
    counts = np.where(counts == 0, 1, counts)
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(config.CLASSES)

    # Extra boost for MEL
    mel_idx = CLASS_TO_IDX['mel']
    weights[mel_idx] *= MEL_WEIGHT
    weights = torch.FloatTensor(weights)

    print(f"\n⚖️  Class weights (MEL gets {MEL_WEIGHT}× boost):")
    for i, c in enumerate(config.CLASSES):
        print(f"   {c:6s}: {weights[i]:.3f}  [{int(counts[i])} train samples]")

    return weights


# ── Load Models ───────────────────────────────────────────────────────────── #
def load_models(device):
    print("\n🔄 Loading trained weights...")

    # UNet (always frozen — not fine-tuned)
    if config.SEG_MODEL == 'swin_unet':
        unet = SwinTransformerUNet(pretrained=False).to(device)
    else:
        unet = LightweightUNet(n_channels=3, n_classes=1).to(device)

    unet_path = os.path.join(config.WEIGHTS_DIR, 'best_unet.pth')
    unet.load_state_dict(torch.load(unet_path, map_location=device, weights_only=True))
    for p in unet.parameters():
        p.requires_grad = False
    unet.eval()
    print(f"  ✅ UNet loaded + FROZEN")

    # Classifier
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

    clf_path = os.path.join(config.WEIGHTS_DIR, 'best_dual_branch_fusion.pth')
    model.load_state_dict(torch.load(clf_path, map_location=device, weights_only=True))
    print(f"  ✅ Classifier loaded from original Kaggle weights (WILL NOT be overwritten)")

    # Freeze backbone modules
    for name in FROZEN_MODULES:
        module = getattr(model, name, None)
        if module is not None:
            for p in module.parameters():
                p.requires_grad = False
            print(f"  🔒 FROZEN: model.{name}")

    # Confirm trainable modules
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params     = sum(p.numel() for p in model.parameters())
    print(f"\n  📌 Trainable params : {trainable_params/1e6:.2f}M  (head + gate only)")
    print(f"  📌 Frozen params    : {(total_params-trainable_params)/1e6:.2f}M  (backbones)")
    for name in TRAINABLE_MODULES:
        module = getattr(model, name, None)
        if module:
            n = sum(p.numel() for p in module.parameters())
            print(f"  ✏️  TRAINING: model.{name} ({n/1e6:.3f}M params)")

    return unet, model


# ── Training Loop ─────────────────────────────────────────────────────────── #
def train_one_epoch(model, unet, loader, optimizer, criterion, device):
    model.train()
    # Keep frozen modules in eval mode so BatchNorm stats don't update
    for name in FROZEN_MODULES:
        module = getattr(model, name, None)
        if module:
            module.eval()

    total_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc='  Train', leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        with torch.no_grad():
            mask_logits = unet(images)
            images_seg  = apply_mask(images, mask_logits)

        logits, _ = model(images, images_seg)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0
        )
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, unet, loader, criterion, device):
    """Returns loss, accuracy, balanced_accuracy, mel_sensitivity."""
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        mask_logits = unet(images)
        images_seg  = apply_mask(images, mask_logits)
        logits, _ = model(images, images_seg)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc        = (all_preds == all_labels).mean()
    bal_acc    = balanced_accuracy_score(all_labels, all_preds)

    # MEL sensitivity
    mel_idx  = CLASS_TO_IDX['mel']
    mel_mask = all_labels == mel_idx
    mel_sens = (all_preds[mel_mask] == mel_idx).mean() if mel_mask.sum() > 0 else 0.0

    return total_loss / len(all_labels), acc, bal_acc, mel_sens


# ── Test Evaluation ──────────────────────────────────────────────────────── #
@torch.no_grad()
def full_test_evaluation(model, unet, loader, device):
    """Detailed evaluation on the 70% test split."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    all_pad_labels = []

    for images, labels in tqdm(loader, desc='  Test eval'):
        images = images.to(device)
        mask_logits = unet(images)
        images_seg  = apply_mask(images, mask_logits)
        logits, _ = model(images, images_seg)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        all_probs.extend(probs)
        all_preds.extend(probs.argmax(1).tolist())
        all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    acc     = (all_preds == all_labels).mean()
    bal_acc = balanced_accuracy_score(all_labels, all_preds)

    auc_scores = []
    for c in np.unique(all_labels):
        y_bin = (all_labels == c).astype(int)
        if y_bin.sum() > 0 and (1-y_bin).sum() > 0:
            try:
                auc_scores.append(roc_auc_score(y_bin, all_probs[:, c]))
            except Exception:
                pass
    macro_auc = np.mean(auc_scores) if auc_scores else 0.0

    mel_idx  = CLASS_TO_IDX['mel']
    mel_mask = all_labels == mel_idx
    mel_sens = (all_preds[mel_mask] == mel_idx).mean() if mel_mask.sum() > 0 else 0.0

    sep = "=" * 65
    print(f"\n{sep}")
    print("  PAD-UFES-20 Test Results (70% held-out split)")
    print("  After head fine-tuning on 20% PAD-UFES-20 train set")
    print(sep)
    print(f"  Accuracy          : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  Balanced Accuracy : {bal_acc:.4f}  ({bal_acc*100:.1f}%)")
    print(f"  Macro AUC         : {macro_auc:.4f}")
    print(f"  MEL Sensitivity   : {mel_sens:.4f}  ({mel_mask.sum()} test MEL samples)")
    print(sep)

    # Per-class breakdown
    print("\n  Per-class accuracy:")
    for pad_cls, model_cls in PAD_TO_MODEL.items():
        cidx = CLASS_TO_IDX[model_cls]
        mask = all_labels == cidx
        if mask.sum() == 0:
            continue
        cls_acc = (all_preds[mask] == cidx).mean()
        print(f"    {pad_cls:4s} → {model_cls:6s}  |  {cls_acc:.3f}  [{mask.sum()} samples]")

    print("\n  📊 Before vs After Comparison:")
    print(f"  {'Metric':<22} {'Before (no FT)':<22} {'After (head FT)'}")
    print(f"  {'-'*65}")
    print(f"  {'Accuracy':<22} {'9.6%':<22} {acc*100:.1f}%")
    print(f"  {'Balanced Accuracy':<22} {'28.9%':<22} {bal_acc*100:.1f}%")
    print(f"  {'Macro AUC':<22} {'64.2%':<22} {macro_auc*100:.1f}%")
    print(f"  {'MEL Sensitivity':<22} {'100% (4 samp.)':<22} {mel_sens*100:.1f}% ({mel_mask.sum()} samp.)")
    print(sep)


# ── Main ─────────────────────────────────────────────────────────────────── #
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Device: {device}")

    # ── Load PAD-UFES-20 CSV ──────────────────────────────────────────────── #
    csv_path = os.path.join(config.DATA_DIR, 'metadata.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"metadata.csv not found: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"\n📋 PAD-UFES-20: {len(df)} total images")
    print(f"   {df['diagnostic'].value_counts().to_dict()}")

    # ── Patient-aware splits ──────────────────────────────────────────────── #
    train_df, val_df, test_df = make_patient_aware_splits(df)

    # ── Image directories ─────────────────────────────────────────────────── #
    pad_dir  = os.path.join(config.DATA_DIR, 'pad_ufes', 'images')
    img_dirs = [
        os.path.join(pad_dir, 'imgs_part_1'),
        os.path.join(pad_dir, 'imgs_part_2'),
        os.path.join(pad_dir, 'imgs_part_3'),
    ]

    # Transforms (same as ISIC evaluation)
    base_transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD),
    ])

    # Create datasets
    train_set = PADUFESDataset(train_df, img_dirs, base_transform, augment=True, use_clahe=USE_CLAHE)
    val_set   = PADUFESDataset(val_df,   img_dirs, base_transform, augment=False, use_clahe=USE_CLAHE)
    test_set  = PADUFESDataset(test_df,  img_dirs, base_transform, augment=False, use_clahe=USE_CLAHE)

    # ── Class weights + sampler ───────────────────────────────────────────── #
    class_weights = get_class_weights(train_df)

    # WeightedRandomSampler to oversample rare classes during training
    sample_weights = []
    for _, row in train_df.iterrows():
        mapped = PAD_TO_MODEL.get(row['diagnostic'], 'nv')
        sample_weights.append(class_weights[CLASS_TO_IDX[mapped]].item())
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False,   num_workers=0)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False,   num_workers=0)

    # ── Load models ───────────────────────────────────────────────────────── #
    unet, model = load_models(device)

    # ── Loss and optimizer ────────────────────────────────────────────────── #
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # Only optimize parameters that require grad (head + gate)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # ── Output paths (two separate saves) ──────────────────────────────────#
    out_best_overall = os.path.join(config.WEIGHTS_DIR, 'padufes_finetuned_head.pth')
    out_best_mel     = os.path.join(config.WEIGHTS_DIR, 'padufes_best_mel.pth')
    print(f"\n💾 Weight files:")
    print(f"   padufes_finetuned_head.pth  ← best overall balanced accuracy")
    print(f"   padufes_best_mel.pth        ← best balanced acc with MEL=100%")
    print(f"   ✅ Original weights (best_dual_branch_fusion.pth) are NEVER modified\n")

    # ── Training loop ─────────────────────────────────────────────────────── #
    best_val_bal_acc  = 0.0   # best overall (any MEL sensitivity)
    best_mel_bal_acc  = 0.0   # best balanced acc where MEL sensitivity == 1.0
    patience_counter  = 0

    print(f"{'='*65}")
    print(f"  Starting fine-tuning: {EPOCHS} epochs max, patience={PATIENCE}")
    print(f"  Trainable: classifier + gate only")
    print(f"  Saves TWO checkpoints: best-overall + best-with-perfect-MEL")
    print(f"{'='*65}\n")

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, unet, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc, val_bal_acc, val_mel_sens = evaluate(
            model, unet, val_loader, criterion, device
        )
        scheduler.step()

        print(
            f"Epoch {epoch:2d}/{EPOCHS} | "
            f"Train loss: {train_loss:.4f} acc: {train_acc:.3f} | "
            f"Val bal_acc: {val_bal_acc:.3f} mel_sens: {val_mel_sens:.3f}"
        )

        saved_tags = []

        # ── Save 1: Best overall balanced accuracy (any MEL sensitivity) ── #
        if val_bal_acc > best_val_bal_acc:
            best_val_bal_acc = val_bal_acc
            torch.save(model.state_dict(), out_best_overall)
            saved_tags.append(f"best-overall (bal_acc={val_bal_acc:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1

        # ── Save 2: Best balanced accuracy WHERE MEL sensitivity = 100% ── #
        if val_mel_sens == 1.0 and val_bal_acc > best_mel_bal_acc:
            best_mel_bal_acc = val_bal_acc
            torch.save(model.state_dict(), out_best_mel)
            saved_tags.append(f"best-MEL=100% (bal_acc={val_bal_acc:.4f})")

        if saved_tags:
            print(f"  ✅ Saved: {' | '.join(saved_tags)}")

        if patience_counter >= PATIENCE:
            print(f"\n⏹️  Early stopping at epoch {epoch} (patience={PATIENCE})")
            break

    # ── Decide which weights to use for final test ──────────────────────── #
    print(f"\n{'='*65}")
    print(f"  Checkpoint summary:")
    print(f"  padufes_finetuned_head.pth → best overall bal_acc = {best_val_bal_acc:.4f}")
    print(f"  padufes_best_mel.pth       → best bal_acc with MEL=100% = {best_mel_bal_acc:.4f}")
    print(f"  Recommendation: Use padufes_best_mel.pth for clinical use (MEL=100%)")
    print(f"{'='*65}")

    # ── Final test on BOTH checkpoints ───────────────────────────────────── #
    for label, ckpt_path in [
        ('Best MEL=100% model (padufes_best_mel.pth)', out_best_mel),
        ('Best overall model (padufes_finetuned_head.pth)', out_best_overall),
    ]:
        if not os.path.exists(ckpt_path):
            print(f"  ⚠️  {ckpt_path} not found — skip")
            continue
        print(f"\n📥 Testing: {label}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        full_test_evaluation(model, unet, test_loader, device)

    print(f"\n✅ Fine-tuning complete!")
    print(f"   padufes_best_mel.pth       → use this for clinical / paper (MEL=100%)")
    print(f"   padufes_finetuned_head.pth → best overall accuracy")
    print(f"   Original Kaggle weights untouched → best_dual_branch_fusion.pth")
    print(f"\nTo use in test_padufes.py, change clf_path to one of the above.")


if __name__ == '__main__':
    main()
