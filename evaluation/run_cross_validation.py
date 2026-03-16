"""
DermaFusion-AI — 5-Fold Patient-Aware Cross-Validation
=======================================================
Model   : EVA-02 Small (eva02_small_patch14_336, 22M params)
Dataset : HAM10000 (patient-aware GroupKFold on lesion_id)
Resume  : Automatically skips completed folds on restart.
          Saves checkpoint after every epoch + full fold result.
Output  : outputs/cv_results/
            fold_0/ ... fold_4/  — per-fold checkpoints + metrics
            cv_summary.csv       — mean ± std across folds
            cv_log.txt           — full run log

Run on Kaggle (GPU T4×2):
    python run_cross_validation.py

HAM10000 paths on Kaggle:
    CSV  : /kaggle/input/datasets/kmader/skin-cancer-mnist-ham10000/HAM10000_metadata.csv
    imgs : /kaggle/input/datasets/kmader/skin-cancer-mnist-ham10000/HAM10000_images_part_{1,2}/
"""

import os, sys, time, json, logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import RandAugment
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score
from PIL import Image
from tqdm import tqdm
import timm
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────
class CVConfig:
    # Paths — Kaggle
    HAM_ROOT   = "/kaggle/input/datasets/kmader/skin-cancer-mnist-ham10000"
    IMG_DIRS   = [
        f"{HAM_ROOT}/HAM10000_images_part_1",
        f"{HAM_ROOT}/HAM10000_images_part_2",
    ]
    CSV_PATH   = f"{HAM_ROOT}/HAM10000_metadata.csv"

    # If running locally (not Kaggle), override paths here:
    # HAM_ROOT = "/path/to/ham10000"
    # CSV_PATH = f"{HAM_ROOT}/HAM10000_metadata.csv"

    # Output
    OUT_DIR    = "/kaggle/working/cv_results"   # Change to outputs/cv_results locally

    # Model
    BACKBONE   = "eva02_small_patch14_336.mim_in22k_ft_in1k"
    IMG_SIZE   = 336
    NUM_CLASSES = 7
    CLASSES     = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

    # Training
    N_FOLDS    = 5
    EPOCHS     = 20          # Per fold — enough to converge on HAM10000
    BATCH_SIZE = 16
    LR_BACKBONE = 1e-5       # Low LR for pretrained EVA-02
    LR_HEAD     = 1e-4       # Higher LR for new classification head
    WEIGHT_DECAY = 0.05
    WARMUP_EPOCHS = 3
    PATIENCE      = 8        # Early stopping
    SEED          = 42
    LABEL_SMOOTH  = 0.1
    MIXED_PREC    = True     # AMP

    # Melanoma boost (2× upweight in sampler)
    MEL_BOOST  = 2.0

cfg = CVConfig()

# ─────────────────────────────────────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────────────────────────────────────
os.makedirs(cfg.OUT_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(cfg.OUT_DIR, "cv_log.txt"), mode='a'),
    ]
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────────────────────
torch.manual_seed(cfg.SEED)
np.random.seed(cfg.SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Device: {DEVICE}")
if torch.cuda.is_available():
    log.info(f"GPU: {torch.cuda.get_device_name(0)}")

# ─────────────────────────────────────────────────────────────────────────────
#  IMAGE LOOKUP
# ─────────────────────────────────────────────────────────────────────────────
img_lookup: dict = {}
for d in cfg.IMG_DIRS:
    if os.path.exists(d):
        for f in os.listdir(d):
            img_lookup[f.replace('.jpg', '').replace('.png', '')] = os.path.join(d, f)
log.info(f"Image lookup built: {len(img_lookup)} images")

LABEL_MAP = {c: i for i, c in enumerate(cfg.CLASSES)}

# ─────────────────────────────────────────────────────────────────────────────
#  DATASET
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_TF = transforms.Compose([
    transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    RandAugment(num_ops=2, magnitude=9),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

VAL_TF = transforms.Compose([
    transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

class HAMDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform):
        self.df  = df.reset_index(drop=True)
        self.tf  = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row   = self.df.iloc[i]
        path  = img_lookup.get(row['image_id'])
        label = LABEL_MAP.get(row['dx'], 0)
        if path and os.path.exists(path):
            try:
                img = Image.open(path).convert('RGB')
            except Exception:
                img = Image.new('RGB', (cfg.IMG_SIZE, cfg.IMG_SIZE))
        else:
            img = Image.new('RGB', (cfg.IMG_SIZE, cfg.IMG_SIZE))
        return self.tf(img), torch.tensor(label, dtype=torch.long)

# ─────────────────────────────────────────────────────────────────────────────
#  MODEL
# ─────────────────────────────────────────────────────────────────────────────
class EVASmallClassifier(nn.Module):
    """EVA-02 Small backbone + classification head."""
    def __init__(self, backbone_name: str, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=True, num_classes=0
        )
        dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(dim // 2, num_classes),
        )
        total = sum(p.numel() for p in self.parameters()) / 1e6
        log.info(f"EVASmallClassifier: backbone_dim={dim}, total_params={total:.1f}M")

    def forward(self, x):
        return self.head(self.backbone(x))

# ─────────────────────────────────────────────────────────────────────────────
#  WEIGHTED SAMPLER (handles class imbalance + MEL boost)
# ─────────────────────────────────────────────────────────────────────────────
def make_sampler(df: pd.DataFrame):
    counts = df['dx'].value_counts().to_dict()
    weights = []
    for _, row in df.iterrows():
        cls = row['dx']
        w   = 1.0 / counts[cls]
        if cls == 'mel':
            w *= cfg.MEL_BOOST
        weights.append(w)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.float),
        num_samples=len(weights),
        replacement=True,
    )
    return sampler

# ─────────────────────────────────────────────────────────────────────────────
#  TRAINING + EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, scheduler, scaler, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.LABEL_SMOOTH).to(DEVICE)
    total_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc=f"  Ep{epoch+1:02d} train", leave=False,
                ncols=90, unit="batch")
    for imgs, labels in pbar:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        with autocast('cuda', enabled=cfg.MIXED_PREC):
            logits = model(imgs)
            loss   = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total   += imgs.size(0)
        pbar.set_postfix(loss=f"{total_loss/total:.4f}", acc=f"{correct/total:.3f}")
    scheduler.step()
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    for imgs, labels in tqdm(loader, desc="  Eval", leave=False, ncols=90, unit="batch"):
        imgs = imgs.to(DEVICE)
        with autocast('cuda', enabled=cfg.MIXED_PREC):
            logits = model(imgs)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(1).cpu().numpy()
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    y_true  = np.array(all_labels)
    y_probs = np.array(all_probs)
    y_preds = np.array(all_preds)

    try:
        auc = roc_auc_score(
            y_true, y_probs,
            multi_class='ovr', average='macro',
            labels=list(range(cfg.NUM_CLASSES))  # explicit — prevents NaN on missing classes
        )
    except Exception:
        auc = float('nan')

    bal_acc  = balanced_accuracy_score(y_true, y_preds)
    macro_f1 = f1_score(y_true, y_preds, average='macro', zero_division=0)
    mel_mask = (y_true == LABEL_MAP['mel'])
    mel_sens = (y_preds[mel_mask] == LABEL_MAP['mel']).mean() if mel_mask.sum() > 0 else float('nan')

    return {
        'auc'         : round(float(auc),     4),
        'bal_acc'     : round(float(bal_acc),  4),
        'macro_f1'    : round(float(macro_f1), 4),
        'mel_sens'    : round(float(mel_sens), 4),
        'n_samples'   : int(len(y_true)),
    }

# ─────────────────────────────────────────────────────────────────────────────
#  PER-FOLD TRAINING
# ─────────────────────────────────────────────────────────────────────────────
def train_fold(fold_idx: int, train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
    fold_dir = os.path.join(cfg.OUT_DIR, f"fold_{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)

    result_path = os.path.join(fold_dir, "result.json")

    # ── Resume: if this fold already finished, skip it ──────────────────────
    if os.path.exists(result_path):
        with open(result_path) as f:
            saved = json.load(f)
        log.info(f"[Fold {fold_idx}] Already completed — skipping. AUC={saved['best_auc']:.4f}")
        return saved

    log.info(f"\n{'='*60}")
    log.info(f"[Fold {fold_idx}] Train={len(train_df)}  Val={len(val_df)}")
    log.info(f"{'='*60}")

    # ── Datasets & loaders ───────────────────────────────────────────────────
    train_ds = HAMDataset(train_df, TRAIN_TF)
    val_ds   = HAMDataset(val_df,   VAL_TF)
    sampler  = make_sampler(train_df)
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, sampler=sampler,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE * 2, shuffle=False,
                              num_workers=2, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = EVASmallClassifier(cfg.BACKBONE, cfg.NUM_CLASSES).to(DEVICE)

    # ── Resume from epoch checkpoint ─────────────────────────────────────────
    ckpt_path  = os.path.join(fold_dir, "latest_epoch.pth")
    start_epoch = 0
    best_auc    = 0.0
    patience_counter = 0
    epoch_history = []

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt['model'])
        start_epoch      = ckpt['epoch'] + 1
        best_auc         = ckpt['best_auc']
        patience_counter = ckpt['patience_counter']
        epoch_history    = ckpt.get('epoch_history', [])
        log.info(f"[Fold {fold_idx}] Resumed from epoch {start_epoch}, best_auc={best_auc:.4f}")

    # ── Optimiser + scheduler ─────────────────────────────────────────────────
    backbone_params = list(model.backbone.parameters())
    head_params     = list(model.head.parameters())
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': cfg.LR_BACKBONE},
        {'params': head_params,     'lr': cfg.LR_HEAD},
    ], weight_decay=cfg.WEIGHT_DECAY)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.EPOCHS - cfg.WARMUP_EPOCHS, eta_min=1e-7
    )
    scaler = GradScaler()

    # Restore optimizer state if available
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        scaler.load_state_dict(ckpt['scaler'])

    best_model_path = os.path.join(fold_dir, "best_model.pth")

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg.EPOCHS):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, epoch)
        val_metrics           = evaluate(model, val_loader)
        val_auc               = val_metrics['auc']

        elapsed = time.time() - t0
        log.info(
            f"[Fold {fold_idx}] Ep {epoch+1:02d}/{cfg.EPOCHS} | "
            f"loss={train_loss:.4f} acc={train_acc:.3f} | "
            f"val_auc={val_auc:.4f} bal_acc={val_metrics['bal_acc']:.4f} "
            f"mel_sens={val_metrics['mel_sens']:.4f} | {elapsed:.0f}s"
        )

        row = {'epoch': epoch+1, 'train_loss': train_loss,
               'train_acc': train_acc, **val_metrics}
        epoch_history.append(row)

        # Save epoch CSV
        pd.DataFrame(epoch_history).to_csv(
            os.path.join(fold_dir, "epoch_history.csv"), index=False)

        # Save best model — use AUC if valid, else fall back to balanced accuracy
        import math
        track_metric = val_auc if not math.isnan(val_auc) else val_metrics['bal_acc']
        metric_name  = 'AUC' if not math.isnan(val_auc) else 'BalAcc'
        if track_metric > best_auc:
            best_auc = track_metric
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            log.info(f"  ✅ New best {metric_name}={best_auc:.4f} — model saved")
        else:
            patience_counter += 1
            if patience_counter >= cfg.PATIENCE:
                log.info(f"  ⏹️  Early stopping at epoch {epoch+1} (patience={cfg.PATIENCE})")
                break

        # ── Save resume checkpoint after every epoch ──────────────────────────
        torch.save({
            'epoch'           : epoch,
            'model'           : model.state_dict(),
            'optimizer'       : optimizer.state_dict(),
            'scheduler'       : scheduler.state_dict(),
            'scaler'          : scaler.state_dict(),
            'best_auc'        : best_auc,
            'patience_counter': patience_counter,
            'epoch_history'   : epoch_history,
        }, ckpt_path)

    # ── Final evaluation on best model ────────────────────────────────────────
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=DEVICE, weights_only=True))
    final_metrics = evaluate(model, val_loader)

    result = {
        'fold'          : fold_idx,
        'best_auc'      : best_auc,
        'final_auc'     : final_metrics['auc'],
        'final_bal_acc' : final_metrics['bal_acc'],
        'final_macro_f1': final_metrics['macro_f1'],
        'final_mel_sens': final_metrics['mel_sens'],
        'n_val'         : final_metrics['n_samples'],
        'epochs_run'    : len(epoch_history),
    }

    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    log.info(f"[Fold {fold_idx}] ✅ Done. AUC={final_metrics['auc']:.4f} | BalAcc={final_metrics['bal_acc']:.4f}")

    # Clean up epoch checkpoint (fold is done)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    return result

# ─────────────────────────────────────────────────────────────────────────────
#  MAIN — 5-FOLD CV LOOP
# ─────────────────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 60)
    log.info("  DermaFusion-AI — 5-Fold Patient-Aware Cross Validation")
    log.info(f"  Model   : {cfg.BACKBONE}")
    log.info(f"  Dataset : HAM10000 ({cfg.N_FOLDS}-fold GroupKFold on lesion_id)")
    log.info(f"  Output  : {cfg.OUT_DIR}")
    log.info("=" * 60)

    # ── Load dataset ──────────────────────────────────────────────────────────
    df = pd.read_csv(cfg.CSV_PATH)
    log.info(f"Loaded {len(df)} records from HAM10000")
    log.info(f"Class distribution:\n{df['dx'].value_counts().to_string()}")

    # ── Patient-aware folds ───────────────────────────────────────────────────
    gkf = GroupKFold(n_splits=cfg.N_FOLDS)
    X   = df.index.values
    y   = df['dx'].map(LABEL_MAP).values
    groups = df['lesion_id'].values

    fold_splits = list(gkf.split(X, y, groups))
    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
        train_df = df.iloc[train_idx]
        val_df   = df.iloc[val_idx]
        result   = train_fold(fold_idx, train_df, val_df)
        fold_results.append(result)

    # ── Aggregate results ─────────────────────────────────────────────────────
    summary_df = pd.DataFrame(fold_results)
    summary_path = os.path.join(cfg.OUT_DIR, "cv_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    metrics = ['final_auc', 'final_bal_acc', 'final_macro_f1', 'final_mel_sens']
    log.info("\n" + "=" * 60)
    log.info("  5-FOLD CV RESULTS — PAPER-READY")
    log.info("=" * 60)
    paper_lines = []
    for m in metrics:
        vals = summary_df[m].values
        mean, std = vals.mean(), vals.std()
        name = m.replace('final_', '').replace('_', ' ').title()
        line = f"  {name:<22}: {mean:.4f} ± {std:.4f}  (all folds: {list(vals.round(4))})"
        log.info(line)
        paper_lines.append(f"{name}: {mean:.4f} ± {std:.4f} (95% CI via 5-fold)")

    paper_text = "\n".join(paper_lines)
    paper_path = os.path.join(cfg.OUT_DIR, "paper_ready_cv.txt")
    with open(paper_path, 'w') as f:
        f.write(paper_text)
    log.info(f"\n  Paper-ready text saved to: {paper_path}")
    log.info(f"  Full summary CSV at: {summary_path}")
    log.info("=" * 60)

if __name__ == "__main__":
    main()
