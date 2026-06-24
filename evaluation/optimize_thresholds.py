"""
Per-Class Threshold Optimization + Melanoma-Safety Operating Point
==================================================================
Two legitimate, retraining-free improvements, tuned on the VALIDATION set and
applied to the TEST set (never tuned on test — that would be leakage):

  1. MACRO-F1 MODE — per-class one-vs-rest thresholds chosen to maximize each
     class's F1 on val. Decision rule = argmax_c (prob_c / threshold_c), which
     reduces to plain argmax when all thresholds are 0.5. This fixes the low
     macro-F1 caused by extreme class imbalance (rare classes ranked well —
     high AUC — but suppressed at the default 0.5 threshold).

  2. MELANOMA-SAFETY MODE — for the clinical use case (missing a melanoma can
     kill the patient), pick the mel decision threshold that achieves a TARGET
     SENSITIVITY (default 0.95) on val, then report the resulting test
     sensitivity & specificity. This raises recall, NOT AUC — AUC is fixed by
     training. It is the correct, honest way to make the model "not miss mel".

NOTE: threshold tuning improves F1 / sensitivity, NOT AUC (AUC is
threshold-independent). Report it transparently in the paper.

Run:  PYTHONPATH=. python -m evaluation.optimize_thresholds
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
for k in list(sys.modules.keys()):
    if k == 'datasets' or k.startswith('datasets.'):
        sys.modules.pop(k)

import numpy as np
import torch
from torch.amp import autocast
from tqdm import tqdm
from sklearn.metrics import f1_score

from configs.config import config
from utils.seed import seed_everything
from datasets.unified_dataset import get_unified_dataloaders
from models.transformer_unet import SwinTransformerUNet
from models.unet import LightweightUNet
from models.dual_branch_fusion import DualBranchFusionClassifier
from training.train_utils import apply_mask


# =========================================================================== #
#                  PURE THRESHOLD LOGIC (unit-testable)                        #
# =========================================================================== #

def optimize_f1_thresholds(probs: np.ndarray, labels: np.ndarray,
                           num_classes: int, grid=None) -> np.ndarray:
    """Per-class one-vs-rest threshold that maximizes that class's F1 (on val)."""
    if grid is None:
        grid = np.arange(0.05, 0.95, 0.01)
    thresholds = np.full(num_classes, 0.5, dtype=np.float64)
    for c in range(num_classes):
        y = (labels == c).astype(int)
        if y.sum() == 0:
            continue
        best_t, best_f1 = 0.5, -1.0
        for t in grid:
            f1 = f1_score(y, (probs[:, c] > t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        thresholds[c] = best_t
    return thresholds


def predict_with_thresholds(probs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Multi-class decision: argmax of (prob_c / threshold_c).

    Dividing by the per-class threshold up-weights classes with a low optimal
    threshold (the rare ones). Reduces to plain argmax(probs) when all
    thresholds are equal — a clean, single-label, defensible rule.
    """
    scaled = probs / thresholds[None, :].clip(min=1e-6)
    return scaled.argmax(axis=1)


def sensitivity_target_threshold(probs_c: np.ndarray, y_c: np.ndarray,
                                  target_sens: float = 0.95) -> float:
    """Largest one-vs-rest threshold for class c that still achieves
    sensitivity (recall) >= target on val. Higher threshold = fewer false
    positives while keeping the required recall."""
    pos = probs_c[y_c == 1]
    if len(pos) == 0:
        return 0.5
    # To catch >= target fraction of positives, threshold must be <= the
    # (1-target) quantile of positive scores.
    thr = float(np.quantile(pos, 1.0 - target_sens))
    return min(max(thr, 1e-4), 0.999)


def _sens_spec(probs_c, y_c, t):
    pred = (probs_c >= t).astype(int)
    tp = int(((pred == 1) & (y_c == 1)).sum()); fn = int(((pred == 0) & (y_c == 1)).sum())
    tn = int(((pred == 0) & (y_c == 0)).sum()); fp = int(((pred == 1) & (y_c == 0)).sum())
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return sens, spec


# =========================================================================== #
#                          INFERENCE (n_views=1)                               #
# =========================================================================== #

@torch.no_grad()
def infer(model, unet, loader, device):
    model.eval(); unet.eval()
    all_labels, all_probs = [], []
    for batch in tqdm(loader, desc="infer"):
        images = batch['image'].to(device)
        labels = batch['label']
        with autocast('cuda', enabled=(device == 'cuda')):
            mask_logits = unet(images)
            images_seg = apply_mask(images, mask_logits)
            mask_prob = torch.sigmoid(mask_logits.float())
            logits, _ = model(images, images_seg, mask_prob)
        all_probs.append(torch.softmax(logits, dim=1).float().cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_labels), np.concatenate(all_probs)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-sala", action="store_true", dest="no_sala",
                    help="use the final no-SALA model (best_dual_branch_fusion_nosala.pth)")
    args = ap.parse_args()
    tag = "_nosala" if args.no_sala else ""

    seed_everything(config.SEED)
    dev = config.DEVICE
    C = config.NUM_CLASSES
    mel_idx = config.CLASSES.index('mel') if 'mel' in config.CLASSES else 4

    _, val_loader, test_loader, _ = get_unified_dataloaders(
        config.DATA_DIR, masks_dir=os.path.join(config.DATA_DIR, "masks"))

    unet = (SwinTransformerUNet(pretrained=False) if config.SEG_MODEL == 'swin_unet'
            else LightweightUNet(3, 1)).to(dev)
    unet.load_state_dict(torch.load(os.path.join(config.WEIGHTS_DIR, "best_unet.pth"),
                                    map_location=dev, weights_only=True))
    model = DualBranchFusionClassifier(
        eva02_name=config.EVA02_BACKBONE, eva02_pretrained=False,
        convnext_name=config.CONVNEXT_BACKBONE, convnext_pretrained=False,
        fusion_dim=config.FUSION_EMBED_DIM, num_heads=config.FUSION_NUM_HEADS,
        num_classes=C, dropout=config.FUSION_DROPOUT,
        use_spatial_fusion=getattr(config, 'USE_SPATIAL_FUSION', True),
        fusion_grid=getattr(config, 'FUSION_GRID', 14)).to(dev)
    model.load_state_dict(torch.load(
        os.path.join(config.WEIGHTS_DIR, f"best_dual_branch_fusion{tag}.pth"),
        map_location=dev, weights_only=True))
    print(f"[thresholds] using classifier weights: best_dual_branch_fusion{tag}.pth")

    print("\n[1/2] Inference on VALIDATION (for threshold tuning)...")
    val_labels, val_probs = infer(model, unet, val_loader, dev)
    print("[2/2] Inference on TEST (for final report)...")
    test_labels, test_probs = infer(model, unet, test_loader, dev)

    # ---- Macro-F1 mode ----
    base_pred = test_probs.argmax(1)
    base_f1 = f1_score(test_labels, base_pred, average='macro', zero_division=0)
    thr = optimize_f1_thresholds(val_probs, val_labels, C)
    opt_pred = predict_with_thresholds(test_probs, thr)
    opt_f1 = f1_score(test_labels, opt_pred, average='macro', zero_division=0)

    print("\n" + "=" * 60)
    print("  MACRO-F1 (thresholds tuned on val, applied to test)")
    print("=" * 60)
    print(f"  Default argmax macro-F1 : {base_f1:.4f}")
    print(f"  Threshold-optimized     : {opt_f1:.4f}")
    print("  Per-class thresholds    :",
          {config.CLASSES[c]: round(float(thr[c]), 2) for c in range(C)})

    # ---- Melanoma-safety mode ----
    yv = (val_labels == mel_idx).astype(int)
    yt = (test_labels == mel_idx).astype(int)
    t_def = 0.5
    sens_def, spec_def = _sens_spec(test_probs[:, mel_idx], yt, t_def)
    t_safe = sensitivity_target_threshold(val_probs[:, mel_idx], yv, target_sens=0.95)
    sens_safe, spec_safe = _sens_spec(test_probs[:, mel_idx], yt, t_safe)

    print("\n" + "=" * 60)
    print("  MELANOMA SAFETY OPERATING POINT (≥95% sensitivity target on val)")
    print("=" * 60)
    print(f"  Default thr=0.50 : sensitivity {sens_def:.3f}  specificity {spec_def:.3f}")
    print(f"  Mel-safe thr={t_safe:.3f} : sensitivity {sens_safe:.3f}  specificity {spec_safe:.3f}")
    print("  → Use the mel-safe threshold clinically: catches more melanomas")
    print("    at the cost of more false positives (extra biopsies). AUC unchanged.")
    print("=" * 60)


if __name__ == "__main__":
    main()
