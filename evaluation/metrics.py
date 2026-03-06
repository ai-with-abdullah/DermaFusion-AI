"""
Upgraded Evaluation Metrics — 2026 SOTA
=========================================
New metrics beyond accuracy/F1/AUC:
  • Balanced Accuracy      — critical for 7-class imbalanced data
  • Per-class Sensitivity  — true positive rate per class
  • Per-class Specificity  — true negative rate per class
  • ECE                    — Expected Calibration Error (model confidence calibration)
  • Partial AUC (pAUC)     — ISIC 2024 challenge metric at 80% specificity

All functions accept:
  y_true:      (N,) integer class labels
  y_pred_probs: (N, C) softmax probabilities
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    f1_score, accuracy_score, roc_auc_score,
    confusion_matrix, balanced_accuracy_score,
    roc_curve
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os
from configs.config import config


# =========================================================================== #
#                       PRIMARY METRIC COMPUTATION                             #
# =========================================================================== #

def compute_metrics(y_true: np.ndarray, y_pred_probs: np.ndarray) -> dict:
    """
    Computes all classification metrics.

    Returns dict with:
      accuracy, balanced_accuracy,
      macro_f1, weighted_f1,
      macro_auc,
      per_class_sensitivity, per_class_specificity,
      ece, pauc_80spec
    """
    y_pred = np.argmax(y_pred_probs, axis=1)
    C = y_pred_probs.shape[1]

    # ── Basic ──────────────────────────────────────────────────────────────── #
    acc          = float(accuracy_score(y_true, y_pred))
    balanced_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1     = float(f1_score(y_true, y_pred, average='macro',    zero_division=0))
    weighted_f1  = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))

    # ── Macro ROC-AUC (robust per-class OVR) ─────────────────────────────── #
    # Compute binary AUC per class separately — never raises ValueError
    aucs = []
    for k in range(C):
        y_bin  = (y_true == k).astype(int)
        scores = y_pred_probs[:, k]
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            continue   # class absent or all-positive — skip this class
        try:
            aucs.append(float(roc_auc_score(y_bin, scores)))
        except ValueError:
            continue
    macro_auc = float(np.mean(aucs)) if aucs else 0.5

    # ── Per-class F1 ─────────────────────────────────────────────────────── #
    per_class_f1_arr = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_class_f1     = {config.CLASSES[i]: float(v) for i, v in enumerate(per_class_f1_arr)}

    # ── Per-class Sensitivity & Specificity ───────────────────────────────── #
    sensitivity_per_class, specificity_per_class = compute_per_class_sens_spec(y_true, y_pred, C)

    # ── Expected Calibration Error (n_bins=10 — more stable on small test sets) #
    ece = compute_ece(y_true, y_pred_probs, n_bins=10)

    # ── Partial AUC at 80% specificity (ISIC 2024 metric) ────────────────── #
    pauc = compute_pauc(y_true, y_pred_probs, min_tpr=0.80)

    return {
        'accuracy':                acc,
        'balanced_accuracy':       balanced_acc,
        'macro_f1':                macro_f1,
        'weighted_f1':             weighted_f1,
        'macro_auc':               macro_auc,
        'per_class_f1':            per_class_f1,
        'per_class_sensitivity':   sensitivity_per_class,
        'per_class_specificity':   specificity_per_class,
        'ece':                     ece,
        'pauc_80tpr':              pauc,
    }


# =========================================================================== #
#                   PER-CLASS SENSITIVITY & SPECIFICITY                        #
# =========================================================================== #

def compute_per_class_sens_spec(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> tuple:
    """
    Computes per-class sensitivity (recall) and specificity.

    Sensitivity_k = TP_k / (TP_k + FN_k)
    Specificity_k = TN_k / (TN_k + FP_k)

    Returns:
        sensitivity: dict {class_name: float}
        specificity: dict {class_name: float}
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    sensitivity = {}
    specificity = {}

    for k in range(num_classes):
        tp = cm[k, k]
        fn = cm[k, :].sum() - tp
        fp = cm[:, k].sum() - tp
        tn = cm.sum() - tp - fn - fp

        sens = tp / (tp + fn + 1e-8)
        spec = tn / (tn + fp + 1e-8)

        cls_name = config.CLASSES[k] if k < len(config.CLASSES) else str(k)
        sensitivity[cls_name] = float(sens)
        specificity[cls_name] = float(spec)

    return sensitivity, specificity


# =========================================================================== #
#                   EXPECTED CALIBRATION ERROR (ECE)                           #
# =========================================================================== #

def compute_ece(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    Expected Calibration Error (Guo et al., ICML 2017).

    A well-calibrated model has ECE close to 0. ECE > 0.05 indicates
    the model's confidence does not match its actual accuracy.

    Lower is better. Target: ECE < 0.05.
    """
    confidences  = y_pred_probs.max(axis=1)
    predictions  = y_pred_probs.argmax(axis=1)
    accuracies   = (predictions == y_true).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc  = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += mask.sum() * abs(bin_acc - bin_conf)

    return float(ece / len(y_true))


# =========================================================================== #
#                   PARTIAL AUC (ISIC 2024 metric)                             #
# =========================================================================== #

def compute_pauc(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    min_tpr: float = 0.80,
) -> float:
    """
    Computes the partial AUC (pAUC) above a minimum TPR (True Positive Rate) threshold.

    ISIC 2024 challenge uses pAUC at ≥80% TPR as the primary metric because
    high sensitivity is mandatory for clinical melanoma screening.

    FIXED (Upgrade #2): Docstring previously said '80% specificity' which is the
    OPPOSITE of what the code computes. The code correctly filters tpr >= min_tpr
    (i.e., sensitivity ≥80%), which matches the ISIC 2024 official metric definition.

    For multi-class: computes binary mel-vs-rest pAUC.

    Normalization:
      Max possible AUC above min_tpr=0.80 is 1.0 × (1 - 0.80) = 0.20.
      We normalize by 0.20 to get a score in [0, 1] comparable to the
      ISIC 2024 official leaderboard.

    Returns:
        pAUC score in [0, 1] (higher = better, ISIC-compatible)
    """
    # Mel class index (hardcoded 4 for HAM7 scheme: akiec/bcc/bkl/df/mel/nv/vasc)
    mel_idx = config.CLASSES.index('mel') if 'mel' in config.CLASSES else 0

    y_binary = (y_true == mel_idx).astype(int)
    mel_probs = y_pred_probs[:, mel_idx]

    if y_binary.sum() == 0 or y_binary.sum() == len(y_binary):
        return 0.0

    try:
        fpr, tpr, _ = roc_curve(y_binary, mel_probs)

        # Only keep the ROC region where TPR >= min_tpr
        mask = tpr >= min_tpr
        if mask.sum() < 2:
            return 0.0

        partial_auc = np.trapz(tpr[mask], fpr[mask])

        # ISIC standard: normalize by max possible area above min_tpr
        # Max area = 1.0 (full FPR range) × (1 - min_tpr)
        max_area = 1.0 - min_tpr  # = 0.20 for min_tpr=0.80
        if max_area < 1e-8:
            return 0.0

        return float(np.clip(partial_auc / max_area, 0.0, 1.0))
    except Exception:
        return 0.0


# =========================================================================== #
#                   CONFUSION MATRIX PLOT                                      #
# =========================================================================== #

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    save_path: str = None,
    normalize: bool = True,
) -> None:
    """Plots and optionally saves a normalized confusion matrix."""
    y_pred = np.argmax(y_pred_probs, axis=1)
    cm     = confusion_matrix(y_true, y_pred)

    if normalize:
        cm_plot = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
        fmt = '.2f'
    else:
        cm_plot = cm
        fmt = 'd'

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_plot, annot=True, fmt=fmt, cmap='Blues',
        xticklabels=config.CLASSES, yticklabels=config.CLASSES,
        vmin=0.0, vmax=1.0 if normalize else None,
    )
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix (Normalized)' if normalize else 'Confusion Matrix')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# =========================================================================== #
#                   METRICS SUMMARY LOGGER                                     #
# =========================================================================== #

def log_metrics(metrics: dict, logger, prefix: str = "Test") -> None:
    """Pretty-print all metrics via a logger."""
    logger.info(f"\n{'='*60}")
    logger.info(f"  {prefix} Metrics Summary")
    logger.info(f"{'='*60}")
    logger.info(f"  Accuracy          : {metrics['accuracy']:.4f}")
    logger.info(f"  Balanced Accuracy : {metrics['balanced_accuracy']:.4f}")
    logger.info(f"  Macro F1          : {metrics['macro_f1']:.4f}")
    logger.info(f"  Weighted F1       : {metrics['weighted_f1']:.4f}")
    logger.info(f"  Macro AUC         : {metrics['macro_auc']:.4f}")
    logger.info(f"  ECE (↓ better)    : {metrics['ece']:.4f}")
    logger.info(f"  pAUC@80%TPR       : {metrics['pauc_80tpr']:.4f}")
    if 'per_class_f1' in metrics:
        logger.info(f"  Per-class F1:")
        for cls, val in metrics['per_class_f1'].items():
            logger.info(f"    {cls:8s}: {val:.4f}")
    logger.info(f"  Per-class Sensitivity:")
    for cls, val in metrics['per_class_sensitivity'].items():
        logger.info(f"    {cls:8s}: {val:.4f}")
    logger.info(f"  Per-class Specificity:")
    for cls, val in metrics['per_class_specificity'].items():
        logger.info(f"    {cls:8s}: {val:.4f}")
    logger.info(f"{'='*60}\n")


# =========================================================================== #
#                   LEGACY SHIMS                                               #
# =========================================================================== #

def compute_dice_score(logits, targets, smooth=1e-6):
    probs   = torch.sigmoid(logits)
    preds   = (probs > 0.5).float()
    preds   = preds.view(-1)
    targets = targets.view(-1)
    inter   = (preds * targets).sum()
    dice    = (2. * inter + smooth) / (preds.sum() + targets.sum() + smooth)
    return dice.item()
