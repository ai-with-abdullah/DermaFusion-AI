"""
Bootstrap Confidence Intervals for AUC and All Key Metrics
============================================================
Computes 95% confidence intervals via bootstrap resampling (N=2000).
Produces paper-ready numbers like: AUC = 0.9908 (95% CI: 0.9881–0.9934)

Usage:
    python -m evaluation.run_confidence_intervals

Requirements:
    • outputs/val_predictions.npz   (pre-saved predictions, if available)
    OR runs fresh inference if not found.

Output:
    • Printed table of all metrics with 95% CI
    • outputs/confidence_intervals.csv
"""

import os
import sys

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

import numpy as np
import pandas as pd
import torch
from torch.amp import autocast
from tqdm import tqdm
from scipy import stats

from configs.config import config
from utils.seed import seed_everything
from utils.logger import setup_logger
from datasets.unified_dataset import get_unified_dataloaders
from models.transformer_unet import SwinTransformerUNet
from models.dual_branch_fusion import DualBranchFusionClassifier
from training.train_utils import apply_mask
from training.tta import TTAInference
from evaluation.metrics import compute_metrics


# =========================================================================== #
#                       BOOTSTRAP CI FUNCTIONS                                 #
# =========================================================================== #

def bootstrap_auc_ci(y_true: np.ndarray, y_pred_probs: np.ndarray,
                     n_bootstrap: int = 2000, ci: float = 0.95,
                     seed: int = 42) -> dict:
    """
    Bootstrap confidence interval for macro AUC.

    Args:
        y_true:       Ground-truth labels (N,)
        y_pred_probs: Softmax probabilities (N, C)
        n_bootstrap:  Number of bootstrap iterations (2000 recommended for papers)
        ci:           Confidence level (0.95 = 95% CI)
        seed:         Random seed for reproducibility

    Returns:
        dict with keys: mean, ci_low, ci_high, std, n_bootstrap
    """
    rng = np.random.default_rng(seed)
    n   = len(y_true)
    auc_samples = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)          # bootstrap resample with replacement
        y_boot  = y_true[idx]
        pr_boot = y_pred_probs[idx]

        # Per-class one-vs-rest AUC (skip absent classes)
        from sklearn.metrics import roc_auc_score
        aucs = []
        C = pr_boot.shape[1]
        for k in range(C):
            y_bin = (y_boot == k).astype(int)
            if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
                continue
            try:
                aucs.append(float(roc_auc_score(y_bin, pr_boot[:, k])))
            except ValueError:
                continue
        if aucs:
            auc_samples.append(float(np.mean(aucs)))

    auc_samples = np.array(auc_samples)
    alpha = 1.0 - ci
    ci_low  = float(np.percentile(auc_samples, 100 * alpha / 2))
    ci_high = float(np.percentile(auc_samples, 100 * (1 - alpha / 2)))

    return {
        "mean":        float(np.mean(auc_samples)),
        "ci_low":      ci_low,
        "ci_high":     ci_high,
        "std":         float(np.std(auc_samples)),
        "n_bootstrap": n_bootstrap,
        "ci_level":    ci,
    }


def bootstrap_metric_ci(y_true: np.ndarray, y_pred_probs: np.ndarray,
                         metric_fn, n_bootstrap: int = 2000,
                         ci: float = 0.95, seed: int = 42) -> dict:
    """
    Generic bootstrap CI for any scalar metric function.

    Args:
        metric_fn: callable(y_true, y_pred_probs) -> float

    Returns:
        dict with mean, ci_low, ci_high, std
    """
    rng = np.random.default_rng(seed)
    n   = len(y_true)
    values = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        try:
            v = metric_fn(y_true[idx], y_pred_probs[idx])
            values.append(float(v))
        except Exception:
            continue

    values = np.array(values)
    alpha  = 1.0 - ci
    return {
        "mean":    float(np.mean(values)),
        "ci_low":  float(np.percentile(values, 100 * alpha / 2)),
        "ci_high": float(np.percentile(values, 100 * (1 - alpha / 2))),
        "std":     float(np.std(values)),
    }


def compute_all_metrics_with_ci(y_true: np.ndarray, y_pred_probs: np.ndarray,
                                 n_bootstrap: int = 2000) -> pd.DataFrame:
    """
    Compute all key metrics with 95% CI for paper reporting.

    Returns a DataFrame with columns: Metric, Value, CI_Low, CI_High, CI_String
    CI_String is formatted as "0.9908 (95% CI: 0.9881–0.9934)" for copy-paste.
    """
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
    )

    # ── Point estimates ─────────────────────────────────────────────────── #
    y_pred = y_pred_probs.argmax(axis=1)

    # Macro AUC
    auc_ci   = bootstrap_auc_ci(y_true, y_pred_probs, n_bootstrap=n_bootstrap)

    # Balanced accuracy
    bal_acc_ci = bootstrap_metric_ci(
        y_true, y_pred_probs,
        metric_fn=lambda yt, yp: balanced_accuracy_score(yt, yp.argmax(axis=1)),
        n_bootstrap=n_bootstrap,
    )

    # Macro F1
    mac_f1_ci = bootstrap_metric_ci(
        y_true, y_pred_probs,
        metric_fn=lambda yt, yp: f1_score(yt, yp.argmax(axis=1),
                                           average="macro", zero_division=0),
        n_bootstrap=n_bootstrap,
    )

    # MEL sensitivity (recall for class index 4)
    mel_idx = config.CLASSES.index("mel") if "mel" in config.CLASSES else 4

    def mel_sensitivity(yt, yp):
        y_pred_c = yp.argmax(axis=1)
        mask = (yt == mel_idx)
        if mask.sum() == 0:
            return float("nan")
        return float((y_pred_c[mask] == mel_idx).mean())

    mel_sens_ci = bootstrap_metric_ci(
        y_true, y_pred_probs,
        metric_fn=mel_sensitivity,
        n_bootstrap=n_bootstrap,
    )

    # Weighted F1
    wt_f1_ci = bootstrap_metric_ci(
        y_true, y_pred_probs,
        metric_fn=lambda yt, yp: f1_score(yt, yp.argmax(axis=1),
                                           average="weighted", zero_division=0),
        n_bootstrap=n_bootstrap,
    )

    # ── Assemble results ─────────────────────────────────────────────────── #
    rows = []
    for name, result, fmt in [
        ("Macro AUC",            auc_ci,      ".4f"),
        ("Balanced Accuracy",    bal_acc_ci,  ".4f"),
        ("Macro F1",             mac_f1_ci,   ".4f"),
        ("Weighted F1",          wt_f1_ci,    ".4f"),
        ("MEL Sensitivity",      mel_sens_ci, ".4f"),
    ]:
        ci_str = (
            f"{result['mean']:{fmt}} "
            f"(95% CI: {result['ci_low']:{fmt}}–{result['ci_high']:{fmt}})"
        )
        rows.append({
            "Metric":    name,
            "Value":     round(result["mean"], 6),
            "CI_Low":    round(result["ci_low"],  6),
            "CI_High":   round(result["ci_high"], 6),
            "Std":       round(result["std"],     6),
            "CI_String": ci_str,
        })

    return pd.DataFrame(rows)


# =========================================================================== #
#                               MAIN                                           #
# =========================================================================== #

def main():
    seed_everything(config.SEED)
    config.setup_dirs()
    logger = setup_logger(
        "confidence_intervals",
        os.path.join(config.OUTPUT_DIR, "confidence_intervals.log"),
    )
    logger.info("=" * 65)
    logger.info("  DermaFusion-AI — Bootstrap Confidence Intervals (N=2000)")
    logger.info("=" * 65)

    # ── Try to load cached predictions first ─────────────────────────────── #
    cache_path = os.path.join(config.OUTPUT_DIR, "test_predictions.npz")

    if os.path.exists(cache_path):
        logger.info(f"Loading cached test predictions from {cache_path}")
        data         = np.load(cache_path)
        y_true       = data["y_true"]
        y_pred_probs = data["y_pred_probs"]
        logger.info(f"  Loaded {len(y_true)} samples.")
    else:
        logger.info("No cached predictions found — running fresh inference with TTA...")

        # ── Load data ───────────────────────────────────────────────────── #
        _, _, test_loader, _ = get_unified_dataloaders(
            config.DATA_DIR,
            masks_dir=os.path.join(config.DATA_DIR, "masks"),
        )

        # ── Load models ─────────────────────────────────────────────────── #
        unet = SwinTransformerUNet(pretrained=False).to(config.DEVICE)

        unet_path = os.path.join(config.WEIGHTS_DIR, "best_unet.pth")
        if os.path.exists(unet_path):
            unet.load_state_dict(torch.load(unet_path, map_location=config.DEVICE,
                                            weights_only=True))

        model = DualBranchFusionClassifier(
            eva02_name=config.EVA02_BACKBONE,     eva02_pretrained=False,
            convnext_name=config.CONVNEXT_BACKBONE, convnext_pretrained=False,
            fusion_dim=config.FUSION_EMBED_DIM,  num_heads=config.FUSION_NUM_HEADS,
            num_classes=config.NUM_CLASSES,      dropout=config.FUSION_DROPOUT,
        ).to(config.DEVICE)

        model_path = os.path.join(config.WEIGHTS_DIR, "best_dual_branch_fusion.pth")
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=config.DEVICE,
                                             weights_only=True))

        # ── TTA inference ────────────────────────────────────────────────── #
        n_views = config.TTA_N_VIEWS if config.USE_TTA else 1
        tta_engine = TTAInference(model, unet, config.DEVICE, n_views=n_views)
        y_true, y_pred_probs, _ = tta_engine.predict_batch(
            test_loader, desc=f"TTA inference (N={n_views})"
        )

        # Cache for future runs
        np.savez(cache_path, y_true=y_true, y_pred_probs=y_pred_probs)
        logger.info(f"  Cached predictions to {cache_path} for future fast runs.")

    # ── Compute CIs ──────────────────────────────────────────────────────── #
    logger.info(f"\nRunning bootstrap (N=2000, seed={config.SEED})...")
    logger.info("  This takes ~30–60 seconds depending on test set size...")
    results_df = compute_all_metrics_with_ci(y_true, y_pred_probs, n_bootstrap=2000)

    # ── Print paper-ready table ───────────────────────────────────────────── #
    logger.info("\n" + "=" * 65)
    logger.info("  PAPER-READY RESULTS WITH 95% CONFIDENCE INTERVALS")
    logger.info("=" * 65)
    for _, row in results_df.iterrows():
        logger.info(f"  {row['Metric']:<22}: {row['CI_String']}")

    print("\n" + "=" * 65)
    print("  PAPER-READY RESULTS WITH 95% CONFIDENCE INTERVALS")
    print("=" * 65)
    for _, row in results_df.iterrows():
        print(f"  {row['Metric']:<22}: {row['CI_String']}")

    # ── Save CSV ─────────────────────────────────────────────────────────── #
    csv_path = os.path.join(config.OUTPUT_DIR, "confidence_intervals.csv")
    results_df.to_csv(csv_path, index=False)
    logger.info(f"\n  Results saved to {csv_path}")
    print(f"\n✅ Done. Saved to {csv_path}")


if __name__ == "__main__":
    main()
