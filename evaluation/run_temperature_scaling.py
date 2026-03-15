"""
Temperature Scaling Runner
===========================
Finds the optimal temperature T on the validation set, then saves it to
outputs/temperature.pt for use during inference.

Usage:
    python -m evaluation.run_temperature_scaling

Requirements:
    • best_dual_branch_fusion.pth  (trained classifier checkpoint)
    • best_unet.pth                (segmentation model checkpoint)

Output:
    • outputs/temperature.pt       (T scalar used at inference time)
    • Printed ECE before/after + reliability diagram comparison
"""

import os
import sys

# Add project root to path so relative imports work when run as a script
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

import numpy as np
import torch
from torch.amp import autocast
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from configs.config import config
from utils.seed import seed_everything
from utils.logger import setup_logger
from datasets.unified_dataset import get_unified_dataloaders
from models.transformer_unet import SwinTransformerUNet
from models.dual_branch_fusion import DualBranchFusionClassifier
from training.train_utils import apply_mask
from evaluation.calibration import TemperatureScaler


# =========================================================================== #
#                         LOGIT COLLECTION (no TTA)                            #
# =========================================================================== #

def collect_val_logits(model, unet, val_loader, device):
    """
    One forward pass over the validation set WITHOUT TTA, returning raw logits
    (before softmax). Temperature scaling MUST be fit on raw logits.

    Returns:
        logits_all : np.ndarray  shape (N, C)
        labels_all : np.ndarray  shape (N,)
    """
    model.eval()
    unet.eval()
    logits_list, labels_list = [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Collecting val logits"):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            with autocast("cuda", enabled=(device == "cuda")):
                mask_logits = unet(images)
                images_seg  = apply_mask(images, mask_logits)
                logits, _   = model(images, images_seg)  # raw logits, NOT softmax

            logits_list.append(logits.cpu().numpy())
            labels_list.append(labels.cpu().numpy())

    return np.vstack(logits_list), np.concatenate(labels_list)


# =========================================================================== #
#                         RELIABILITY DIAGRAM                                  #
# =========================================================================== #

def plot_reliability_diagram(logits, labels, T_before, T_after, save_path):
    """
    Side-by-side reliability diagrams before and after temperature scaling.
    A perfectly calibrated model has bars on the diagonal.
    """
    from scipy.special import softmax as sp_softmax

    n_bins = 10
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, T, title in zip(
        axes,
        [T_before, T_after],
        ["Before Temperature Scaling (T=1.0)", f"After Temperature Scaling (T={T_after:.3f})"],
    ):
        probs = sp_softmax(logits / T, axis=1)
        confs = probs.max(axis=1)
        preds = probs.argmax(axis=1)
        accs  = (preds == labels).astype(float)

        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_accs, bin_confs, bin_counts = [], [], []

        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (confs > lo) & (confs <= hi)
            if mask.sum() == 0:
                bin_accs.append(0)
                bin_confs.append((lo + hi) / 2)
                bin_counts.append(0)
            else:
                bin_accs.append(accs[mask].mean())
                bin_confs.append(confs[mask].mean())
                bin_counts.append(mask.sum())

        bins_mid = [(lo + hi) / 2 for lo, hi in zip(bin_edges[:-1], bin_edges[1:])]
        ax.bar(bins_mid, bin_accs, width=0.09, alpha=0.7, label="Accuracy", color="steelblue")
        ax.plot([0, 1], [0, 1], "r--", label="Perfect calibration")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title(title)
        ax.legend(fontsize=8)

        # ECE annotation
        from scipy.special import softmax as sp2
        probs2 = sp2(logits / T, axis=1)
        confs2 = probs2.max(axis=1)
        preds2 = probs2.argmax(axis=1)
        accs2  = (preds2 == labels).astype(float)
        ece_val = 0.0
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (confs2 > lo) & (confs2 <= hi)
            if mask.sum() > 0:
                ece_val += mask.sum() * abs(accs2[mask].mean() - confs2[mask].mean())
        ece_val /= len(labels)
        ax.annotate(f"ECE = {ece_val:.4f}", xy=(0.05, 0.92), xycoords="axes fraction",
                    fontsize=10, color="darkred")

    plt.suptitle("DermaFusion-AI Calibration — Reliability Diagram", fontsize=13)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Plot] Reliability diagram saved to {save_path}")


# =========================================================================== #
#                               MAIN                                           #
# =========================================================================== #

def main():
    seed_everything(config.SEED)
    config.setup_dirs()
    logger = setup_logger(
        "temperature_scaling",
        os.path.join(config.OUTPUT_DIR, "temperature_scaling.log"),
    )
    logger.info("=" * 60)
    logger.info("  DermaFusion-AI — Temperature Scaling Calibration")
    logger.info("=" * 60)

    # ── Fast-path: use pre-saved val logits (downloaded from Colab/Kaggle) ── #
    # If you saved val_logits.npz from your training machine, place it at:
    #   outputs/val_logits.npz
    # This skips model loading entirely — much faster on a laptop without data.
    fast_path = os.path.join(config.OUTPUT_DIR, "val_logits.npz")
    if os.path.exists(fast_path):
        logger.info(f"Fast-path: loading pre-saved val logits from {fast_path}")
        data        = np.load(fast_path)
        logits_val  = data["logits"].astype(np.float32)
        labels_val  = data["labels"].astype(int)
        logger.info(f"  Loaded {len(labels_val)} val samples, {logits_val.shape[1]} classes")

        scaler  = TemperatureScaler(device="cpu")
        opt_T   = scaler.fit(logits_val, labels_val, verbose=True)
        temp_save_path = os.path.join(config.OUTPUT_DIR, "temperature.pt")
        scaler.save(temp_save_path)

        diagram_path = os.path.join(config.PLOTS_DIR, "reliability_diagram_calibration.png")
        os.makedirs(config.PLOTS_DIR, exist_ok=True)
        plot_reliability_diagram(logits_val, labels_val, T_before=1.0, T_after=opt_T,
                                 save_path=diagram_path)

        from scipy.special import softmax as sp_softmax
        from evaluation.metrics import compute_ece
        ece_before = compute_ece(labels_val, sp_softmax(logits_val / 1.0, axis=1))
        ece_after  = compute_ece(labels_val, sp_softmax(logits_val / opt_T, axis=1))
        print(f"\n✅ Temperature scaling complete (fast-path).")
        print(f"   T = {opt_T:.4f}  |  ECE: {ece_before:.4f} → {ece_after:.4f}")
        print(f"   Saved to: {temp_save_path}")
        print(f"   Reliability diagram: {diagram_path}")
        print(f"\n   ADD TO PAPER: 'Temperature scaling (T={opt_T:.3f}) reduced ECE from {ece_before:.4f} to {ece_after:.4f}.'")
        return

    # ── Full-path: collect logits by running the model (needs real data) ── #
    logger.info("val_logits.npz not found — running full model inference...")
    logger.info("(To skip this, download val_logits.npz from Colab and place in outputs/)")

    # ── Dataloaders (we only need the val set) ─────────────────────────── #
    _, val_loader, _, _ = get_unified_dataloaders(
        config.DATA_DIR,
        masks_dir=os.path.join(config.DATA_DIR, "masks"),
    )

    # ── Segmentation model (always SwinTransformerUNet — config.SEG_MODEL='swin_unet') ── #
    unet = SwinTransformerUNet(pretrained=False).to(config.DEVICE)

    unet_path = os.path.join(config.WEIGHTS_DIR, "best_unet.pth")
    if os.path.exists(unet_path):
        unet.load_state_dict(
            torch.load(unet_path, map_location=config.DEVICE, weights_only=True)
        )
        logger.info(f"Loaded UNet weights from {unet_path}")
    else:
        logger.warning("UNet weights not found — segmentation masks will be incorrect!")

    # ── Classification model ────────────────────────────────────────────── #
    model = DualBranchFusionClassifier(
        eva02_name=config.EVA02_BACKBONE,
        eva02_pretrained=False,
        convnext_name=config.CONVNEXT_BACKBONE,
        convnext_pretrained=False,
        fusion_dim=config.FUSION_EMBED_DIM,
        num_heads=config.FUSION_NUM_HEADS,
        num_classes=config.NUM_CLASSES,
        dropout=config.FUSION_DROPOUT,
    ).to(config.DEVICE)

    model_path = os.path.join(config.WEIGHTS_DIR, "best_dual_branch_fusion.pth")
    if os.path.exists(model_path):
        model.load_state_dict(
            torch.load(model_path, map_location=config.DEVICE, weights_only=True)
        )
        logger.info(f"Loaded classifier weights from {model_path}")
    else:
        logger.warning("Classifier weights not found!")

    # ── Collect raw logits (NO softmax, NO TTA) ─────────────────────────── #
    logger.info("Collecting validation set raw logits (no TTA, no softmax)...")
    logits_val, labels_val = collect_val_logits(model, unet, val_loader, config.DEVICE)
    logger.info(f"  Val set size: {len(labels_val)} samples, {logits_val.shape[1]} classes")

    # ── Fit temperature ──────────────────────────────────────────────────── #
    logger.info("Fitting temperature scaler on validation logits...")
    scaler = TemperatureScaler(device=config.DEVICE)
    opt_T  = scaler.fit(logits_val, labels_val, verbose=True)

    logger.info(f"  Optimal temperature T = {opt_T:.4f}")
    if opt_T > 1.5:
        logger.info("  → T > 1.5: model is significantly overconfident — calibration strongly recommended.")
    elif opt_T > 1.0:
        logger.info("  → T slightly > 1.0: mild overconfidence, calibration gives moderate improvement.")
    else:
        logger.info("  → T ≤ 1.0: model is underconfident or already well-calibrated.")

    # ── Save temperature ─────────────────────────────────────────────────── #
    temp_save_path = os.path.join(config.OUTPUT_DIR, "temperature.pt")
    scaler.save(temp_save_path)
    logger.info(f"  Saved temperature to {temp_save_path}")

    # ── Reliability diagram ──────────────────────────────────────────────── #
    diagram_path = os.path.join(config.PLOTS_DIR, "reliability_diagram_calibration.png")
    plot_reliability_diagram(logits_val, labels_val, T_before=1.0, T_after=opt_T,
                             save_path=diagram_path)
    logger.info(f"  Reliability diagram saved to {diagram_path}")

    # ── Summary for paper ────────────────────────────────────────────────── #
    from scipy.special import softmax as sp_softmax
    from evaluation.metrics import compute_ece

    probs_before = sp_softmax(logits_val / 1.0, axis=1)
    probs_after  = sp_softmax(logits_val / opt_T, axis=1)
    ece_before   = compute_ece(labels_val, probs_before)
    ece_after    = compute_ece(labels_val, probs_after)

    logger.info("\n" + "=" * 60)
    logger.info("  PAPER-READY CALIBRATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  ECE before temperature scaling : {ece_before:.4f}")
    logger.info(f"  ECE after  temperature scaling : {ece_after:.4f}")
    logger.info(f"  ECE reduction                  : {ece_before - ece_after:.4f} "
                f"({100*(ece_before-ece_after)/ece_before:.1f}% improvement)")
    logger.info(f"  Optimal temperature T          : {opt_T:.4f}")
    logger.info("=" * 60)
    logger.info("  ADD TO PAPER: 'Temperature scaling (T={:.3f}) reduced ECE from {:.4f} to {:.4f}'.".format(
        opt_T, ece_before, ece_after
    ))

    print("\n✅ Temperature scaling complete.")
    print(f"   T = {opt_T:.4f}  |  ECE: {ece_before:.4f} → {ece_after:.4f}")
    print(f"   Saved to: {temp_save_path}")
    print(f"   Reliability diagram: {diagram_path}")


if __name__ == "__main__":
    main()
