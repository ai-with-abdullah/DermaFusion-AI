"""
Evaluation Pipeline — 2026 SOTA Upgrade
=========================================
Upgrades over previous version:
  ✓ Test-Time Augmentation (TTA, N=5 views) for better calibrated predictions
  ✓ Full metric suite: balanced acc, per-class sens/spec, ECE, pAUC
  ✓ GradCAM++ (EVA-02 + ConvNeXt V2) for XAI visualizations
  ✓ Multi-dataset unified loader
"""

import os
import torch
from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from torch.amp import autocast

from configs.config import config
from utils.seed import seed_everything
from utils.logger import setup_logger
from datasets.unified_dataset import get_unified_dataloaders
from models.transformer_unet import SwinTransformerUNet
from models.unet import LightweightUNet
from models.dual_branch_fusion import DualBranchFusionClassifier
from evaluation.metrics import compute_metrics, plot_confusion_matrix, log_metrics
from evaluation.calibration import TemperatureScaler, apply_mel_threshold_boost
from evaluation.gradcam_plus_plus import generate_dual_gradcam
from evaluation.explainability import visualize_batch_diagnostics
from training.train_utils import apply_mask
from training.tta import TTAInference


def plot_multiclass_roc(y_true, y_pred_probs, save_path):
    with plt.style.context("dark_background"):
        plt.figure(figsize=(11, 9), facecolor="#0d0d0d")
        ax = plt.gca()
        ax.set_facecolor("#0d0d0d")

        # Color palette for curves
        colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(config.CLASSES)))

        for i, cls in enumerate(config.CLASSES):
            y_true_bin = (y_true == i).astype(int)
            fpr, tpr, _ = roc_curve(y_true_bin, y_pred_probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2.5, color=colors[i], label=f'{cls.upper()} (AUC = {roc_auc:.4f})')

        plt.plot([0, 1], [0, 1], 'k--', color='#555555', lw=1.5, linestyle="--")
        plt.xlabel('FALSE POSITIVE RATE', color='#94a3b8', fontsize=10, fontweight="semibold", labelpad=12)
        plt.ylabel('TRUE POSITIVE RATE', color='#94a3b8', fontsize=10, fontweight="semibold", labelpad=12)
        plt.title('RECEIVER OPERATING CHARACTERISTIC (ROC) CURVES', color='#ffec5c', fontsize=13, fontweight="bold", pad=20)

        plt.legend(loc="lower right", fontsize=9, facecolor='#151515', edgecolor='#333333', labelcolor='#e2e8f0')
        plt.grid(color='#222222', linestyle='-', linewidth=0.5)

        # Style spine/borders
        for spine in ax.spines.values():
            spine.set_color('#333333')
            spine.set_linewidth(1.0)

        plt.tight_layout()
        plt.savefig(save_path, dpi=180, bbox_inches='tight', facecolor='#0d0d0d')
        plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="DermaFusion Evaluation")
    parser.add_argument("--ablation", type=str, default=None,
                        choices=["no_tta", "convnext_only", "eva_only", "no_attention", "no_segmentation"],
                        help="Ablation model to evaluate")
    args = parser.parse_args()
    ablation = args.ablation

    seed_everything(config.SEED)
    config.setup_dirs()
    log_name = f"evaluate_{ablation}.log" if ablation else "evaluate_dual_branch.log"
    logger = setup_logger("evaluate", os.path.join(config.OUTPUT_DIR, log_name))
    logger.info("Starting Dual-Branch Fusion Evaluation (TTA enabled)")
    if ablation:
        logger.info(f"  Active Ablation:        {ablation}")

    # ── Dataloaders ──────────────────────────────────────────────────────── #
    _, _, test_loader, _ = get_unified_dataloaders(
        config.DATA_DIR, masks_dir=os.path.join(config.DATA_DIR, "masks")
    )

    # ── Load Segmentation Model ───────────────────────────────────────────── #
    if config.SEG_MODEL == 'swin_unet':
        unet = SwinTransformerUNet(pretrained=False).to(config.DEVICE)
    else:
        unet = LightweightUNet(n_channels=3, n_classes=1).to(config.DEVICE)

    unet_path = os.path.join(config.WEIGHTS_DIR, "best_unet.pth")
    if os.path.exists(unet_path):
        # FIXED (Fix #12): weights_only=True for security, loaded ONCE only.
        unet.load_state_dict(torch.load(unet_path, map_location=config.DEVICE,
                                         weights_only=True))
        logger.info(f"Loaded UNet weights from {unet_path}")
    else:
        logger.warning(f"UNet weights not found at {unet_path}. Using untrained model.")

    # ── Load Classification Model ─────────────────────────────────────────── #
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

    best_filename = f"best_classifier_{ablation}.pth" if ablation else "best_dual_branch_fusion.pth"
    model_path = os.path.join(config.WEIGHTS_DIR, best_filename)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=config.DEVICE,
                                          weights_only=True))
        logger.info(f"Loaded classifier weights from {model_path}")
    else:
        logger.warning(f"Classifier weights not found at {model_path}.")


    # ── TTA Inference ─────────────────────────────────────────────────────── #
    if ablation == "no_tta":
        n_views = 1
    else:
        n_views = config.TTA_N_VIEWS if config.USE_TTA else 1
    logger.info(f"Running inference with TTA (n_views={n_views})...")
    tta_engine = TTAInference(model, unet, config.DEVICE, n_views=n_views, ablation=ablation)

    # FIXED (Bug #4): TTAInference.predict_batch() already averages post-softmax
    # probabilities internally — the returned array is *probabilities*, not raw logits.
    # Previously named y_pred_logits which caused a NameError on line 123 where
    # y_pred_probs was referenced but never assigned.
    y_true, y_pred_probs, dataset_names = tta_engine.predict_batch(
        test_loader, desc=f'TTA Eval (N={n_views})'
    )
    # FIXED (Fix #13): Temperature scaling is skipped when TTA is active.
    # Temperature scaling must be applied to RAW LOGITS before softmax averaging.
    # When n_views > 1, TTAInference already averages post-softmax probabilities,
    # so applying a temperature scaler after the fact is mathematically incorrect
    # and can actually worsen calibration (ECE).
    # To use temperature scaling correctly: collect raw logits inside TTAInference
    # before the softmax call, then scale and average. That refactor is tracked
    # as a future improvement. For now, TTA itself provides implicit calibration.
    if config.USE_TTA and n_views > 1:
        logger.info(
            f"Temperature scaling SKIPPED — TTA (n_views={n_views}) averages post-softmax "
            "probabilities. Applying temperature to averaged probs is mathematically incorrect. "
            "Run evaluation/calibration.py on val set (n_views=1) for proper calibration."
        )
        y_pred_probs_calibrated = y_pred_probs
    else:
        temp_path = os.path.join(config.OUTPUT_DIR, 'temperature.pt')
        if os.path.exists(temp_path):
            scaler = TemperatureScaler.load(temp_path, device=config.DEVICE)
            T = scaler.get_temperature()
            logger.info(f"Applying temperature scaling (T={T:.3f})...")
            # Convert probabilities back to logits, apply temperature, re-softmax
            # Note: this approximation is only valid for single-view (no TTA) evaluation.
            import torch.nn.functional as F
            logits_approx = torch.tensor(np.log(y_pred_probs + 1e-8))  # inverse softmax approx
            logits_scaled = logits_approx / T
            y_pred_probs_calibrated = F.softmax(logits_scaled, dim=1).numpy()
        else:
            logger.warning(
                "No temperature.pt found — ECE may be inflated. "
                "Run evaluation/calibration.py first for proper calibration."
            )
            y_pred_probs_calibrated = y_pred_probs

    # ── Metrics: Standard (raw threshold) ────────────────────────────────── #
    # Paper reporting note: test set composition (after patient-aware split):
    #   HAM10000 (~10% of 10015), ISIC2020 (~10% of 584),
    #   ISIC2024 (~10% pos + 20:1 neg downsampling for computational efficiency).
    #   No patient overlap between train/val/test guaranteed by _patient_aware_split().
    metrics = compute_metrics(y_true, y_pred_probs_calibrated)
    log_metrics(metrics, logger, prefix="Test (TTA, calibrated)")

    # ── GradCAM++ XAI for first batch ────────────────────────────────────── #
    xai_dir = os.path.join(config.OUTPUT_DIR, "explainability_heatmaps")
    os.makedirs(xai_dir, exist_ok=True)
    logger.info("Generating GradCAM++ XAI visualizations...")

    model.eval()
    unet.eval()
    for batch in test_loader:
        images   = batch['image'].to(config.DEVICE)
        labels   = batch['label'].to(config.DEVICE)
        img_ids  = batch['image_id']

        with torch.no_grad():
            with autocast('cuda', enabled=(config.DEVICE == 'cuda')):
                mask_logits = unet(images)
                images_seg  = apply_mask(images, mask_logits)
            logits, _ = model(images, images_seg)

        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)

        # GradCAM++ for up to 4 samples
        for k in range(min(4, len(images))):
            try:
                generate_dual_gradcam(
                    model=model,
                    image_tensor=images[k].cpu(),
                    target_class=int(labels[k].item()),
                    device=config.DEVICE,
                    save_dir=xai_dir,
                    image_id=str(img_ids[k]),
                    image_seg_tensor=images_seg[k].cpu(),
                )
            except Exception as e:
                logger.warning(f"GradCAM++ failed for sample {img_ids[k]}: {e}")

        # Legacy full diagnostic panel
        try:
            visualize_batch_diagnostics(
                model=model, batch=batch, predictions=preds, probs=probs,
                save_dir=xai_dir, device=config.DEVICE, max_samples=5,
                class_names=config.CLASSES,
            )
        except Exception as e:
            logger.warning(f"Legacy diagnostic panel failed: {e}")
        break  # only first batch for XAI

    logger.info("Generating confusion matrix and ROC curves...")
    cm_name = f"confusion_matrix_{ablation}.png" if ablation else "confusion_matrix_dual_branch.png"
    roc_name = f"roc_curve_{ablation}.png" if ablation else "roc_curve_dual_branch.png"
    cm_path = os.path.join(config.PLOTS_DIR, cm_name)
    roc_path = os.path.join(config.PLOTS_DIR, roc_name)
    
    plot_confusion_matrix(
        y_true, y_pred_probs_calibrated,
        save_path=cm_path,
        normalize=True,
        title=f"CONFUSION MATRIX ({ablation.upper() if ablation else 'COMBINED'})",
    )
    plot_multiclass_roc(
        y_true, y_pred_probs_calibrated,
        save_path=roc_path,
    )

    # ── Separate Confusion Matrices per Dataset ───────────────────────────── #
    if dataset_names is not None and len(dataset_names) > 0:
        logger.info("Generating separate confusion matrices for each dataset...")
        unique_datasets = sorted(list(set(dataset_names)))
        for dataset in unique_datasets:
            idx = [i for i, name in enumerate(dataset_names) if name == dataset]
            if len(idx) == 0:
                continue
            y_true_ds = y_true[idx]
            y_pred_probs_ds = y_pred_probs_calibrated[idx]
            
            ds_filename = f"confusion_matrix_{dataset.lower()}_{ablation}.png" if ablation else f"confusion_matrix_{dataset.lower()}.png"
            plot_confusion_matrix(
                y_true_ds, y_pred_probs_ds,
                save_path=os.path.join(config.PLOTS_DIR, ds_filename),
                normalize=True,
                title=f"CONFUSION MATRIX ({dataset.upper()} - {ablation.upper() if ablation else 'FULL MODEL'})",
            )
            logger.info(f"Saved separate confusion matrix for {dataset} to {ds_filename}")

    # ── Run Ablation Study & Dashboard (Only when evaluating Full Model) ──── #
    if ablation is None:
        logger.info("Starting Ablation Study execution...")
        ablation_path = os.path.join(config.PLOTS_DIR, "ablation_study_bar.png")
        try:
            from evaluation.run_ablation_study import main as run_ablations
            run_ablations()
            logger.info("Ablation Study executed successfully.")
        except Exception as e:
            logger.error(f"Ablation Study failed: {e}")

        dashboard_path = os.path.join(config.PLOTS_DIR, "evaluation_dashboard.png")
        logger.info(f"Generating combined evaluation dashboard → {dashboard_path}...")
        try:
            generate_combined_dashboard(cm_path, roc_path, ablation_path, dashboard_path)
            logger.info("Combined evaluation dashboard generated successfully.")
        except Exception as e:
            logger.error(f"Failed to generate dashboard: {e}")

    # ── Mel Threshold Boost — DIAGNOSTIC ONLY ────────────────────────────── #
    # IMPORTANT: This boost is strictly for logging a supplementary sensitivity
    # metric. It does NOT modify y_pred_probs_calibrated and has NO effect on
    # the confusion matrix, ROC curves, or primary metrics above.
    # Clinical rationale: dermatologists accept ~3× higher FP rate vs FN for mel.
    # A 1.5× boost shifts the effective classification threshold: 0.50 → ~0.33.
    mel_idx = config.CLASSES.index('mel') if 'mel' in config.CLASSES else 4
    preds_boosted = apply_mel_threshold_boost(y_pred_probs_calibrated, mel_idx=mel_idx, boost_factor=1.5)
    mel_sens_boosted  = (preds_boosted[y_true == mel_idx] == mel_idx).mean()
    preds_standard    = y_pred_probs_calibrated.argmax(axis=1)
    mel_sens_standard = (preds_standard[y_true == mel_idx] == mel_idx).mean()
    logger.info(
        f"[DiagnosticOnly] Mel sensitivity @ standard threshold: {mel_sens_standard:.4f} | "
        f"with 1.5× boost (lower threshold): {mel_sens_boosted:.4f}  "
        f"(primary predictions above are UNAFFECTED by this boost)"
    )

    logger.info(f"Evaluation complete. Outputs saved → {config.OUTPUT_DIR}")


def generate_combined_dashboard(cm_path, roc_path, ablation_path, save_path):
    """Combines individual plots into a premium multi-panel dashboard image."""
    import cv2
    if not (os.path.exists(cm_path) and os.path.exists(roc_path) and os.path.exists(ablation_path)):
        raise FileNotFoundError("One or more input images for dashboard are missing.")
    
    # Load images
    img_cm = cv2.imread(cm_path)
    img_roc = cv2.imread(roc_path)
    img_ablation = cv2.imread(ablation_path)
    
    # Convert BGR to RGB
    img_cm = cv2.cvtColor(img_cm, cv2.COLOR_BGR2RGB)
    img_roc = cv2.cvtColor(img_roc, cv2.COLOR_BGR2RGB)
    img_ablation = cv2.cvtColor(img_ablation, cv2.COLOR_BGR2RGB)
    
    # Create 3-panel figure:
    # Top row: Confusion Matrix and ROC curve side-by-side
    # Bottom row: Ablation Study stretched across the bottom
    with plt.style.context("dark_background"):
        fig = plt.figure(figsize=(20, 16), facecolor="#0d0d0d")
        
        # Grid specification: 2 rows, 2 columns
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.85], hspace=0.12, wspace=0.06)
        
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(img_cm)
        ax1.axis('off')
        ax1.set_title("CONFUSION MATRIX", color='#ffec5c', fontsize=12, fontweight="bold", pad=8)
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(img_roc)
        ax2.axis('off')
        ax2.set_title("ROC CURVES", color='#ffec5c', fontsize=12, fontweight="bold", pad=8)
        
        ax3 = fig.add_subplot(gs[1, :])
        ax3.imshow(img_ablation)
        ax3.axis('off')
        ax3.set_title("MODEL ABLATION SYSTEM STUDY", color='#ffec5c', fontsize=12, fontweight="bold", pad=8)
        
        plt.suptitle("DERMA-FUSION SYSTEM EVALUATION DASHBOARD", color='#ffffff', fontsize=18, fontweight="bold", y=0.96)
        
        plt.savefig(save_path, dpi=180, bbox_inches='tight', facecolor='#0d0d0d')
        plt.close()


if __name__ == "__main__":
    main()
