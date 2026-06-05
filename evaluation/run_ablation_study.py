"""
Ablation Study Pipeline — 2026 SOTA Evaluation
===============================================
Evaluates 6 configurations on the unified test set:
  1. Ablation 1: No TTA (Single-view evaluation)
  2. Ablation 2: ConvNeXt Only (Bypass EVA-02 and fusion)
  3. Ablation 3: EVA-02 Only (Bypass ConvNeXt and fusion)
  4. Ablation 4: No Cross-Attention (Fuse via simple average)
  5. Ablation 5: No Segmentation (Feed original image to ConvNeXt)
  6. Full Model (Dual-Branch + Segmentation + Cross-Attention + TTA)

Saves results to outputs/ablation_study_results.csv
Generates comparison plot outputs/plots/ablation_study_bar.png
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.amp import autocast

from configs.config import config
from utils.seed import seed_everything
from utils.logger import setup_logger
from datasets.unified_dataset import get_unified_dataloaders
from models.transformer_unet import SwinTransformerUNet
from models.unet import LightweightUNet
from models.dual_branch_fusion import DualBranchFusionClassifier
from evaluation.metrics import compute_metrics, log_metrics
from training.train_utils import apply_mask
from training.tta import _apply_tta_view


@torch.no_grad()
def evaluate_config(model, unet, loader, device, config_name, n_views=5):
    """
    Evaluates a specific model configuration on the dataset loader.
    
    Args:
        model: DualBranchFusionClassifier
        unet: Segmentation model
        loader: Test DataLoader
        device: PyTorch device
        config_name: Name of the configuration to run
        n_views: Number of TTA views (1 for no TTA, 5 for TTA)
    """
    model.eval()
    if unet is not None:
        unet.eval()
        
    all_targets = []
    all_probs = []
    
    for batch in tqdm(loader, desc=f"Eval: {config_name}"):
        images = batch['image'].to(device)
        labels = batch['label']
        B = images.size(0)
        
        # We will collect prediction probabilities for each sample across TTA views
        batch_probs_views = []
        
        for view_idx in range(n_views):
            aug_images = _apply_tta_view(images, view_idx)
            
            with autocast('cuda', enabled=(device == 'cuda')):
                if config_name == "Ablation 2: ConvNeXt Only":
                    # Generate mask and apply it
                    mask_logits = unet(aug_images)
                    images_seg = apply_mask(aug_images, mask_logits)
                    # Bypass EVA-02 branch: project ConvNeXt features directly to classifier
                    feat_conv = model.branch_conv(images_seg)
                    feat_conv = model.proj_conv(feat_conv)
                    logits = model.classifier(feat_conv)
                    
                elif config_name == "Ablation 3: EVA-02 Only":
                    # Bypass ConvNeXt branch: project EVA-02 features directly to classifier
                    feat_eva = model.branch_eva(aug_images)
                    feat_eva = model.proj_eva(feat_eva)
                    logits = model.classifier(feat_eva)
                    
                elif config_name == "Ablation 4: No Cross-Attention":
                    # Generate mask and apply it
                    mask_logits = unet(aug_images)
                    images_seg = apply_mask(aug_images, mask_logits)
                    # Forward pass through backbones
                    feat_eva = model.branch_eva(aug_images)
                    feat_eva = model.proj_eva(feat_eva)
                    feat_conv = model.branch_conv(images_seg)
                    feat_conv = model.proj_conv(feat_conv)
                    # Bypass cross-attention, fuse by simple average
                    fused = (feat_eva + feat_conv) / 2.0
                    combined = model.gate(fused, feat_eva, feat_conv)
                    logits = model.classifier(combined)
                    
                elif config_name == "Ablation 5: No Segmentation":
                    # Feed standard (unmasked) image to the ConvNeXt branch instead of the Swin-UNet soft mask
                    feat_eva = model.branch_eva(aug_images)
                    feat_eva = model.proj_eva(feat_eva)
                    feat_conv = model.branch_conv(aug_images) # Feed aug_images to ConvNeXt
                    feat_conv = model.proj_conv(feat_conv)
                    # Normal fusion and classification
                    fused, _ = model.fusion(feat_eva, feat_conv)
                    combined = model.gate(fused, feat_eva, feat_conv)
                    logits = model.classifier(combined)
                    
                else: # Ablation 1 (No TTA, n_views=1) or Full Model (n_views=5)
                    mask_logits = unet(aug_images)
                    images_seg = apply_mask(aug_images, mask_logits)
                    logits, _ = model(aug_images, images_seg)
            
            probs = torch.softmax(logits, dim=1).cpu().float().numpy()
            batch_probs_views.append(probs)
            
        # Average TTA views
        avg_probs = np.mean(batch_probs_views, axis=0) # (B, C)
        
        all_targets.extend(labels.numpy().tolist())
        all_probs.extend(avg_probs.tolist())
        
    return np.array(all_targets), np.array(all_probs)


def plot_ablation_results(results_df, save_path):
    """Plots a premium dark-themed bar chart comparing the ablation configurations."""
    # Order configurations logically
    results_df = results_df.copy()
    
    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(12, 7), facecolor="#0d0d0d")
        ax.set_facecolor("#0d0d0d")
        
        x = np.arange(len(results_df))
        width = 0.35
        
        # Gold/Yellow for Balanced Accuracy, Pink/Magenta for Mel Sensitivity
        rects1 = ax.bar(x - width/2, results_df['balanced_accuracy'] * 100, width, 
                        label='Balanced Accuracy (%)', color='#eab308', edgecolor='#1e1b4b', linewidth=1)
        rects2 = ax.bar(x + width/2, results_df['mel_sensitivity'] * 100, width, 
                        label='Melanoma Sensitivity (%)', color='#db2777', edgecolor='#1e1b4b', linewidth=1)
        
        # Add labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Score (%)', color='#94a3b8', fontsize=11, fontweight="semibold", labelpad=12)
        ax.set_title('DERMA-FUSION MODEL ABLATION STUDY', color='#ffec5c', fontsize=14, fontweight="bold", pad=20)
        ax.set_xticks(x)
        # Wrap long labels
        labels_wrapped = [name.replace(" (", "\n(") for name in results_df['configuration']]
        ax.set_xticklabels(labels_wrapped, rotation=15, ha="right", color="#e2e8f0", fontsize=9)
        
        ax.legend(loc="lower left", fontsize=10, facecolor='#151515', edgecolor='#333333', labelcolor='#e2e8f0')
        ax.grid(color='#222222', linestyle='-', linewidth=0.5, axis='y')
        
        # Style spines
        for spine in ax.spines.values():
            spine.set_color('#333333')
            spine.set_linewidth(1.0)
            
        # Add values on top of bars
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.1f}%',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8, color='#e2e8f0')
                                
        autolabel(rects1)
        autolabel(rects2)
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=180, bbox_inches='tight', facecolor='#0d0d0d')
        plt.close()


def main():
    seed_everything(config.SEED)
    config.setup_dirs()
    logger = setup_logger("ablation", os.path.join(config.OUTPUT_DIR, "ablation_study.log"))
    logger.info("Starting Ablation Study Pipeline")
    
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
        unet.load_state_dict(torch.load(unet_path, map_location=config.DEVICE, weights_only=True))
        logger.info(f"Loaded UNet weights from {unet_path}")
    else:
        logger.warning(f"UNet weights not found at {unet_path}. Using untrained UNet.")
        
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
    
    model_path = os.path.join(config.WEIGHTS_DIR, "best_dual_branch_fusion.pth")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=config.DEVICE, weights_only=True))
        logger.info(f"Loaded classifier weights from {model_path}")
    else:
        logger.warning(f"Classifier weights not found at {model_path}. Using untrained classifier.")
        
    # ── Configurations to Evaluate ────────────────────────────────────────── #
    # Standard classes order is: ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    # 'mel' is class index 4
    mel_idx = config.CLASSES.index('mel') if 'mel' in config.CLASSES else 4
    
    configs_to_run = [
        {"name": "Ablation 1: No TTA", "n_views": 1},
        {"name": "Ablation 2: ConvNeXt Only", "n_views": 5},
        {"name": "Ablation 3: EVA-02 Only", "n_views": 5},
        {"name": "Ablation 4: No Cross-Attention", "n_views": 5},
        {"name": "Ablation 5: No Segmentation", "n_views": 5},
        {"name": "Full Model", "n_views": 5},
    ]
    
    results = []
    
    for run_cfg in configs_to_run:
        name = run_cfg["name"]
        n_views = run_cfg["n_views"]
        logger.info(f"Evaluating: {name} (TTA views = {n_views})")
        
        y_true, y_pred_probs = evaluate_config(
            model=model,
            unet=unet,
            loader=test_loader,
            device=config.DEVICE,
            config_name=name,
            n_views=n_views
        )
        
        # Compute metrics
        metrics = compute_metrics(y_true, y_pred_probs)
        log_metrics(metrics, logger, prefix=name)
        
        mel_sens = metrics['per_class_sensitivity'].get('mel', 0.0)
        mel_spec = metrics['per_class_specificity'].get('mel', 0.0)
        
        results.append({
            "configuration": name,
            "accuracy": metrics['accuracy'],
            "balanced_accuracy": metrics['balanced_accuracy'],
            "macro_f1": metrics['macro_f1'],
            "macro_auc": metrics['macro_auc'],
            "ece": metrics['ece'],
            "mel_sensitivity": mel_sens,
            "mel_specificity": mel_spec,
        })
        
    # Save to CSV
    results_df = pd.DataFrame(results)
    csv_save_path = os.path.join(config.OUTPUT_DIR, "ablation_study_results.csv")
    results_df.to_csv(csv_save_path, index=False)
    logger.info(f"Ablation results saved to: {csv_save_path}")
    
    # Plot results
    plot_save_path = os.path.join(config.PLOTS_DIR, "ablation_study_bar.png")
    plot_ablation_results(results_df, plot_save_path)
    logger.info(f"Ablation comparison plot saved to: {plot_save_path}")


if __name__ == "__main__":
    main()
