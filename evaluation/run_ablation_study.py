"""
Ablation Study Pipeline — 2026 SOTA Evaluation
===============================================
Evaluates the following on the unified test set:
  - Ablation 1: No TTA (Single-view evaluation)
  - Ablation 5: No Segmentation (Feed original unmasked image to both branches)
  - Ablation 6/7/8: Novelty toggles (uncertainty bias / asymmetry / plain spatial)
  - Full Model (Dual-Branch + Segmentation + Cross-Attention + TTA)

Per-branch contribution is measured separately by FROZEN-FEATURE LINEAR PROBES
(run_branch_probes): a fresh classifier head is trained on each branch's frozen
features under an identical protocol — EVA-02 alone, ConvNeXt alone, and their
concatenation. This replaces the earlier "single branch routed through the
end-to-end-trained head" approach, which fed off-manifold inputs to a head that only
ever saw the gated, LayerNorm'd, cross-attention-fused vector and therefore produced
sub-random AUC (a measurement artifact, not a branch-quality signal).

Saves results to outputs/ablation_study_results.csv
Generates comparison plot outputs/plots/ablation_study_bar.png
"""

import sys
import os
# Prioritize local modules over Hugging Face 'datasets' library
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
for k in list(sys.modules.keys()):
    if k == 'datasets' or k.startswith('datasets.'):
        sys.modules.pop(k)

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader as _TensorLoader
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
from models.dual_branch_fusion import DualBranchFusionClassifier, ClassifierHead
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
                # NOTE: Single-branch ablations ("EVA-02 only" / "ConvNeXt only") are
                # NOT handled here. Feeding a bare single-branch projection into the
                # end-to-end-trained classifier head is off-manifold (the head only ever
                # saw the LayerNorm'd, gated, cross-attention-fused vector) and yields
                # sub-random AUC — a measurement artifact, not branch quality. They are
                # measured properly via the frozen-feature linear-probe protocol in
                # run_branch_probes(). See main().
                # NOTE: "Ablation 4: No Cross-Attention" has been REMOVED. There is no
                # valid eval-time way to delete the cross-attention: bug_attn IS the
                # trained fusion core, and gate + classifier were trained on its output.
                # Bypassing it (or using the untrained pooled `model.fusion`) collapses
                # to sub-random. "Ablation 8: Plain Spatial Fusion" (both novelties off)
                # is the valid fusion ablation.
                if config_name == "Ablation 5: No Segmentation":
                    # Clean INPUT-level ablation: route through the REAL trained forward,
                    # but feed the ORIGINAL (unmasked) image to BOTH branches so the
                    # ConvNeXt branch never sees the segmentation. (The old version used
                    # the untrained pooled path → sub-random; this does not.)
                    mask_logits = unet(aug_images)
                    mask_prob   = torch.sigmoid(mask_logits.float())
                    logits, _ = model(aug_images, aug_images, mask_prob)

                elif config_name == "Ablation 6: No Uncertainty Bias":
                    # Novelty #2 ablation: BUG-Attn with γ=δ off → plain spatial attn
                    mask_logits = unet(aug_images)
                    images_seg  = apply_mask(aug_images, mask_logits)
                    mask_prob   = torch.sigmoid(mask_logits.float())
                    logits, _ = model(aug_images, images_seg, mask_prob,
                                      disable_uncertainty_bias=True)

                elif config_name == "Ablation 7: No Mirror-Asymmetry":
                    # Novelty #3 ablation: turn off the asymmetry contribution
                    mask_logits = unet(aug_images)
                    images_seg  = apply_mask(aug_images, mask_logits)
                    mask_prob   = torch.sigmoid(mask_logits.float())
                    logits, _ = model(aug_images, images_seg, mask_prob,
                                      disable_asymmetry=True)

                elif config_name == "Ablation 8: Plain Spatial Fusion":
                    # Both novelties #2+#3 off → plain spatial cross-attention
                    # (the arXiv:2510.17773-equivalent base design)
                    mask_logits = unet(aug_images)
                    images_seg  = apply_mask(aug_images, mask_logits)
                    mask_prob   = torch.sigmoid(mask_logits.float())
                    logits, _ = model(aug_images, images_seg, mask_prob,
                                      disable_uncertainty_bias=True, disable_asymmetry=True)

                else: # Ablation 1 (No TTA, n_views=1) or Full Model (n_views=5)
                    mask_logits = unet(aug_images)
                    images_seg = apply_mask(aug_images, mask_logits)
                    mask_prob  = torch.sigmoid(mask_logits.float())
                    logits, _ = model(aug_images, images_seg, mask_prob)
            
            probs = torch.softmax(logits, dim=1).cpu().float().numpy()
            batch_probs_views.append(probs)
            
        # Average TTA views
        avg_probs = np.mean(batch_probs_views, axis=0) # (B, C)
        
        all_targets.extend(labels.numpy().tolist())
        all_probs.extend(avg_probs.tolist())
        
    return np.array(all_targets), np.array(all_probs)


@torch.no_grad()
def _extract_branch_features(model, unet, loader, device, max_batches=None, desc="Extract"):
    """
    Runs a single (no-TTA) forward pass through the frozen backbones and collects
    global-average-pooled features from the TRAINED spatial trunk (forward_tokens).

    IMPORTANT: we must NOT use model.proj_eva / model.branch_conv(...) pooled outputs.
    With use_spatial_fusion=True (the trained config), those pooled projection layers
    are never called during training and are therefore random/untrained — features
    from them are garbage. forward_tokens() is exactly the trunk the full model uses,
    so these are genuine trained features.

    Returns:
        feat_eva  : (N, eva_dim)        float32 CPU tensor — EVA-02 trunk features
        feat_conv : (N, conv_bb_dim)    float32 CPU tensor — ConvNeXt trunk features
        labels    : (N,)                int64   CPU tensor
    """
    model.eval()
    if unet is not None:
        unet.eval()

    eva_list, conv_list, lab_list = [], [], []
    for i, batch in enumerate(tqdm(loader, desc=desc)):
        if max_batches is not None and i >= max_batches:
            break
        images = batch['image'].to(device)
        labels = batch['label']

        with autocast('cuda', enabled=(device == 'cuda')):
            mask_logits = unet(images)
            images_seg  = apply_mask(images, mask_logits)
            # Same routing as the full model: EVA sees the original image, ConvNeXt
            # sees the segmented image. Use the TRAINED trunk via forward_tokens()
            # + global-average-pool (NOT the untrained pooled projection layers).
            eva_grid  = model.branch_eva.forward_tokens(images)       # (B, eva_dim, G, G)
            conv_grid = model.branch_conv.forward_tokens(images_seg)  # (B, conv_bb_dim, G, G)
            feat_eva  = eva_grid.mean(dim=(2, 3))                     # (B, eva_dim)
            feat_conv = conv_grid.mean(dim=(2, 3))                    # (B, conv_bb_dim)

        eva_list.append(feat_eva.float().cpu())
        conv_list.append(feat_conv.float().cpu())
        lab_list.append(labels.cpu())

    return (torch.cat(eva_list), torch.cat(conv_list), torch.cat(lab_list))


def _train_probe_and_predict(train_x, train_y, test_x, test_y, num_classes,
                             device, logger, name, epochs=60, lr=1e-3, batch_size=256):
    """
    Trains a fresh ClassifierHead on cached frozen features (a linear/MLP probe) and
    returns (y_true, y_pred_probs) on the test features.

    The probe protocol is identical across all branch conditions (EVA-only,
    ConvNeXt-only, Concat), so the ONLY variable is the feature source — which is what
    makes the single-branch comparison interpretable and publishable. The training
    stream is class-balanced (the train loader uses a WeightedRandomSampler), so a plain
    cross-entropy objective already targets balanced accuracy.
    """
    in_features = train_x.size(1)
    head = ClassifierHead(
        in_features=in_features,
        hidden_features=in_features // 2,
        num_classes=num_classes,
        dropout=config.FUSION_DROPOUT,
    ).to(device)

    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()

    ds = TensorDataset(train_x, train_y)
    dl = _TensorLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    head.train()
    for ep in range(epochs):
        running = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(head(xb), yb)
            loss.backward()
            opt.step()
            running += loss.item() * xb.size(0)
        sched.step()
        if (ep + 1) % 20 == 0 or ep == 0:
            logger.info(f"    [{name}] probe epoch {ep+1}/{epochs} — loss {running/len(ds):.4f}")

    head.eval()
    probs_list = []
    with torch.no_grad():
        for xb, _ in _TensorLoader(TensorDataset(test_x, test_y), batch_size=512, shuffle=False):
            xb = xb.to(device)
            probs_list.append(torch.softmax(head(xb), dim=1).cpu().float().numpy())
    y_pred_probs = np.concatenate(probs_list, axis=0)
    return test_y.numpy(), y_pred_probs


def run_branch_probes(model, unet, train_loader, test_loader, device, logger,
                      mel_idx, max_train_batches):
    """
    Frozen-feature linear-probe ablation for per-branch contribution.

    Extracts frozen features once for the (class-balanced) train stream and the test
    set, then trains an identical fresh head on: EVA-02 alone, ConvNeXt alone, and their
    concatenation. Returns a list of result dicts ready to append to the study table.
    """
    logger.info(f"Extracting frozen TRAIN features for probes (max_batches={max_train_batches})...")
    tr_eva, tr_conv, tr_y = _extract_branch_features(
        model, unet, train_loader, device, max_batches=max_train_batches, desc="Probe train feats")
    logger.info(f"Extracting frozen TEST features for probes...")
    te_eva, te_conv, te_y = _extract_branch_features(
        model, unet, test_loader, device, max_batches=None, desc="Probe test feats")

    tr_cat = torch.cat([tr_eva, tr_conv], dim=1)
    te_cat = torch.cat([te_eva, te_conv], dim=1)

    probe_specs = [
        ("EVA-02 Branch (frozen-feature probe)",   tr_eva,  te_eva),
        ("ConvNeXt Branch (frozen-feature probe)", tr_conv, te_conv),
        ("Concat Fusion (frozen-feature probe)",   tr_cat,  te_cat),
    ]

    out = []
    for name, trx, tex in probe_specs:
        logger.info(f"Training probe: {name}  (in_features={trx.size(1)})")
        y_true, y_pred_probs = _train_probe_and_predict(
            trx, tr_y, tex, te_y, config.NUM_CLASSES, device, logger, name)
        metrics = compute_metrics(y_true, y_pred_probs)
        log_metrics(metrics, logger, prefix=name)
        out.append({
            "configuration": name,
            "accuracy": metrics['accuracy'],
            "balanced_accuracy": metrics['balanced_accuracy'],
            "macro_f1": metrics['macro_f1'],
            "macro_auc": metrics['macro_auc'],
            "ece": metrics['ece'],
            "mel_sensitivity": metrics['per_class_sensitivity'].get('mel', 0.0),
            "mel_specificity": metrics['per_class_specificity'].get('mel', 0.0),
        })
    return out


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
    # train_loader is class-balanced (WeightedRandomSampler) and is used to build the
    # frozen-feature dataset for the single-branch linear probes.
    train_loader, _, test_loader, _ = get_unified_dataloaders(
        config.DATA_DIR, masks_dir=os.path.join(config.DATA_DIR, "masks")
    )
    # Number of (balanced) train batches to extract for the probes. With BATCH_SIZE=2
    # this is ~6k balanced samples, ample for a frozen-feature probe. Raise for more.
    PROBE_TRAIN_BATCHES = int(os.environ.get("PROBE_TRAIN_BATCHES", 3000))
    
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
    
    # NOTE: "ConvNeXt Only" / "EVA-02 Only" are intentionally NOT in this list. Bypassing
    # a branch into the end-to-end-trained head is off-manifold and produces sub-random
    # AUC (a measurement artifact). Per-branch contribution is instead measured by the
    # frozen-feature linear probes below (run_branch_probes), which is the publishable
    # methodology.
    # Optional fast mode: cap TTA views (e.g. ABLATION_MAX_VIEWS=1 → no TTA, ~4× faster).
    # Useful when the Kaggle session / power / network is unstable and you want the run
    # to fit in a short window. Novelty deltas stay valid as long as you compare against
    # the matching no-TTA Full Model (= Ablation 1).
    max_views = int(os.environ.get("ABLATION_MAX_VIEWS", 5))

    configs_to_run = [
        {"name": "Ablation 1: No TTA", "n_views": 1},
        {"name": "Ablation 5: No Segmentation", "n_views": 5},
        # Novelty ablations (eval-time toggles on the SAME trained full model):
        {"name": "Ablation 6: No Uncertainty Bias", "n_views": 5},   # Novelty #2 off
        {"name": "Ablation 7: No Mirror-Asymmetry", "n_views": 5},   # Novelty #3 off
        {"name": "Ablation 8: Plain Spatial Fusion", "n_views": 5},  # #2 + #3 off
        {"name": "Full Model", "n_views": 5},
    ]

    csv_save_path = os.path.join(config.OUTPUT_DIR, "ablation_study_results.csv")

    # ── Resume support: load already-completed rows so a crashed/timed-out run can be
    #    re-launched and pick up where it stopped (each row is saved as soon as it is
    #    computed, below). Delete the CSV to force a clean re-run.
    results = []
    done = set()
    if os.path.exists(csv_save_path):
        prev = pd.read_csv(csv_save_path)
        results = prev.to_dict("records")
        done = set(prev["configuration"].astype(str).tolist())
        logger.info(f"[RESUME] Found {len(done)} completed row(s) in {csv_save_path}: {sorted(done)}")

    def _save():
        pd.DataFrame(results).to_csv(csv_save_path, index=False)

    for run_cfg in configs_to_run:
        name = run_cfg["name"]
        if name in done:
            logger.info(f"[RESUME] Skipping already-completed: {name}")
            continue
        n_views = min(run_cfg["n_views"], max_views)
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
        _save()   # incremental save — survives a later crash/timeout
        logger.info(f"[SAVED] {name} → {csv_save_path}")

    # ── Frozen-feature linear probes (per-branch contribution) ────────────── #
    # Replaces the invalid "single branch through trained head" ablations.
    probe_names = [
        "EVA-02 Branch (frozen-feature probe)",
        "ConvNeXt Branch (frozen-feature probe)",
        "Concat Fusion (frozen-feature probe)",
    ]
    if all(n in done for n in probe_names):
        logger.info("[RESUME] All probe rows already present; skipping probe stage.")
    else:
        logger.info("Running frozen-feature linear probes for per-branch contribution...")
        probe_results = run_branch_probes(
            model=model, unet=unet, train_loader=train_loader, test_loader=test_loader,
            device=config.DEVICE, logger=logger, mel_idx=mel_idx,
            max_train_batches=PROBE_TRAIN_BATCHES,
        )
        for r in probe_results:
            if r["configuration"] not in done:
                results.append(r)
        _save()

    logger.info(f"Ablation results saved to: {csv_save_path}")
    results_df = pd.DataFrame(results)
    
    # Plot results
    plot_save_path = os.path.join(config.PLOTS_DIR, "ablation_study_bar.png")
    plot_ablation_results(results_df, plot_save_path)
    logger.info(f"Ablation comparison plot saved to: {plot_save_path}")


if __name__ == "__main__":
    main()
