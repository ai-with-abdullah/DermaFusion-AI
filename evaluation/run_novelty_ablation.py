"""
Fast Novelty Ablation — proves the architecture contributions in ~40 min
========================================================================
The full run_ablation_study.py evaluates 9 configs with TTA (~6 h). For the
PAPER, the rows that matter are the ones that isolate YOUR novelties. This
script runs only those, single-view (no TTA), on the trained full model via the
eval-time flags — NO retraining:

  • Full Model              — BUG-Attn + Mirror-Asymmetry ON
  • No Uncertainty Bias     — BUG-Attn's γ·u−δ(1−p) OFF  (Novelty #2 ablation)
  • No Mirror-Asymmetry     — MAA OFF                     (Novelty #3 ablation)
  • Plain Spatial Fusion    — both OFF (≈ the published base dual-branch design)

(The SALA ablation — Novelty #1 — needs the separately-trained --no-sala model;
evaluate that with `python evaluate.py --no-sala` and add it to the table.)

Writes outputs/novelty_ablation_results.csv.

Run:  PYTHONPATH=. python -m evaluation.run_novelty_ablation
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
for k in list(sys.modules.keys()):
    if k == 'datasets' or k.startswith('datasets.'):
        sys.modules.pop(k)

import numpy as np
import pandas as pd
import torch
from torch.amp import autocast
from tqdm import tqdm

from configs.config import config
from utils.seed import seed_everything
from datasets.unified_dataset import get_unified_dataloaders
from models.transformer_unet import SwinTransformerUNet
from models.unet import LightweightUNet
from models.dual_branch_fusion import DualBranchFusionClassifier
from evaluation.metrics import compute_metrics
from training.train_utils import apply_mask


# Eval-time flag combinations that isolate each architecture novelty.
ABLATIONS = [
    ("Full Model",            dict(disable_uncertainty_bias=False, disable_asymmetry=False)),
    ("No Uncertainty Bias",   dict(disable_uncertainty_bias=True,  disable_asymmetry=False)),
    ("No Mirror-Asymmetry",   dict(disable_uncertainty_bias=False, disable_asymmetry=True)),
    ("Plain Spatial Fusion",  dict(disable_uncertainty_bias=True,  disable_asymmetry=True)),
]


@torch.no_grad()
def run(model, unet, loader, device):
    """One pass over the test set; collect softmax probs for every ablation config."""
    model.eval(); unet.eval()
    labels_all = []
    probs_all = {name: [] for name, _ in ABLATIONS}

    for batch in tqdm(loader, desc="novelty-ablation"):
        images = batch['image'].to(device)
        labels_all.append(batch['label'].numpy())
        with autocast('cuda', enabled=(device == 'cuda')):
            mask_logits = unet(images)
            images_seg  = apply_mask(images, mask_logits)
            mask_prob   = torch.sigmoid(mask_logits.float())
            for name, flags in ABLATIONS:
                logits, _ = model(images, images_seg, mask_prob, **flags)
                probs_all[name].append(torch.softmax(logits, 1).float().cpu().numpy())

    y = np.concatenate(labels_all)
    return y, {name: np.concatenate(p) for name, p in probs_all.items()}


def main():
    seed_everything(config.SEED)
    dev = config.DEVICE

    _, _, test_loader, _ = get_unified_dataloaders(
        config.DATA_DIR, masks_dir=os.path.join(config.DATA_DIR, "masks"))

    unet = (SwinTransformerUNet(pretrained=False) if config.SEG_MODEL == 'swin_unet'
            else LightweightUNet(3, 1)).to(dev)
    unet.load_state_dict(torch.load(os.path.join(config.WEIGHTS_DIR, "best_unet.pth"),
                                    map_location=dev, weights_only=True))
    model = DualBranchFusionClassifier(
        eva02_name=config.EVA02_BACKBONE, eva02_pretrained=False,
        convnext_name=config.CONVNEXT_BACKBONE, convnext_pretrained=False,
        fusion_dim=config.FUSION_EMBED_DIM, num_heads=config.FUSION_NUM_HEADS,
        num_classes=config.NUM_CLASSES, dropout=config.FUSION_DROPOUT,
        use_spatial_fusion=getattr(config, 'USE_SPATIAL_FUSION', True),
        fusion_grid=getattr(config, 'FUSION_GRID', 14)).to(dev)
    model.load_state_dict(torch.load(
        os.path.join(config.WEIGHTS_DIR, "best_dual_branch_fusion.pth"),
        map_location=dev, weights_only=True))

    y, probs = run(model, unet, test_loader, dev)

    rows = []
    for name, _ in ABLATIONS:
        m = compute_metrics(y, probs[name])
        rows.append({
            "configuration":      name,
            "accuracy":           round(m['accuracy'], 4),
            "balanced_accuracy":  round(m['balanced_accuracy'], 4),
            "macro_f1":           round(m['macro_f1'], 4),
            "macro_auc":          round(m['macro_auc'], 4),
            "ece":                round(m['ece'], 4),
        })

    df = pd.DataFrame(rows)
    out = os.path.join(config.OUTPUT_DIR, "novelty_ablation_results.csv")
    df.to_csv(out, index=False)

    print("\n" + "=" * 78)
    print("  NOVELTY ABLATION (single-view, eval-time toggles on the trained model)")
    print("=" * 78)
    print(df.to_string(index=False))
    print("=" * 78)
    print(f"\nSaved → {out}")
    full = df[df.configuration == "Full Model"].iloc[0]
    print("\nContribution of each novelty (Full minus ablation, on macro_auc / balanced_accuracy):")
    for name, _ in ABLATIONS[1:]:
        r = df[df.configuration == name].iloc[0]
        print(f"  {name:24s}: ΔAUC {full.macro_auc - r.macro_auc:+.4f}   "
              f"ΔBalAcc {full.balanced_accuracy - r.balanced_accuracy:+.4f}")
    print("\n(Positive Δ = the novelty helps. Add the No-SALA row from "
          "`python evaluate.py --no-sala` for Novelty #1.)")


if __name__ == "__main__":
    main()
