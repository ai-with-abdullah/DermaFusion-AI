"""
Segmentation On/Off Ablation  (paper comment #261)
==================================================
Question (reviewer): "Did segmentation actually improve classification?"

We answer it with the SAME frozen-feature methodology the paper already uses
for its architecture ablation (Table 6), so the numbers are directly
comparable and the experiment is cheap (no full retraining):

  Both configurations share the identical, frozen, trained backbones and an
  identical linear head / optimiser / budget. The ONLY difference is the image
  fed to the local (ConvNeXt) branch:

    WITH segmentation  : ConvNeXt sees the Swin-U-Net mask-applied image
    WITHOUT segmentation: ConvNeXt sees the ORIGINAL image (no mask)

  The global (EVA-02) branch always sees the original image, so any difference
  isolates the value of masking the local branch.

Pooled features use forward_tokens() + global-average-pool on the TRAINED
trunk (NOT the pooled projector, which is untrained), so both feature sets are
faithful representations of the fine-tuned backbones.

Run (Kaggle / CUDA, from repo root):
    PYTHONPATH=. python -m evaluation.run_segmentation_ablation
    PYTHONPATH=. python -m evaluation.run_segmentation_ablation --seeds 5

Outputs:
    • printed table  Configuration | Macro-F1 | Balanced acc | Rare-mean F1
    • outputs/segmentation_ablation.csv
"""

import os, sys, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
for k in list(sys.modules.keys()):
    if k == "datasets" or k.startswith("datasets."):
        sys.modules.pop(k)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import autocast
from tqdm import tqdm
from sklearn.metrics import f1_score, balanced_accuracy_score

from configs.config import config
from utils.seed import seed_everything
from datasets.unified_dataset import get_unified_dataloaders
from models.transformer_unet import SwinTransformerUNet
from models.unet import LightweightUNet
from models.dual_branch_fusion import DualBranchFusionClassifier
from evaluation.metrics import compute_metrics
from training.train_utils import apply_mask

CACHE = os.path.join(config.OUTPUT_DIR, "seg_ablation_features")


# --------------------------------------------------------------------------- #
#                         PHASE 1 — feature extraction                         #
# --------------------------------------------------------------------------- #

@torch.no_grad()
def _pool(tokens):
    """(B,C,G,G) -> (B,C) global-average-pool."""
    return tokens.mean(dim=(2, 3))


@torch.no_grad()
def extract(model, unet, loader, device):
    """Return (X_with_seg, X_no_seg, y). EVA features shared across both."""
    model.eval(); unet.eval()
    Xw, Xn, ys = [], [], []
    for batch in tqdm(loader, desc="extract"):
        images = batch["image"].to(device)
        with autocast("cuda", enabled=(device == "cuda")):
            mask_logits = unet(images)
            images_seg  = apply_mask(images, mask_logits)
            f_eva  = _pool(model.branch_eva.forward_tokens(images))          # original img
            f_conv_seg = _pool(model.branch_conv.forward_tokens(images_seg)) # masked img
            f_conv_raw = _pool(model.branch_conv.forward_tokens(images))     # original img
        Xw.append(torch.cat([f_eva, f_conv_seg], 1).float().cpu().numpy())
        Xn.append(torch.cat([f_eva, f_conv_raw], 1).float().cpu().numpy())
        ys.append(batch["label"].numpy())
    return np.concatenate(Xw), np.concatenate(Xn), np.concatenate(ys)


def cache_features():
    seed_everything(config.SEED); dev = config.DEVICE
    tr, va, te, _ = get_unified_dataloaders(
        config.DATA_DIR, masks_dir=os.path.join(config.DATA_DIR, "masks"))
    unet = (SwinTransformerUNet(pretrained=False) if config.SEG_MODEL == "swin_unet"
            else LightweightUNet(3, 1)).to(dev)
    unet.load_state_dict(torch.load(os.path.join(config.WEIGHTS_DIR, "best_unet.pth"),
                                    map_location=dev, weights_only=True))
    model = DualBranchFusionClassifier(
        eva02_name=config.EVA02_BACKBONE, eva02_pretrained=False,
        convnext_name=config.CONVNEXT_BACKBONE, convnext_pretrained=False,
        fusion_dim=config.FUSION_EMBED_DIM, num_heads=config.FUSION_NUM_HEADS,
        num_classes=config.NUM_CLASSES, dropout=config.FUSION_DROPOUT,
        use_spatial_fusion=getattr(config, "USE_SPATIAL_FUSION", True),
        fusion_grid=getattr(config, "FUSION_GRID", 14)).to(dev)
    wname = "best_dual_branch_fusion_nosala.pth"
    if not os.path.exists(os.path.join(config.WEIGHTS_DIR, wname)):
        wname = "best_dual_branch_fusion.pth"
    model.load_state_dict(torch.load(os.path.join(config.WEIGHTS_DIR, wname),
                                     map_location=dev, weights_only=True))
    print(f"[extract] frozen backbone: {wname}")

    os.makedirs(CACHE, exist_ok=True)
    for split, loader in [("train", tr), ("val", va), ("test", te)]:
        Xw, Xn, y = extract(model, unet, loader, dev)
        np.savez(os.path.join(CACHE, f"{split}.npz"), Xw=Xw, Xn=Xn, y=y)
        print(f"  cached {split}: with{Xw.shape} no{Xn.shape}")


# --------------------------------------------------------------------------- #
#                         PHASE 2 — linear-probe compare                       #
# --------------------------------------------------------------------------- #

def _load(split):
    d = np.load(os.path.join(CACHE, f"{split}.npz"))
    return d["Xw"], d["Xn"], d["y"]


def _train_head(Xtr, ytr, Xva, yva, C, device, epochs=60):
    D = Xtr.shape[1]
    head = nn.Linear(D, C).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    counts = np.bincount(ytr, minlength=C).astype(np.float64)
    ce_w = torch.tensor(counts.sum() / (C * np.maximum(counts, 1)),
                        dtype=torch.float32, device=device)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.long, device=device)
    Xva_t = torch.tensor(Xva, dtype=torch.float32, device=device)
    N, bs = len(ytr), 4096
    best_va, best_state = -1, None
    for _ in range(epochs):
        perm = np.random.permutation(N)
        head.train()
        for i in range(0, N, bs):
            b = perm[i:i + bs]
            loss = nn.functional.cross_entropy(head(Xtr_t[b]), ytr_t[b], weight=ce_w)
            opt.zero_grad(); loss.backward(); opt.step()
        head.eval()
        with torch.no_grad():
            va = balanced_accuracy_score(yva, head(Xva_t).argmax(1).cpu().numpy())
        if va > best_va:
            best_va = va
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state)
    return head


def _eval(Xtr, ytr, Xva, yva, Xte, yte, C, dev):
    head = _train_head(Xtr, ytr, Xva, yva, C, dev)
    with torch.no_grad():
        probs = torch.softmax(
            head(torch.tensor(Xte, dtype=torch.float32, device=dev)), 1).cpu().numpy()
    m = compute_metrics(yte, probs)
    per = f1_score(yte, probs.argmax(1), average=None, labels=list(range(C)), zero_division=0)
    rare = [config.CLASSES.index(c) for c in ("df", "vasc") if c in config.CLASSES]
    return {"macro_f1": m["macro_f1"], "balanced_acc": m["balanced_accuracy"],
            "macro_auc": m["macro_auc"], "rare_mean_f1": float(np.mean([per[i] for i in rare]))}


def compare(seeds=1):
    dev, C = config.DEVICE, config.NUM_CLASSES
    Xw_tr, Xn_tr, ytr = _load("train")
    Xw_va, Xn_va, yva = _load("val")
    Xw_te, Xn_te, yte = _load("test")
    print(f"features: train{Xw_tr.shape} test{Xw_te.shape} | seeds={seeds}")

    configs = {"Without segmentation (ConvNeXt on original image)": (Xn_tr, Xn_va, Xn_te),
               "With segmentation (ConvNeXt on masked image)":      (Xw_tr, Xw_va, Xw_te)}
    METRICS = ["macro_f1", "balanced_acc", "macro_auc", "rare_mean_f1"]
    runs = {name: {k: [] for k in METRICS} for name in configs}
    for s in range(seeds):
        np.random.seed(config.SEED + s); torch.manual_seed(config.SEED + s)
        for name, (Xtr, Xva, Xte) in configs.items():
            r = _eval(Xtr, ytr, Xva, yva, Xte, yte, C, dev)
            for k in METRICS:
                runs[name][k].append(r[k])

    rows = []
    for name in configs:
        row = {"Configuration": name}
        for k in METRICS:
            a = np.array(runs[name][k])
            row[k] = f"{a.mean():.4f} ± {a.std():.4f}" if seeds > 1 else f"{a.mean():.4f}"
        rows.append(row)
    df = pd.DataFrame(rows)
    out = os.path.join(config.OUTPUT_DIR, f"segmentation_ablation{'_seeds%d'%seeds if seeds>1 else ''}.csv")
    df.to_csv(out, index=False)
    print("\n" + "=" * 92)
    print("  SEGMENTATION ON/OFF ABLATION (frozen features, linear head)")
    print("=" * 92)
    print(df.to_string(index=False))
    print("=" * 92)
    print(f"saved -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", action="store_true", help="reuse cached features")
    ap.add_argument("--seeds", type=int, default=1)
    args = ap.parse_args()
    ready = os.path.exists(os.path.join(CACHE, "test.npz"))
    if not (args.compare and ready):
        if not ready:
            print("[cache missing] extracting features...")
        cache_features()
    compare(seeds=args.seeds)


if __name__ == "__main__":
    main()
