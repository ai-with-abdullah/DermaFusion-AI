"""
Decoupled Rebalancing Study — the methods contribution
======================================================
Motivation (from our ablations): on multi-source dermoscopy, the standard
imbalance fix — logit adjustment — BACKFIRES, collapsing the rare classes
(SALA dropped df F1 to 0.22, vasc to 0.30). This study isolates the FIX in the
classifier head, on FROZEN backbone features (the "decoupling" recipe), and
compares four ways to train that head:

  A. Plain                    — class-weighted CE, no special rebalancing
  B. Per-Source Logit Adj.    — SALA-style per-source class-prior margin (the failure)
  C. Global Logit Adj.        — Menon et al. 2021, single global class-prior margin
  D. Decoupled Rebalance (ours) — class-AND-source-balanced sampling + LDAM rare-class
                                  margin. The multi-source-aware fix.

CLAIM TO VALIDATE: D beats A/B/C, especially on the rare classes (df, vasc),
without hurting the common ones — i.e. it fixes the multi-source rare-class
collapse that logit adjustment cannot.

Two phases (auto): (1) extract & cache frozen concat features for train/val/test
once (~1h); (2) train the four heads on the cache (~minutes) and report.

Run:  PYTHONPATH=. python -m evaluation.decoupled_study            # extract + compare
      PYTHONPATH=. python -m evaluation.decoupled_study --compare  # reuse cache, fast
"""

import sys, os, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
for k in list(sys.modules.keys()):
    if k == 'datasets' or k.startswith('datasets.'):
        sys.modules.pop(k)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import autocast
from tqdm import tqdm

from configs.config import config
from utils.seed import seed_everything
from datasets.unified_dataset import get_unified_dataloaders, NUM_SOURCES
from models.transformer_unet import SwinTransformerUNet
from models.unet import LightweightUNet
from models.dual_branch_fusion import DualBranchFusionClassifier
from evaluation.metrics import compute_metrics
from training.train_utils import apply_mask

CACHE = os.path.join(config.OUTPUT_DIR, "decoupled_features")


# =========================================================================== #
#                    PHASE 1 — frozen feature extraction                       #
# =========================================================================== #

@torch.no_grad()
def extract(model, unet, loader, device):
    """Concat of FROZEN pooled EVA-02 + ConvNeXt features (the representation)."""
    model.eval(); unet.eval()
    feats, labels, sources = [], [], []
    for batch in tqdm(loader, desc="extract"):
        images = batch['image'].to(device)
        with autocast('cuda', enabled=(device == 'cuda')):
            mask_logits = unet(images)
            images_seg  = apply_mask(images, mask_logits)
            f_eva  = model.branch_eva(images)        # (B, eva_dim)
            f_conv = model.branch_conv(images_seg)   # (B, conv_dim)
            f = torch.cat([f_eva, f_conv], dim=1)    # (B, eva+conv)
        feats.append(f.float().cpu().numpy())
        labels.append(batch['label'].numpy())
        sources.append(batch['source_id'].numpy())
    return np.concatenate(feats), np.concatenate(labels), np.concatenate(sources)


def cache_features():
    seed_everything(config.SEED); dev = config.DEVICE
    train_loader, val_loader, test_loader, _ = get_unified_dataloaders(
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
    # frozen no-SALA backbone (the best representation)
    wname = "best_dual_branch_fusion_nosala.pth"
    if not os.path.exists(os.path.join(config.WEIGHTS_DIR, wname)):
        wname = "best_dual_branch_fusion.pth"
    model.load_state_dict(torch.load(os.path.join(config.WEIGHTS_DIR, wname),
                                     map_location=dev, weights_only=True))
    print(f"[extract] frozen backbone: {wname}")

    os.makedirs(CACHE, exist_ok=True)
    for split, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        X, y, s = extract(model, unet, loader, dev)
        np.savez(os.path.join(CACHE, f"{split}.npz"), X=X, y=y, s=s)
        print(f"  cached {split}: X{X.shape} y{y.shape}")


# =========================================================================== #
#                    PHASE 2 — train the four heads                            #
# =========================================================================== #

def _load(split):
    d = np.load(os.path.join(CACHE, f"{split}.npz"))
    return d['X'], d['y'], d['s']


def _priors(y, C):
    counts = np.bincount(y, minlength=C).astype(np.float64)
    return counts / counts.sum(), counts


def _train_head(Xtr, ytr, str_, Xva, yva, C, device, method, epochs=60, tau=1.0):
    """Train a linear head on frozen features. `method` selects the rebalancing."""
    D = Xtr.shape[1]
    head = nn.Linear(D, C).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.long, device=device)
    Xva_t = torch.tensor(Xva, dtype=torch.float32, device=device)

    glob_prior, counts = _priors(ytr, C)
    glob_logprior = torch.tensor(np.log(glob_prior + 1e-12), dtype=torch.float32, device=device)
    # per-source class log-priors (for B)
    src_logprior = torch.zeros(NUM_SOURCES, C, device=device)
    for d in range(NUM_SOURCES):
        m = (str_ == d)
        if m.sum() > 0:
            p, _ = _priors(ytr[m], C); src_logprior[d] = torch.tensor(np.log(p + 1e-12), device=device)
    str_t = torch.tensor(str_, dtype=torch.long, device=device)
    # LDAM per-class margins (for D): bigger margin for rarer classes
    ldam = 0.5 * (counts.max() ** 0.25) / (np.maximum(counts, 1) ** 0.25)
    ldam_t = torch.tensor(ldam, dtype=torch.float32, device=device)

    # sampling distribution
    N = len(ytr)
    if method == "D":  # class- AND source-balanced sampling
        cls_w = 1.0 / np.maximum(counts[ytr], 1)
        srccnt = np.bincount(str_, minlength=NUM_SOURCES).astype(np.float64)
        src_w = 1.0 / np.maximum(srccnt[str_], 1)
        w = cls_w * src_w
    else:
        w = np.ones(N)
    w = w / w.sum()

    # class-weighted CE weight (for A): inverse-frequency
    ce_w = torch.tensor((counts.sum() / (C * np.maximum(counts, 1))), dtype=torch.float32, device=device)

    bs = 4096
    best_va, best_state = -1, None
    for ep in range(epochs):
        idx = np.random.choice(N, size=N, replace=True, p=w)  # weighted resample / epoch
        head.train()
        for i in range(0, N, bs):
            b = idx[i:i + bs]
            xb, yb, sb = Xtr_t[b], ytr_t[b], str_t[b]
            logits = head(xb)
            if method == "A":
                loss = nn.functional.cross_entropy(logits, yb, weight=ce_w)
            elif method == "B":   # per-source logit adjustment (SALA-style)
                loss = nn.functional.cross_entropy(logits + tau * src_logprior[sb], yb)
            elif method == "C":   # global logit adjustment (Menon)
                loss = nn.functional.cross_entropy(logits + tau * glob_logprior[None, :], yb)
            else:                  # D: LDAM margin on the true class + balanced sampling
                adj = logits.clone()
                adj[torch.arange(len(yb)), yb] -= ldam_t[yb]
                loss = nn.functional.cross_entropy(adj, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        # val (raw logits)
        head.eval()
        with torch.no_grad():
            va_pred = head(Xva_t).argmax(1).cpu().numpy()
        from sklearn.metrics import balanced_accuracy_score
        va = balanced_accuracy_score(yva, va_pred)
        if va > best_va:
            best_va, best_state = va, {k: v.detach().clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state)
    return head


def compare():
    seed_everything(config.SEED)
    dev = config.DEVICE
    C = config.NUM_CLASSES
    Xtr, ytr, str_ = _load("train")
    Xva, yva, _    = _load("val")
    Xte, yte, _    = _load("test")
    print(f"features: train{Xtr.shape} val{Xva.shape} test{Xte.shape}")

    from sklearn.metrics import f1_score
    methods = {"A": "Plain (class-weighted CE)",
               "B": "Per-Source Logit Adj (SALA)",
               "C": "Global Logit Adj (Menon)",
               "D": "Decoupled Rebalance (OURS)"}
    rare = [config.CLASSES.index(c) for c in ('df', 'vasc') if c in config.CLASSES]
    rows = []
    for m, label in methods.items():
        head = _train_head(Xtr, ytr, str_, Xva, yva, C, dev, m)
        with torch.no_grad():
            probs = torch.softmax(head(torch.tensor(Xte, dtype=torch.float32, device=dev)), 1).cpu().numpy()
        met = compute_metrics(yte, probs)
        per_f1 = f1_score(yte, probs.argmax(1), average=None, labels=list(range(C)), zero_division=0)
        rows.append({
            "method": label,
            "balanced_acc": round(met['balanced_accuracy'], 4),
            "macro_f1":     round(met['macro_f1'], 4),
            "macro_auc":    round(met['macro_auc'], 4),
            "df_f1":        round(float(per_f1[config.CLASSES.index('df')]), 4),
            "vasc_f1":      round(float(per_f1[config.CLASSES.index('vasc')]), 4),
            "rare_mean_f1": round(float(np.mean([per_f1[i] for i in rare])), 4),
        })

    df = pd.DataFrame(rows)
    out = os.path.join(config.OUTPUT_DIR, "decoupled_study_results.csv")
    df.to_csv(out, index=False)
    print("\n" + "=" * 92)
    print("  DECOUPLED REBALANCING STUDY (frozen features, head-only) — the methods contribution")
    print("=" * 92)
    print(df.to_string(index=False))
    print("=" * 92)
    print(f"saved → {out}")
    print("\nCLAIM: row D (OURS) should have the highest rare_mean_f1 (df+vasc) while keeping")
    print("macro_auc competitive — i.e. it fixes the multi-source rare-class collapse that B fails on.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", action="store_true", help="skip extraction, reuse cached features")
    args = ap.parse_args()
    if not args.compare and not os.path.exists(os.path.join(CACHE, "test.npz")):
        cache_features()
    elif not args.compare:
        print(f"[cache exists at {CACHE}] — re-extracting. Use --compare to reuse.")
        cache_features()
    compare()


if __name__ == "__main__":
    main()
