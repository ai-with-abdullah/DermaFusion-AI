"""
Zero-shot external validation on PAD-UFES-20  (generalization test)
==================================================================
Tests the DEPLOYED ISIC-trained model on the PAD-UFES-20 clinical
(smartphone) dataset with NO fine-tuning and NO retraining — a true
cross-domain generalization test (the honest reviewer-requested experiment).

Differs from the older test_padufes.py, which loaded a PAD-UFES *fine-tuned*
head (padufes_finetuned_head.pth) — that is adaptation, not zero-shot, and
must not be reported as external generalization.

  • Loads best_dual_branch_fusion_nosala.pth (the deployed model) as-is.
  • Runs the real pipeline: Swin-U-Net mask -> dual-branch classifier, 5-view TTA.
  • PAD-UFES-20 has 6 clinical classes mapping onto 5 of our 7 (no df/vasc);
    metrics are computed over the classes actually present.
  • Reports BOTH raw and CLAHE-preprocessed so the effect of enhancement is
    transparent (report the raw number as the honest headline).

Expect a substantial drop vs the in-domain test set: PAD-UFES is clinical
photography, the model and segmentation net were trained on dermoscopy. That
domain-shift drop is the point of the experiment; report it honestly.

Run (Kaggle / CUDA, from repo root):
    PYTHONPATH=. python -m evaluation.run_external_padufes            # raw + CLAHE, all images
    PYTHONPATH=. python -m evaluation.run_external_padufes --n 300    # quick subset

Data layout expected:
    data/metadata.csv                       (cols: img_id, diagnostic)
    data/pad_ufes/images/imgs_part_{1,2,3}/ (*.png)
Outputs: outputs/padufes_external/  (metrics.csv, predictions.csv, confusion_matrix.png)
"""

import os, sys, glob, argparse, warnings
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
for k in list(sys.modules.keys()):
    if k == "datasets" or k.startswith("datasets."):
        sys.modules.pop(k)
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import (balanced_accuracy_score, f1_score, roc_auc_score,
                             confusion_matrix, recall_score)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from configs.config import config
from models.dual_branch_fusion import DualBranchFusionClassifier
from models.transformer_unet import SwinTransformerUNet
from models.unet import LightweightUNet
from training.tta import TTAInference

# PAD-UFES-20 (6 clinical classes) -> our 7-class scheme (no df/vasc present)
PAD_TO_MODEL = {"MEL": "mel", "BCC": "bcc", "NEV": "nv",
                "ACK": "akiec", "SEK": "bkl", "SCC": "akiec"}


def find_metadata():
    """Locate the PAD-UFES metadata.csv wherever the Kaggle dataset linked it."""
    import pandas as _pd
    quick = [os.path.join(config.DATA_DIR, "metadata.csv"),
             os.path.join(config.DATA_DIR, "pad_ufes", "metadata.csv"),
             os.path.join(config.DATA_DIR, "pad_ufes", "pad_ufes", "metadata.csv")]
    roots = [os.path.join(config.DATA_DIR, "pad_ufes"), config.DATA_DIR]
    for r in roots:
        if os.path.isdir(r):
            for dp, _, fs in os.walk(r):
                for f in fs:
                    if f.lower() == "metadata.csv":
                        quick.append(os.path.join(dp, f))
    for c in quick:
        if os.path.exists(c):
            try:
                cols = set(_pd.read_csv(c, nrows=1).columns)
                if {"img_id", "diagnostic"} <= cols:
                    return c
            except Exception:
                pass
    return None


def find_image_dirs():
    """All directories under data/pad_ufes that contain .png images."""
    base = os.path.join(config.DATA_DIR, "pad_ufes")
    dirs = []
    if os.path.isdir(base):
        for dp, _, fs in os.walk(base):
            if any(f.lower().endswith(".png") for f in fs):
                dirs.append(dp)
    return dirs


def enhance(img: Image.Image) -> Image.Image:
    """CLAHE on L channel + mild sharpening (optional domain-shift reducer)."""
    a = np.array(img)
    lab = cv2.cvtColor(a, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32)
    k = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
    out = np.clip(cv2.filter2D(out, -1, k), 0, 255).astype(np.uint8)
    return Image.fromarray(out)


class PadUfes(Dataset):
    def __init__(self, df, img_dirs, tfm, use_clahe):
        self.df = df.reset_index(drop=True); self.tfm = tfm; self.use_clahe = use_clahe
        self.cls2idx = {c: i for i, c in enumerate(config.CLASSES)}
        self.pmap = {}
        for d in img_dirs:
            if os.path.exists(d):
                for f in glob.glob(os.path.join(d, "*.png")):
                    self.pmap[os.path.basename(f)] = f

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        label = self.cls2idx[PAD_TO_MODEL.get(row["diagnostic"], "nv")]
        path = self.pmap.get(row["img_id"])
        if path is None:
            img = Image.fromarray(np.zeros((448, 448, 3), np.uint8))
        else:
            img = Image.open(path).convert("RGB")
            if self.use_clahe:
                img = enhance(img)
        return self.tfm(img), label, str(row["diagnostic"])


def evaluate(use_clahe, df, img_dirs, model, unet, device, n_views):
    tfm = transforms.Compose([transforms.Resize((448, 448)), transforms.ToTensor(),
                              transforms.Normalize(mean=config.MEAN, std=config.STD)])
    ds = PadUfes(df, img_dirs, tfm, use_clahe)
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
    tta = TTAInference(model, unet, device, n_views=n_views)   # correct spatial-fusion path

    y_true, y_prob, pad_lab = [], [], []
    for images, labels, pad in tqdm(loader, desc=f"PAD-UFES {'CLAHE' if use_clahe else 'raw'}"):
        y_prob.extend(tta.predict(images).tolist())
        y_true.extend(labels.tolist()); pad_lab.extend(list(pad))
    y_true = np.array(y_true); y_prob = np.array(y_prob); y_pred = y_prob.argmax(1)

    present = sorted(set(y_true.tolist()))                       # classes actually in PAD-UFES (5)
    names = [config.CLASSES[c] for c in present]
    acc = float((y_pred == y_true).mean())
    bal = float(balanced_accuracy_score(y_true, y_pred))
    mf1 = float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0))
    aucs = []
    for c in present:
        yb = (y_true == c).astype(int)
        if 0 < yb.sum() < len(yb):
            try: aucs.append(roc_auc_score(yb, y_prob[:, c]))
            except Exception: pass
    mauc = float(np.mean(aucs)) if aucs else float("nan")
    perf1 = f1_score(y_true, y_pred, labels=present, average=None, zero_division=0)
    return dict(acc=acc, bal=bal, mf1=mf1, mauc=mauc, present=present, names=names,
                perf1=dict(zip(names, [round(float(v), 3) for v in perf1])),
                y_true=y_true, y_pred=y_pred, y_prob=y_prob, pad=np.array(pad_lab))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None, help="evaluate a stratified subset of N images (default all)")
    ap.add_argument("--no-clahe", action="store_true", help="skip the CLAHE-enhanced pass")
    args = ap.parse_args()
    dev = config.DEVICE

    csv = find_metadata()
    img_dirs = find_image_dirs()
    if csv is None or not img_dirs:
        raise SystemExit(
            "PAD-UFES-20 not found. Attach the dataset in Kaggle (Add Input -> search "
            "'PAD-UFES-20') so it links under data/pad_ufes/ with metadata.csv + the "
            f"image folders.\n  metadata.csv found: {csv}\n  image dirs found: {img_dirs}")
    print(f"[external] metadata: {csv}\n[external] image dirs: {img_dirs}")
    df = pd.read_csv(csv)
    df = df[df["diagnostic"].isin(PAD_TO_MODEL)].copy()
    print(f"PAD-UFES-20: {len(df)} images | classes: {df['diagnostic'].value_counts().to_dict()}")
    if args.n and args.n < len(df):
        df = df.groupby("diagnostic", group_keys=False).apply(
            lambda x: x.sample(max(1, round(args.n * len(x) / len(df))), random_state=42))
        print(f"  stratified subset: {len(df)} images")

    unet = (SwinTransformerUNet(pretrained=False) if config.SEG_MODEL == "swin_unet"
            else LightweightUNet(3, 1)).to(dev)
    unet.load_state_dict(torch.load(os.path.join(config.WEIGHTS_DIR, "best_unet.pth"),
                                    map_location=dev, weights_only=True)); unet.eval()
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
                                     map_location=dev, weights_only=True)); model.eval()
    print(f"[external] ZERO-SHOT model: {wname} (NO fine-tuning)")

    passes = [False] if args.no_clahe else [False, True]
    rows = []
    keep = None
    for uc in passes:
        r = evaluate(uc, df, img_dirs, model, unet, dev, config.TTA_N_VIEWS)
        rows.append({"preprocessing": "CLAHE" if uc else "raw",
                     "accuracy": round(r["acc"], 4), "balanced_acc": round(r["bal"], 4),
                     "macro_f1": round(r["mf1"], 4), "macro_auc": round(r["mauc"], 4),
                     **{f"F1_{k}": v for k, v in r["perf1"].items()}})
        if not uc: keep = r

    out = os.path.join(config.OUTPUT_DIR, "padufes_external"); os.makedirs(out, exist_ok=True)
    dfres = pd.DataFrame(rows); dfres.to_csv(os.path.join(out, "metrics.csv"), index=False)
    print("\n" + "=" * 80)
    print("  ZERO-SHOT EXTERNAL VALIDATION — PAD-UFES-20 (clinical) | model trained on dermoscopy")
    print("=" * 80)
    print(dfres.to_string(index=False))
    print("=" * 80)
    print("Report the RAW row as the honest zero-shot result; CLAHE is an optional domain-shift reducer.")
    print("Note: only 5 of 7 classes are present in PAD-UFES-20 (no df/vasc); metrics are over present classes.")

    # confusion matrix for the raw pass
    present = keep["present"]; names = keep["names"]
    cm = confusion_matrix(keep["y_true"], keep["y_pred"], labels=present)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=names, yticklabels=names)
    plt.title("PAD-UFES-20 zero-shot (dermoscopy model → clinical photos)")
    plt.ylabel("True"); plt.xlabel("Predicted"); plt.tight_layout()
    plt.savefig(os.path.join(out, "confusion_matrix.png"), dpi=150); plt.close()
    pd.DataFrame({"pad_label": keep["pad"],
                  "true": [config.CLASSES[c] for c in keep["y_true"]],
                  "pred": [config.CLASSES[c] for c in keep["y_pred"]]}
                 ).to_csv(os.path.join(out, "predictions.csv"), index=False)
    print(f"saved -> {out}/  (metrics.csv, predictions.csv, confusion_matrix.png)")


if __name__ == "__main__":
    main()
