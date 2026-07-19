"""
Figure 7 with error bars  (paper comment #241)
==============================================
Reviewer asked to add error bars to Figure 7 (per-class F1 for the rare lesion
classes under each imbalance strategy).

This REUSES the exact decoupled-study pipeline (same cached frozen features,
same linear heads A/B/C/D) so the figure is numerically consistent with
Table 4/5. It trains the four heads across N seeds, records per-class F1 for
the rare classes each seed, and draws a grouped bar chart with mean bars and
standard-deviation error bars.

Run (Kaggle / CUDA, from repo root):
    # if the decoupled feature cache already exists (from decoupled_study):
    PYTHONPATH=. python -m evaluation.plot_rare_f1_errorbars --seeds 5
    # otherwise it will extract features first (~1h), then plot.

Outputs:
    • outputs/plots/fig7_rare_f1_errorbars.png  (300 dpi) + .pdf
    • outputs/fig7_rare_f1_raw.csv   (per-seed values, for the record)
"""

import os, sys, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
for k in list(sys.modules.keys()):
    if k == "datasets" or k.startswith("datasets."):
        sys.modules.pop(k)

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

from configs.config import config
from evaluation.decoupled_study import CACHE, _load, _train_head, METHODS, cache_features

# short labels for the x-groups / legend
STRAT_LABELS = {"A": "Plain (CE)", "B": "Per-source LA (SALA)",
                "C": "Global LA", "D": "Decoupled (ours)"}
RARE = ["df", "vasc"]


def collect(seeds):
    dev, C = config.DEVICE, config.NUM_CLASSES
    Xtr, ytr, str_ = _load("train")
    Xva, yva, _    = _load("val")
    Xte, yte, _    = _load("test")
    rare_idx = [config.CLASSES.index(c) for c in RARE]

    # values[strategy][class] = [f1 per seed]
    values = {m: {c: [] for c in RARE} for m in METHODS}
    for s in range(seeds):
        np.random.seed(config.SEED + s); torch.manual_seed(config.SEED + s)
        print(f"--- seed {s+1}/{seeds} ---")
        for m in METHODS:
            head = _train_head(Xtr, ytr, str_, Xva, yva, C, dev, m)
            with torch.no_grad():
                preds = head(torch.tensor(Xte, dtype=torch.float32, device=dev)).argmax(1).cpu().numpy()
            per = f1_score(yte, preds, average=None, labels=list(range(C)), zero_division=0)
            for c, ci in zip(RARE, rare_idx):
                values[m][c].append(float(per[ci]))
    return values


def plot(values, seeds):
    strategies = list(METHODS.keys())            # A B C D
    x = np.arange(len(RARE))                      # one group per rare class
    width = 0.8 / len(strategies)
    colors = ["#4C72B0", "#DD8452", "#C44E52", "#55A868"]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    rows = []
    for i, m in enumerate(strategies):
        means = [np.mean(values[m][c]) for c in RARE]
        stds  = [np.std(values[m][c])  for c in RARE]
        ax.bar(x + i * width - 0.4 + width / 2, means, width,
               yerr=stds, capsize=4, label=STRAT_LABELS[m], color=colors[i],
               edgecolor="black", linewidth=0.5,
               error_kw=dict(ecolor="#333333", lw=1.0))
        for c, mu, sd in zip(RARE, means, stds):
            rows.append({"strategy": STRAT_LABELS[m], "class": c,
                         "mean_f1": round(mu, 4), "std_f1": round(sd, 4)})

    ax.set_xticks(x)
    ax.set_xticklabels([c.upper() for c in RARE])
    ax.set_ylabel("F1 score")
    ax.set_ylim(0, 1.0)
    ax.set_title(f"Rare-class F1 by imbalance strategy (mean ± s.d., {seeds} seeds)")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    png = os.path.join(config.PLOTS_DIR, "fig7_rare_f1_errorbars.png")
    pdf = os.path.join(config.PLOTS_DIR, "fig7_rare_f1_errorbars.pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    csv = os.path.join(config.OUTPUT_DIR, "fig7_rare_f1_raw.csv")
    pd.DataFrame(rows).to_csv(csv, index=False)
    print(f"\nsaved figure -> {png}\n         pdf -> {pdf}\n     raw csv -> {csv}")
    print("\n" + pd.DataFrame(rows).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    if not os.path.exists(os.path.join(CACHE, "test.npz")):
        print("[cache missing] extracting decoupled features first (~1h)...")
        cache_features()
    values = collect(args.seeds)
    plot(values, args.seeds)


if __name__ == "__main__":
    main()
