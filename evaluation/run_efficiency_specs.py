"""
Efficiency / Complexity Specifications  (paper comment #194)
===========================================================
Measures everything the reviewer asked for in one run and prints a
paper-ready specifications table:

  • GPU name + total VRAM
  • Parameter count  (total, trainable, per-component, +UNet)
  • FLOPs / GMACs     (fvcore → thop → param-based fallback)
  • Inference latency (GPU single-image, end-to-end: UNet + classifier)
  • Throughput        (images/sec at a few batch sizes)
  • Peak GPU memory   (inference; and optional single train-step)

It benchmarks the REAL deployed pipeline:
      image → Swin-UNet → soft mask → apply_mask → dual-branch classifier
so the numbers match what the paper actually runs. Model weights do NOT
affect params/FLOPs/latency/memory, so this runs even if the .pth files
are absent (it will just use randomly-initialised weights of the SAME
architecture and say so).

Run (Kaggle / any CUDA box, from repo root):
    PYTHONPATH=. python -m evaluation.run_efficiency_specs
    PYTHONPATH=. python -m evaluation.run_efficiency_specs --train-mem   # also measure a train-step peak

Outputs:
    • printed paper-ready table
    • outputs/efficiency_specs.csv
"""

import os
import sys
import time
import argparse

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

import numpy as np
import pandas as pd
import torch

from configs.config import config
from utils.seed import seed_everything
from models.transformer_unet import SwinTransformerUNet
from models.dual_branch_fusion import DualBranchFusionClassifier
from training.train_utils import apply_mask


# =========================================================================== #
#                          PARAMETER COUNTING                                  #
# =========================================================================== #

def count_parameters(model, unet):
    """Total / trainable / per-top-level-module params (in millions)."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    unet_tot  = sum(p.numel() for p in unet.parameters())

    breakdown = {}
    for name, module in model.named_children():          # robust: picks up every submodule
        n = sum(p.numel() for p in module.parameters())
        if n > 0:
            breakdown[name] = n / 1e6

    return {
        "classifier_total_M":     total / 1e6,
        "classifier_trainable_M": trainable / 1e6,
        "unet_M":                 unet_tot / 1e6,
        "grand_total_M":          (total + unet_tot) / 1e6,
        "breakdown_M":            breakdown,
    }


# =========================================================================== #
#                          FLOPs ESTIMATION                                    #
# =========================================================================== #

def estimate_flops(model, inputs, device):
    """fvcore → thop → param-based fallback. Returns a printable string."""
    model.eval()
    try:
        from fvcore.nn import FlopCountAnalysis
        fca = FlopCountAnalysis(model, tuple(t.to(device) for t in inputs))
        fca.unsupported_ops_warnings(False)
        fca.uncalled_modules_warnings(False)
        return f"{fca.total() / 1e9:.1f} GFLOPs (fvcore, some timm ops may be uncounted)"
    except Exception as e:
        pass
    try:
        from thop import profile
        macs, _ = profile(model, inputs=tuple(t.to(device) for t in inputs), verbose=False)
        return f"{macs / 1e9:.1f} GMACs (~{macs * 2 / 1e9:.1f} GFLOPs, thop)"
    except Exception:
        pass
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"~{n * 2 / 1e9:.0f} GFLOPs (param-based fallback — install fvcore for exact)"


# =========================================================================== #
#                          LATENCY / THROUGHPUT                                #
# =========================================================================== #

def latency(fn, device, n_warmup=10, n_runs=100):
    """Per-call latency in ms with proper CUDA warm-up and synchronisation."""
    for _ in range(n_warmup):
        with torch.no_grad():
            fn()
    if device == "cuda":
        torch.cuda.synchronize()

    ts = []
    for _ in range(n_runs):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            fn()
        if device == "cuda":
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    ts = np.array(ts)
    return {"mean_ms": round(float(ts.mean()), 2),
            "std_ms":  round(float(ts.std()), 2),
            "p95_ms":  round(float(np.percentile(ts, 95)), 2)}


def throughput(fn, batch, device, n_runs=20):
    for _ in range(3):
        with torch.no_grad():
            fn()
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        with torch.no_grad():
            fn()
    if device == "cuda":
        torch.cuda.synchronize()
    return round(batch * n_runs / (time.perf_counter() - t0), 1)


def peak_mem_gb(fn, device):
    """Peak GPU memory (GB) used during one call of fn."""
    if device != "cuda":
        return None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return round(torch.cuda.max_memory_allocated() / 1e9, 2)


# =========================================================================== #
#                                 MAIN                                         #
# =========================================================================== #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-mem", action="store_true",
                    help="also measure peak memory of ONE forward+backward train step")
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8])
    args = ap.parse_args()

    seed_everything(config.SEED)
    config.setup_dirs()
    device = config.DEVICE
    S = getattr(config, "IMAGE_SIZE", 448)

    print("=" * 68)
    print("  DermaFusion-AI — Efficiency / Complexity Specifications (#194)")
    print("=" * 68)
    print(f"  Device        : {device}")
    if device == "cuda":
        print(f"  GPU           : {torch.cuda.get_device_name(0)}")
        print(f"  Total VRAM    : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  Input size    : {S}x{S}")

    # ── Build the two models (arch only; weights don't change specs) ───────── #
    unet = (SwinTransformerUNet(pretrained=False) if config.SEG_MODEL == "swin_unet"
            else __import__("models.unet", fromlist=["LightweightUNet"]).LightweightUNet(3, 1)).to(device)
    model = DualBranchFusionClassifier(
        eva02_name=config.EVA02_BACKBONE,      eva02_pretrained=False,
        convnext_name=config.CONVNEXT_BACKBONE, convnext_pretrained=False,
        fusion_dim=config.FUSION_EMBED_DIM,    num_heads=config.FUSION_NUM_HEADS,
        num_classes=config.NUM_CLASSES,        dropout=config.FUSION_DROPOUT,
        use_spatial_fusion=getattr(config, "USE_SPATIAL_FUSION", True),
        fusion_grid=getattr(config, "FUSION_GRID", 14),
    ).to(device)

    loaded = []
    up = os.path.join(config.WEIGHTS_DIR, "best_unet.pth")
    if os.path.exists(up):
        try:
            unet.load_state_dict(torch.load(up, map_location=device, weights_only=True))
            loaded.append("best_unet.pth")
        except Exception:
            pass
    for fname in ("best_dual_branch_fusion_nosala.pth", "best_dual_branch_fusion.pth"):
        cp = os.path.join(config.WEIGHTS_DIR, fname)
        if os.path.exists(cp):
            try:
                model.load_state_dict(torch.load(cp, map_location=device, weights_only=True))
                loaded.append(fname)
                break   # prefer the nosala (deployed) checkpoint; stop after first hit
            except Exception:
                pass
    print(f"  Weights loaded: {loaded if loaded else 'NONE (random init — specs are architecture-identical)'}")
    unet.eval(); model.eval()

    # ── Inputs matching the real pipeline (one image → UNet → mask → branches) #
    img = torch.randn(1, 3, S, S, device=device)

    def pipeline():
        """Full deployed forward: segment → apply mask → classify."""
        mask_logits = unet(img)
        img_seg = apply_mask(img, mask_logits)
        mask_prob = torch.sigmoid(mask_logits.float())
        return model(img, img_seg, mask_prob)

    def pipeline_batch(bs):
        x = img.repeat(bs, 1, 1, 1)
        def _f():
            ml = unet(x)
            xs = apply_mask(x, ml)
            mp = torch.sigmoid(ml.float())
            return model(x, xs, mp)
        return _f

    # ── 1. Parameters ──────────────────────────────────────────────────────── #
    pc = count_parameters(model, unet)
    print("\n[1] PARAMETERS")
    print(f"  {'Module':<34}{'Params (M)':>12}")
    print("  " + "-" * 46)
    for name, val in sorted(pc["breakdown_M"].items(), key=lambda kv: -kv[1]):
        print(f"  {name:<34}{val:>12.2f}")
    print("  " + "-" * 46)
    print(f"  {'Classifier total':<34}{pc['classifier_total_M']:>12.2f}")
    print(f"  {'Swin-UNet segmentation':<34}{pc['unet_M']:>12.2f}")
    print(f"  {'GRAND TOTAL (end-to-end)':<34}{pc['grand_total_M']:>12.2f}")

    # ── 2. FLOPs ───────────────────────────────────────────────────────────── #
    print("\n[2] FLOPs")
    dummy_seg = torch.randn(1, 3, S, S, device=device)
    dummy_mp  = torch.rand(1, 1, S, S, device=device)
    flops_clf  = estimate_flops(model, (img, dummy_seg, dummy_mp), device)
    flops_unet = estimate_flops(unet, (img,), device)
    print(f"  Classifier (EVA-02 + ConvNeXt + fusion): {flops_clf}")
    print(f"  Swin-UNet segmentation:                  {flops_unet}")

    # ── 3. Latency (end-to-end, single image) ─────────────────────────────── #
    print("\n[3] LATENCY (single image, end-to-end)")
    unet.to(device); model.to(device)
    lat = latency(pipeline, device, n_warmup=10, n_runs=100)
    print(f"  End-to-end (UNet + classifier): {lat['mean_ms']:.1f} ± {lat['std_ms']:.1f} ms  (p95 {lat['p95_ms']:.1f} ms)")

    # ── 4. Throughput ──────────────────────────────────────────────────────── #
    print("\n[4] THROUGHPUT")
    tps = {}
    for bs in args.batch_sizes:
        try:
            tp = throughput(pipeline_batch(bs), bs, device, n_runs=20)
            tps[bs] = tp
            print(f"  batch {bs:>2}: {tp:>8.1f} images/sec")
        except RuntimeError as e:
            print(f"  batch {bs:>2}: OOM ({str(e)[:60]})")
            break

    # ── 5. Peak inference memory ───────────────────────────────────────────── #
    print("\n[5] PEAK GPU MEMORY")
    infer_mem = peak_mem_gb(lambda: pipeline(), device)
    print(f"  Inference (batch 1): {infer_mem} GB" if infer_mem is not None else "  Inference: n/a (CPU)")

    train_mem = None
    if args.train_mem and device == "cuda":
        # one forward+backward with a dummy CE loss (mirrors training memory)
        model.train(); unet.eval()
        for p in model.parameters():
            p.requires_grad_(True)
        def _train_step():
            ml = unet(img)
            xs = apply_mask(img, ml)
            mp = torch.sigmoid(ml.float())
            logits, _ = model(img, xs, mp)
            loss = torch.nn.functional.cross_entropy(
                logits, torch.zeros(1, dtype=torch.long, device=device))
            loss.backward()
            model.zero_grad(set_to_none=True)
        train_mem = peak_mem_gb(_train_step, device)
        model.eval()
        print(f"  Train step (batch 1, fwd+bwd): {train_mem} GB")

    # ── 6. Paper-ready table + CSV ─────────────────────────────────────────── #
    print("\n" + "=" * 68)
    print("  PAPER-READY SPECIFICATIONS TABLE  (copy into the manuscript)")
    print("=" * 68)
    gpu = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    rows = [
        ("Backbones",                "EVA-02 Large ViT + ConvNeXt V2 Base"),
        ("Segmentation",             "Swin-Transformer U-Net"),
        ("Input resolution",         f"{S}x{S}"),
        ("Parameters (classifier)",  f"{pc['classifier_total_M']:.1f} M"),
        ("Parameters (segmentation)",f"{pc['unet_M']:.1f} M"),
        ("Parameters (end-to-end)",  f"{pc['grand_total_M']:.1f} M"),
        ("FLOPs (classifier)",       flops_clf.split(' (')[0]),
        ("FLOPs (segmentation)",     flops_unet.split(' (')[0]),
        ("Inference latency",        f"{lat['mean_ms']:.1f} ms/image ({gpu})"),
        ("Throughput",               f"{tps.get(1, float('nan'))} img/s (bs=1), {tps.get(max(tps), float('nan')) if tps else 'n/a'} img/s (bs={max(tps) if tps else '-'})"),
        ("Peak memory (inference)",  f"{infer_mem} GB" if infer_mem is not None else "n/a"),
        ("Peak memory (train step)", f"{train_mem} GB" if train_mem is not None else "run with --train-mem"),
        ("Training GPU",             gpu),
        ("Training time",            "FILL FROM LOGS (total wall-clock across epochs)"),
    ]
    for k, v in rows:
        print(f"  {k:<28}: {v}")

    out = os.path.join(config.OUTPUT_DIR, "efficiency_specs.csv")
    pd.DataFrame(rows, columns=["Specification", "Value"]).to_csv(out, index=False)
    print(f"\n  saved -> {out}")


if __name__ == "__main__":
    main()
