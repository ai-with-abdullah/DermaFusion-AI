"""
Inference Time & FLOPs Benchmark
==================================
Measures DermaFusion-AI inference performance for the paper's
complexity/efficiency section.

Benchmarks:
  • Single-image latency (GPU + CPU, ms)
  • Batch throughput (images/second)
  • FLOPs (GFLOPs, using fvcore or thop)
  • Parameter count per component

Usage:
    python -m evaluation.run_inference_benchmark

Output:
    • Printed table (copy-paste into paper)
    • outputs/inference_benchmark.csv
"""

import os
import sys
import time

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

import numpy as np
import pandas as pd
import torch

from configs.config import config
from utils.seed import seed_everything
from utils.logger import setup_logger
from models.transformer_unet import SwinTransformerUNet
from models.dual_branch_fusion import DualBranchFusionClassifier


# =========================================================================== #
#                      LATENCY BENCHMARKING                                    #
# =========================================================================== #

def benchmark_latency(
    model_fn,
    input_tensors: list,
    device: str,
    n_warmup: int = 10,
    n_runs: int = 100,
) -> dict:
    """
    Measure per-image latency with proper GPU warm-up and synchronisation.

    Args:
        model_fn:       callable(*input_tensors) -> any
        input_tensors:  list of tensors fed to model_fn
        device:         'cpu' or 'cuda'
        n_warmup:       warmup iterations (not timed — JIT / cuDNN autotune)
        n_runs:         timed iterations

    Returns:
        dict: mean_ms, std_ms, min_ms, max_ms, median_ms, p95_ms
    """
    inputs = [t.to(device) for t in input_tensors]

    # Warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            model_fn(*inputs)
        if device == "cuda":
            torch.cuda.synchronize()

    # Timed runs
    timings = []
    for _ in range(n_runs):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model_fn(*inputs)
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000)  # ms

    timings = np.array(timings)
    return {
        "mean_ms":   round(float(timings.mean()),   2),
        "std_ms":    round(float(timings.std()),    2),
        "median_ms": round(float(np.median(timings)), 2),
        "min_ms":    round(float(timings.min()),    2),
        "max_ms":    round(float(timings.max()),    2),
        "p95_ms":    round(float(np.percentile(timings, 95)), 2),
    }


def benchmark_throughput(
    model_fn,
    input_tensors: list,
    device: str,
    batch_size: int = 8,
    n_runs: int = 20,
) -> float:
    """
    Batch throughput in images per second.
    """
    inputs = [t.repeat(batch_size, *([1] * (t.dim() - 1))).to(device)
              for t in input_tensors]

    # Warmup
    for _ in range(3):
        with torch.no_grad():
            model_fn(*inputs)
        if device == "cuda":
            torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_runs):
        with torch.no_grad():
            model_fn(*inputs)
    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    total_images = batch_size * n_runs
    elapsed_s    = t1 - t0
    return round(total_images / elapsed_s, 1)


# =========================================================================== #
#                      FLOPs ESTIMATION                                        #
# =========================================================================== #

def estimate_flops(model, dummy_inputs: tuple, device: str) -> str:
    """
    Try fvcore, then thop, then fall back to parameter-based estimate.

    Returns:
        Human-readable string, e.g. "245.3 GFLOPs"
    """
    model_eval = model.eval()

    # Attempt fvcore
    try:
        from fvcore.nn import FlopCountAnalysis
        inputs = tuple(t.to(device) for t in dummy_inputs)
        flops  = FlopCountAnalysis(model_eval, inputs)
        flops.unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)
        gflops = flops.total() / 1e9
        return f"{gflops:.1f} GFLOPs (fvcore)"
    except Exception:
        pass

    # Attempt thop
    try:
        from thop import profile
        inputs = tuple(t.to(device) for t in dummy_inputs)
        macs, _ = profile(model_eval, inputs=inputs, verbose=False)
        return f"{macs / 1e9:.1f} GMACs (~{macs * 2 / 1e9:.1f} GFLOPs, thop)"
    except Exception:
        pass

    # Fallback: estimate from parameter count
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"~{n_params * 2 / 1e9:.0f} GFLOPs (param-based estimate, install fvcore for exact)"


# =========================================================================== #
#                         PARAMETER COUNT TABLE                                #
# =========================================================================== #

def count_parameters(model) -> dict:
    """Per-module parameter counts in millions."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    result = {"total_M": round(total / 1e6, 2), "trainable_M": round(trainable / 1e6, 2)}

    # Component-level breakdown
    components = {
        "branch_eva":   "Branch A (EVA-02 Large)",
        "branch_conv":  "Branch B (ConvNeXt V2)",
        "cross_attn":   "Cross-Attention Fusion",
        "gate_a":       "Gated Residual (A)",
        "gate_b":       "Gated Residual (B)",
        "classifier":   "Classifier Head",
    }
    for attr, label in components.items():
        if hasattr(model, attr):
            n = sum(p.numel() for p in getattr(model, attr).parameters())
            result[label] = round(n / 1e6, 2)

    return result


# =========================================================================== #
#                               MAIN                                           #
# =========================================================================== #

def main():
    seed_everything(config.SEED)
    config.setup_dirs()
    logger = setup_logger(
        "inference_benchmark",
        os.path.join(config.OUTPUT_DIR, "inference_benchmark.log"),
    )
    logger.info("=" * 65)
    logger.info("  DermaFusion-AI — Inference Time & Parameter Benchmark")
    logger.info("=" * 65)

    device = config.DEVICE
    logger.info(f"  Device: {device}")
    if device == "cuda":
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Load models ──────────────────────────────────────────────────────── #
    unet = SwinTransformerUNet(pretrained=False).to(device)
    unet_path = os.path.join(config.WEIGHTS_DIR, "best_unet.pth")
    if os.path.exists(unet_path):
        unet.load_state_dict(torch.load(unet_path, map_location=device, weights_only=True))
    unet.eval()

    model = DualBranchFusionClassifier(
        eva02_name=config.EVA02_BACKBONE,     eva02_pretrained=False,
        convnext_name=config.CONVNEXT_BACKBONE, convnext_pretrained=False,
        fusion_dim=config.FUSION_EMBED_DIM,  num_heads=config.FUSION_NUM_HEADS,
        num_classes=config.NUM_CLASSES,      dropout=config.FUSION_DROPOUT,
    ).to(device)

    model_path = os.path.join(config.WEIGHTS_DIR, "best_dual_branch_fusion.pth")
    if os.path.exists(model_path):
        try:
            model.load_state_dict(
                torch.load(model_path, map_location=device, weights_only=True),
                strict=True,
            )
            logger.info(f"Loaded classifier weights from {model_path}")
        except RuntimeError as e:
            logger.warning(
                f"Weight loading failed (strict=True): {str(e)[:120]}...\n"
                "  → Falling back to RANDOM weights.\n"
                "  → ⚠ This is fine for latency/FLOPs benchmarks — timing depends on\n"
                "  →   architecture, not weight values. Results are still paper-valid.\n"
                "  → To fix: ensure best_dual_branch_fusion.pth is EVA-02 LARGE (dim=1024),\n"
                "  →   not the old EVA-02 Small (dim=384) checkpoint."
            )
    model.eval()


    # ── Dummy inputs ─────────────────────────────────────────────────────── #
    img_eva  = torch.randn(1, 3, 448, 448)   # EVA-02 input
    img_conv = torch.randn(1, 3, 384, 384)   # ConvNeXt input (segmented)
    img_unet = torch.randn(1, 3, 448, 448)   # UNet input

    # ── Parameter counts ─────────────────────────────────────────────────── #
    logger.info("\n[1] Parameter Counts")
    param_info  = count_parameters(model)
    unet_params = sum(p.numel() for p in unet.parameters()) / 1e6

    rows_param = []
    logger.info(f"  {'Component':<35} {'Params (M)':>12}")
    logger.info("  " + "-" * 48)
    for name, val in param_info.items():
        if name.endswith("_M"):
            continue
        logger.info(f"  {name:<35} {val:>12.2f}M")
        rows_param.append({"Component": name, "Params_M": val})

    logger.info(f"  {'Segmentation (Swin-UNet)':<35} {unet_params:>12.2f}M")
    logger.info(f"  {'TOTAL (classifier)':<35} {param_info['total_M']:>12.2f}M")
    logger.info(f"  {'TOTAL (including UNet)':<35} {param_info['total_M'] + unet_params:>12.2f}M")

    # ── FLOPs estimation ─────────────────────────────────────────────────── #
    logger.info("\n[2] FLOPs Estimation")

    def full_model_forward(img_main, img_seg):
        return model(img_main, img_seg)

    flops_classifier = estimate_flops(model, (img_eva, img_conv), device)
    flops_unet       = estimate_flops(unet,  (img_unet,),         device)
    logger.info(f"  Classifier (EVA-02 + ConvNeXt + Fusion): {flops_classifier}")
    logger.info(f"  Segmentation UNet:                       {flops_unet}")

    # ── Latency: GPU single image ─────────────────────────────────────────── #
    logger.info("\n[3] Latency Benchmarks")
    rows_latency = []

    logger.info("  Benchmarking UNet segmentation (GPU, 1 image)...")
    unet_lat = benchmark_latency(
        lambda x: unet(x), [img_unet], device, n_warmup=5, n_runs=50
    )
    logger.info(f"    UNet: {unet_lat['mean_ms']:.1f} ± {unet_lat['std_ms']:.1f} ms")
    rows_latency.append({"Model": "Swin-UNet Segmentation", "Device": device,
                         "Batch": 1, **unet_lat})

    logger.info("  Benchmarking DermaFusion classifier (GPU, 1 image)...")

    def clf_forward(a, b): return model(a, b)
    clf_lat = benchmark_latency(
        clf_forward, [img_eva, img_conv], device, n_warmup=5, n_runs=50
    )
    logger.info(f"    Classifier: {clf_lat['mean_ms']:.1f} ± {clf_lat['std_ms']:.1f} ms")
    rows_latency.append({"Model": "DermaFusion-AI Classifier", "Device": device,
                         "Batch": 1, **clf_lat})

    total_latency = unet_lat["mean_ms"] + clf_lat["mean_ms"]
    logger.info(f"    Total E2E (UNet + Classifier): {total_latency:.1f} ms/image (GPU)")

    # ── Latency: CPU (single image — important for low-resource deployment) ── #
    if device == "cuda":
        logger.info("  Benchmarking on CPU (1 image, 30 runs)...")
        model_cpu = model.cpu()
        unet_cpu  = unet.cpu()

        unet_lat_cpu = benchmark_latency(
            lambda x: unet_cpu(x), [img_unet.cpu()], "cpu", n_warmup=3, n_runs=30
        )
        clf_lat_cpu = benchmark_latency(
            lambda a, b: model_cpu(a, b), [img_eva.cpu(), img_conv.cpu()],
            "cpu", n_warmup=3, n_runs=30
        )
        rows_latency.append({"Model": "DermaFusion-AI (CPU)", "Device": "cpu",
                             "Batch": 1, **clf_lat_cpu})
        logger.info(f"    Classifier (CPU): {clf_lat_cpu['mean_ms']:.0f} ms/image")
        logger.info(f"    E2E (CPU): {unet_lat_cpu['mean_ms'] + clf_lat_cpu['mean_ms']:.0f} ms/image")

        model.to(device)
        unet.to(device)

    # ── Throughput ────────────────────────────────────────────────────────── #
    logger.info("\n[4] Batch Throughput")
    for bs in [1, 4, 8]:
        try:
            tp = benchmark_throughput(clf_forward, [img_eva, img_conv], device,
                                      batch_size=bs, n_runs=20)
            logger.info(f"    Batch {bs:2d}: {tp:.1f} images/sec")
            rows_latency.append({"Model": f"DermaFusion (batch={bs})", "Device": device,
                                  "Batch": bs, "throughput_imgs_per_sec": tp})
        except RuntimeError as e:
            logger.warning(f"  Batch {bs} OOM: {e}")
            break

    # ── Paper-ready summary ───────────────────────────────────────────────── #
    print("\n" + "=" * 65)
    print("  PAPER-READY INFERENCE TABLE")
    print("=" * 65)
    print(f"  {'Configuration':<40} {'Latency':>12}")
    print("  " + "-" * 53)
    print(f"  {'Swin-UNet segmentation (GPU)':<40} {unet_lat['mean_ms']:>8.1f} ms")
    print(f"  {'DermaFusion-AI classifier (GPU)':<40} {clf_lat['mean_ms']:>8.1f} ms")
    print(f"  {'Total end-to-end (GPU)':<40} {total_latency:>8.1f} ms")
    if device == "cuda":
        cpu_total = unet_lat_cpu['mean_ms'] + clf_lat_cpu['mean_ms']
        print(f"  {'Total end-to-end (CPU)':<40} {cpu_total:>8.0f} ms")
    print(f"\n  Parameters: {param_info['total_M']} M (classifier) + {unet_params:.1f} M (UNet)")
    print(f"  FLOPs: {flops_classifier}")
    print(f"  FLOPs (UNet): {flops_unet}")

    # ── Save CSV ──────────────────────────────────────────────────────────── #
    csv_path = os.path.join(config.OUTPUT_DIR, "inference_benchmark.csv")
    pd.DataFrame(rows_latency).to_csv(csv_path, index=False)
    logger.info(f"\n  Results saved to {csv_path}")
    print(f"\n✅ Done. Results saved to {csv_path}")


if __name__ == "__main__":
    main()
