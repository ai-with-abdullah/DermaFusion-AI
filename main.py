"""
Full Pipeline — AI Medical Disease Diagnosis System (2026 SOTA Upgrade)
=======================================================================
STAGE 1: Train SwinTransformerUNet lesion segmenter
STAGE 2: Train Dual-Branch Fusion Classifier (EVA-02 + ConvNeXt V2)
STAGE 3: Evaluate with TTA + GradCAM++ XAI
"""

import os
from configs.config import config


def print_banner(text: str, width: int = 70) -> None:
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


def main():
    config.setup_dirs()

    print_banner("AI Medical Disease Diagnosis System — 2026 SOTA Upgrade")
    print(f"  Device:         {config.DEVICE}")
    print(f"  Multi-dataset:  {config.USE_MULTI_DATASET}")
    print(f"  Seg model:      {config.SEG_MODEL}")
    print(f"  Epochs:         {config.EPOCHS}")
    print(f"  Image size:     {config.IMAGE_SIZE}×{config.IMAGE_SIZE}")
    print(f"  Outputs →       {config.OUTPUT_DIR}")

    # ── Stage 1: Segmentation ────────────────────────────────────────────── #
    print_banner("STAGE 1: Swin-Transformer U-Net Lesion Segmentation")
    import train_segmentation
    train_segmentation.main()

    # ── Stage 2: Classification ──────────────────────────────────────────── #
    print_banner("STAGE 2: Dual-Branch Fusion Classifier (EVA-02 + ConvNeXt V2)")
    import train_classifier
    train_classifier.main()

    # ── Stage 3: Evaluation + XAI ───────────────────────────────────────── #
    print_banner("STAGE 3: Evaluation (TTA) + GradCAM++ Explainability")
    import evaluate
    evaluate.main()

    print_banner("PIPELINE COMPLETE")
    print(f"  ✓ Weights  → {config.WEIGHTS_DIR}")
    print(f"  ✓ Plots    → {config.PLOTS_DIR}")
    print(f"  ✓ XAI maps → {os.path.join(config.OUTPUT_DIR, 'explainability_heatmaps')}")
    print(f"  ✓ Logs     → {config.OUTPUT_DIR}")
    print()


if __name__ == "__main__":
    main()
