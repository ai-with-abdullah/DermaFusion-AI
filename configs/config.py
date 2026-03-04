import os
import sys
import torch


class Config:
    # =========================================================================
    # Project paths
    # =========================================================================
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR     = os.environ.get('DATA_DIR', os.path.join(PROJECT_ROOT, 'data'))
    OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "outputs")
    WEIGHTS_DIR  = os.path.join(OUTPUT_DIR,   "weights")
    PLOTS_DIR    = os.path.join(OUTPUT_DIR,   "plots")
    MASKS_DIR    = os.path.join(OUTPUT_DIR,   "segmentation_masks")
    ATTENTION_DIR= os.path.join(OUTPUT_DIR,   "attention_maps")

    # Legacy single-dataset paths (HAM10000) — kept for backward compatibility
    IMAGE_DIR = os.path.join(DATA_DIR, "images")
    CSV_PATH  = os.path.join(DATA_DIR, "ISIC_2024_metadata.csv")

    # =========================================================================
    # Multi-Dataset paths
    # (set USE_MULTI_DATASET=True to enable all available datasets)
    # =========================================================================
    USE_MULTI_DATASET = True   # ← Master switch

    # Paths — edit these if your datasets live elsewhere
    ISIC_2019_DIR = os.path.join(DATA_DIR, "isic_2019")   # expects sub-folders per spec
    ISIC_2020_DIR = os.path.join(DATA_DIR, "isic_2020")
    ISIC_2024_DIR = os.path.join(DATA_DIR, "isic_2024")
    PH2_DIR       = os.path.join(DATA_DIR, "ph2")

    # =========================================================================
    # Data & Classes
    # =========================================================================
    IMAGE_SIZE  = 448    # EVA-02 Large native resolution (448×448) — SOTA standard
    NUM_CLASSES = 7
    CLASSES     = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    MEAN        = [0.485, 0.456, 0.406]
    STD         = [0.229, 0.224, 0.225]

    # =========================================================================
    # Training — General
    # =========================================================================
    SEED         = 42
    DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
    # FIXED (Upgrade #3): macOS uses 'fork' multiprocessing which deadlocks
    # with OpenCV and Albumentations inside DataLoader workers.
    # Default to 0 workers on macOS, 4 on Linux/Colab.
    NUM_WORKERS  = 0 if sys.platform == 'darwin' else 4
    EPOCHS       = 10   # Kaggle smoke-test: 10 seg + 10 classification (change to 50 for full run)
    PATIENCE     = 15
    BATCH_SIZE   = 2           # EVA-02 Large requires smaller batch to fit VRAM
    GRADIENT_ACCUMULATION_STEPS = 32  # effective batch = 2 × 32 = 64

    # =========================================================================
    # LR Warmup + Layer Decay (2026 SOTA training recipe)
    # =========================================================================
    WARMUP_EPOCHS = 5      # Linear LR warmup before cosine decay kicks in
    LAYER_DECAY   = 0.75   # Each transformer layer gets LR × LAYER_DECAY^depth

    # =========================================================================
    # Regularisation (2026 Standard)
    # =========================================================================
    USE_EMA       = True
    EMA_DECAY     = 0.9998
    MIXUP_ALPHA   = 0.8
    USE_CUTMIX    = True       # ← Now actually implemented in trainer
    CUTMIX_ALPHA  = 1.0
    LABEL_SMOOTHING = 0.10     # Applied in LabelSmoothingFocalLoss
    USE_TTA       = True       # Test-Time Augmentation at eval time (N=5 views)

    # =========================================================================
    # Segmentation Model
    # =========================================================================
    SEG_MODEL         = "swin_unet"    # 'swin_unet' | 'lightweight_unet' (legacy)
    SEG_LR            = 1e-4
    SEG_WEIGHT_DECAY  = 1e-4

    # =========================================================================
    # Dual-Branch Fusion Classifier: EVA-02 + ConvNeXt V2
    # =========================================================================

    # Branch A: EVA-02 Large (Global / Contextual) — 2026 SOTA
    # EVA-02 Large @ 448px: 304M params, ImageNet top-1 89.6%, best open-source ViT for fine-grained classification
    EVA02_BACKBONE   = "eva02_large_patch14_448.mim_in22k_ft_in22k_in1k"
    EVA02_PRETRAINED = True
    EVA02_EMBED_DIM  = 1024   # Large model output dim (was 384 for Small)
    EVA02_LR         = 1e-5   # Lower LR for large pretrained model (was 2e-5 for Small)
    # Branch B: ConvNeXt V2 (Local / Texture)
    CONVNEXT_BACKBONE  = "convnextv2_base.fcmae_ft_in22k_in1k_384"
    CONVNEXT_PRETRAINED = True
    CONVNEXT_LR         = 1e-5
    CONVNEXT_INIT_ARGS  = dict(multi_scale=False) # Disabled — multi-scale returns a tuple which breaks fusion

    # Fusion
    FUSION_EMBED_DIM = 512
    FUSION_NUM_HEADS = 8
    FUSION_DROPOUT   = 0.2

    # Head
    HEAD_LR      = 1e-4
    WEIGHT_DECAY = 0.05
    DROPOUT      = 0.3

    # =========================================================================
    # Legacy — kept for backward compatibility (not used in current pipeline)
    # =========================================================================
    BACKBONE    = "convnextv2_large.fcmae_ft_in22k_in1k_384"
    BACKBONE_LR = 1e-5
    EMBED_DIM   = 1024
    NUM_HEADS   = 8
    # EVA-02 Large is now the active backbone (upgraded from Small).
    # Small reference: eva02_small_patch14_336.mim_in22k_ft_in1k (384px, embed=384)

    # =========================================================================
    # Evaluation
    # =========================================================================
    TTA_N_VIEWS  = 5     # Number of TTA augmentation views to average

    @classmethod
    def setup_dirs(cls):
        for d in [
            cls.DATA_DIR, cls.OUTPUT_DIR, cls.WEIGHTS_DIR,
            cls.PLOTS_DIR, cls.MASKS_DIR, cls.ATTENTION_DIR,
        ]:
            os.makedirs(d, exist_ok=True)


config = Config()
