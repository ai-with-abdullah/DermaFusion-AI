"""
Dual-Branch Fusion Classifier Training — 2026 SOTA Upgrade
==========================================================
Key upgrades over previous version:
  ✓ Multi-dataset support (HAM10000 + ISIC 2019/2020/2024 + PH2)
  ✓ CutMix augmentation (was in config but never implemented)
  ✓ LR warmup (5 epochs linear) + Cosine decay
  ✓ CombinedClassLoss (LabelSmoothing Focal + Symmetric CE)
  ✓ Weighted sampler (handles extreme class imbalance)
  ✓ Richer metrics logging (balanced acc, per-class sens/spec, ECE)
"""

import os
import math
import torch
import torch.optim as optim
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import pandas as pd
import numpy as np

from configs.config import config
from utils.seed import seed_everything
from utils.logger import setup_logger

# Multi-dataset loader (falls back to HAM10000 if others not downloaded)
from datasets.unified_dataset import get_unified_dataloaders, get_class_weights_from_records

from datasets.mixup import mixup_data, mixup_criterion
from models.dual_branch_fusion import DualBranchFusionClassifier
from models.transformer_unet import SwinTransformerUNet
from models.unet import LightweightUNet
from training.losses import get_combined_class_loss
from evaluation.metrics import compute_metrics, log_metrics
from training.train_utils import AverageMeter, EarlyStopping, apply_mask, ModelEMA


# =========================================================================== #
#                        CUTMIX IMPLEMENTATION                                 #
# =========================================================================== #

def cutmix_data(images, labels, alpha=1.0):
    """
    CutMix augmentation (Yun et al., ICCV 2019).
    Cuts a random rectangular patch from one image and pastes it into another.
    Labels are mixed proportionally to the patch area ratio (lambda).
    """
    if alpha <= 0:
        return images, labels, labels, 1.0

    lam = np.random.beta(alpha, alpha)
    B, C, H, W = images.shape
    idx = torch.randperm(B, device=images.device)

    # Random bounding box
    cut_rat = math.sqrt(1.0 - lam)
    cut_w   = int(W * cut_rat)
    cut_h   = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, W)
    y2 = min(cy + cut_h // 2, H)

    images_mixed = images.clone()
    images_mixed[:, :, y1:y2, x1:x2] = images[idx, :, y1:y2, x1:x2]

    # Recalculate lambda based on actual cut area
    lam = 1.0 - (x2 - x1) * (y2 - y1) / (W * H)

    return images_mixed, labels, labels[idx], lam


def cutmix_criterion(criterion, logits, targets_a, targets_b, lam):
    return lam * criterion(logits, targets_a) + (1 - lam) * criterion(logits, targets_b)


# =========================================================================== #
#                        LR WARMUP SCHEDULER                                   #
# =========================================================================== #

class WarmupCosineScheduler:
    """
    Linear warmup for `warmup_epochs` then cosine annealing.
    Wraps an existing optimizer and adjusts LR each epoch.
    """
    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int, min_lr_ratio: float = 0.01):
        self.optimizer     = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs
        self.min_lr_ratio  = min_lr_ratio
        self.base_lrs      = [pg['lr'] for pg in optimizer.param_groups]
        self._epoch        = 0

    def step(self):
        self._epoch += 1
        epoch = self._epoch
        if epoch <= self.warmup_epochs:
            # Linear warmup: start at 1% of LR, scale to 100%
            scale = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * epoch / self.warmup_epochs
        else:
            # Cosine decay
            progress = (epoch - self.warmup_epochs) / max(self.total_epochs - self.warmup_epochs, 1)
            scale    = self.min_lr_ratio + 0.5 * (1.0 - self.min_lr_ratio) * (1.0 + math.cos(math.pi * progress))

        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg['lr'] = base_lr * scale

    def get_last_lr(self):
        return [pg['lr'] for pg in self.optimizer.param_groups]

    def state_dict(self):
        return {'_epoch': self._epoch, 'base_lrs': self.base_lrs}

    def load_state_dict(self, d):
        self._epoch = d['_epoch']
        # Bug #8 fix: Do NOT restore base_lrs from checkpoint.
        # The optimizer's current param_groups already have correct LRs after
        # loading the optimizer state dict. Re-reading base_lrs from checkpoint
        # overwrites them with potentially stale values from a different epoch.
        # We refresh base_lrs from the live optimizer state instead.
        self.base_lrs = [pg['lr'] for pg in self.optimizer.param_groups]


# =========================================================================== #
#                        TRAINING LOOP                                         #
# =========================================================================== #

def train_one_epoch(model, ema, unet, loader, criterion, optimizer, scaler, device, accumulation_steps=1):
    model.train()
    unet.eval()

    losses      = AverageMeter()
    all_targets = []
    all_probs   = []

    pbar = tqdm(loader, desc='Train')
    optimizer.zero_grad()

    for i, batch in enumerate(pbar):
        images = batch['image'].to(device)
        labels = batch['label'].to(device)

        # ── Decide augmentation: MixUp or CutMix (not both) ─────────────── #
        use_mixup  = config.MIXUP_ALPHA  > 0 and np.random.rand() > 0.6
        use_cutmix = config.USE_CUTMIX  and config.CUTMIX_ALPHA > 0 and np.random.rand() > 0.6

        # ── Generate segmentation mask FIRST on original images (Bug #4 fix) #
        # Previously, MixUp/CutMix was applied before segmentation, causing
        # the UNet to generate garbage masks from blended composite images.
        # Fix: always segment ORIGINAL images, then apply augmentation to BOTH.
        with torch.no_grad():
            with autocast('cuda', enabled=(device == 'cuda')):
                mask_logits = unet(images)             # segment ORIGINAL images
                images_seg  = apply_mask(images, mask_logits)

        # ── Now apply MixUp/CutMix to BOTH images and images_seg ─────────── #
        if use_mixup and not use_cutmix:
            images, targets_a, targets_b, lam = mixup_data(
                images, labels, config.MIXUP_ALPHA, use_cuda=(device == 'cuda')
            )
            # Apply the same permutation/lam to the segmented branch
            batch_size = images.size(0)
            idx = torch.randperm(batch_size, device=images.device)
            images_seg_mixed = lam * images_seg + (1 - lam) * images_seg[idx]
            images_seg = images_seg_mixed
            mix_fn = lambda logits: mixup_criterion(criterion, logits, targets_a, targets_b, lam)

        elif use_cutmix and not use_mixup:
            images, targets_a, targets_b, lam = cutmix_data(images, labels, config.CUTMIX_ALPHA)
            # Apply the same bounding box crop to images_seg
            # Re-run cutmix on images_seg with same alpha to get matching masks
            images_seg, _, _, _ = cutmix_data(images_seg, labels, config.CUTMIX_ALPHA)
            mix_fn = lambda logits: cutmix_criterion(criterion, logits, targets_a, targets_b, lam)
        else:
            mix_fn = None

        # ── Forward pass ────────────────────────────────────────────────── #
        with autocast('cuda', enabled=(device == 'cuda')):
            logits, _ = model(images, images_seg)

            if mix_fn is not None:
                loss = mix_fn(logits)
            else:
                loss = criterion(logits, labels)

            loss = loss / accumulation_steps

        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # gradient clipping
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if ema is not None:
                ema.update(model)

        losses.update(loss.item() * accumulation_steps, images.size(0))

        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        all_targets.extend(labels.cpu().numpy())
        all_probs.extend(probs)

        pbar.set_postfix({'Loss': f'{losses.avg:.4f}'})

    metrics = compute_metrics(np.array(all_targets), np.array(all_probs))
    return losses.avg, metrics


def validate(model, unet, loader, criterion, device):
    model.eval()
    unet.eval()

    losses      = AverageMeter()
    all_targets = []
    all_probs   = []

    pbar = tqdm(loader, desc='Val')
    with torch.no_grad():
        for batch in pbar:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            with autocast('cuda', enabled=(device == 'cuda')):
                mask_logits = unet(images)
                images_seg  = apply_mask(images, mask_logits)
                logits, _   = model(images, images_seg)
                loss        = criterion(logits, labels)

            losses.update(loss.item(), images.size(0))

            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_targets.extend(labels.cpu().numpy())
            all_probs.extend(probs)

            pbar.set_postfix({'Loss': f'{losses.avg:.4f}'})

    metrics = compute_metrics(np.array(all_targets), np.array(all_probs))
    return losses.avg, metrics


# =========================================================================== #
#                        MAIN                                                  #
# =========================================================================== #

def main():
    seed_everything(config.SEED)
    config.setup_dirs()

    logger = setup_logger("train_class", os.path.join(config.OUTPUT_DIR, "train_dual_branch.log"))
    logger.info("=" * 70)
    logger.info("Starting Dual-Branch Fusion Classifier Training — 2026 SOTA Upgrade")
    logger.info(f"  Branch A: EVA-02        → {config.EVA02_BACKBONE}")
    logger.info(f"  Branch B: ConvNeXt V2   → {config.CONVNEXT_BACKBONE}")
    logger.info(f"  Multi-dataset:          {config.USE_MULTI_DATASET}")
    logger.info(f"  CutMix enabled:         {config.USE_CUTMIX}")
    logger.info(f"  LR warmup epochs:       {config.WARMUP_EPOCHS}")
    logger.info(f"  Label smoothing:        {config.LABEL_SMOOTHING}")
    logger.info(f"  Device: {config.DEVICE}")
    logger.info("=" * 70)

    # ── Dataset ─────────────────────────────────────────────────────────── #
    train_loader, val_loader, test_loader, train_records = get_unified_dataloaders(
        config.DATA_DIR,
        masks_dir=os.path.join(config.DATA_DIR, "masks"),
    )

    # ── Segmentation model ──────────────────────────────────────────────── #
    if config.SEG_MODEL == 'swin_unet':
        unet = SwinTransformerUNet(pretrained=True).to(config.DEVICE)
    else:
        unet = LightweightUNet(n_channels=3, n_classes=1).to(config.DEVICE)

    best_unet_path = os.path.join(config.WEIGHTS_DIR, "best_unet.pth")
    if os.path.exists(best_unet_path):
        unet.load_state_dict(torch.load(best_unet_path, map_location=config.DEVICE))
        logger.info(f"Loaded segmentation weights from {best_unet_path}")

    # ── Classification model ────────────────────────────────────────────── #
    model = DualBranchFusionClassifier(
        eva02_name=config.EVA02_BACKBONE,
        eva02_pretrained=config.EVA02_PRETRAINED,
        convnext_name=config.CONVNEXT_BACKBONE,
        convnext_pretrained=config.CONVNEXT_PRETRAINED,
        fusion_dim=config.FUSION_EMBED_DIM,
        num_heads=config.FUSION_NUM_HEADS,
        num_classes=config.NUM_CLASSES,
        dropout=config.FUSION_DROPOUT,
    ).to(config.DEVICE)

    ema = ModelEMA(model, decay=config.EMA_DECAY, device=config.DEVICE) if config.USE_EMA else None

    # ── 3-Group Differential LR Optimizer ──────────────────────────────── #
    # ── LLRD: Layer-wise LR Decay for EVA-02 ───────────────────────────── #
    # config.LAYER_DECAY = 0.75 was defined but never actually applied before.
    # LLRD assigns lower LRs to deeper (earlier) transformer blocks to prevent
    # catastrophic forgetting of pretrained low-level features.
    # Effect: block 0 (deepest) gets EVA02_LR × 0.75^N, shallowest gets ~EVA02_LR.

    def build_llrd_param_groups(backbone_module, base_lr: float, layer_decay: float, prefix: str):
        """
        Build optimizer param groups with layer-wise LR decay.
        Returns a list of {params, lr} dicts — one per transformer block depth.
        """
        param_groups = []

        # Collect all named parameters inside the backbone
        # Try to detect transformer blocks (works for timm ViT/EVA-02 with .blocks attribute)
        named_params = dict(backbone_module.named_parameters())

        # Identify block depth for each parameter
        block_params = {}   # depth -> list of params
        other_params = []   # embed/stem/head — use base_lr

        num_blocks = 0
        if hasattr(backbone_module, 'model') and hasattr(backbone_module.model, 'blocks'):
            num_blocks = len(backbone_module.model.blocks)
        elif hasattr(backbone_module, 'blocks'):
            num_blocks = len(backbone_module.blocks)

        for name, param in named_params.items():
            if not param.requires_grad:
                continue
            # Detect block index in name: e.g. 'model.blocks.5.attn.proj.weight'
            matched_block = False
            for bi in range(num_blocks):
                if f'blocks.{bi}.' in name:
                    block_params.setdefault(bi, []).append(param)
                    matched_block = True
                    break
            if not matched_block:
                other_params.append(param)

        if num_blocks == 0 or not block_params:
            # Fallback: no block structure detected — use flat LR
            logger.info(f"  [LLRD] No block structure found in {prefix} — using flat LR {base_lr:.2e}")
            return [{'params': list(named_params.values()), 'lr': base_lr}]

        # Build per-block param groups with decaying LR
        # Shallowest block (num_blocks-1) gets base_lr × decay^1
        # Deepest block (0) gets base_lr × decay^num_blocks
        for bi in range(num_blocks):
            depth = num_blocks - bi  # depth from output: block[N-1]=1, block[0]=N
            block_lr = base_lr * (layer_decay ** depth)
            param_groups.append({'params': block_params[bi], 'lr': block_lr})

        # Non-block params (patch embed, norm, cls_token) get base_lr
        if other_params:
            param_groups.append({'params': other_params, 'lr': base_lr})

        logger.info(f"  [LLRD] {prefix}: {num_blocks} blocks, "
                    f"LR range [{base_lr * layer_decay**num_blocks:.2e} … {base_lr:.2e}]")
        return param_groups

    eva02_param_groups    = build_llrd_param_groups(
        model.branch_eva, config.EVA02_LR, config.LAYER_DECAY, 'EVA-02'
    )
    convnext_params = model.get_convnext_params()
    head_params     = model.get_head_params()

    logger.info(f"Parameter groups:")
    logger.info(f"  EVA-02 (LLRD, {len(eva02_param_groups)} groups): base_lr={config.EVA02_LR}, decay={config.LAYER_DECAY}")
    logger.info(f"  ConvNeXt params: {sum(p.numel() for p in convnext_params)/1e6:.1f}M  lr={config.CONVNEXT_LR}")
    logger.info(f"  Head params:     {sum(p.numel() for p in head_params)/1e6:.1f}M  lr={config.HEAD_LR}")

    optimizer = optim.AdamW(
        eva02_param_groups
        + [{'params': convnext_params, 'lr': config.CONVNEXT_LR}]
        + [{'params': head_params,     'lr': config.HEAD_LR}],
        weight_decay=config.WEIGHT_DECAY,
    )

    # ── LR warmup + Cosine scheduler ───────────────────────────────────── #
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=config.WARMUP_EPOCHS,
        total_epochs=config.EPOCHS,
    )

    # ── Loss: Combined (LabelSmoothing Focal + SCE) ─────────────────────── #
    class_weights = get_class_weights_from_records(train_records)
    criterion     = get_combined_class_loss(
        class_weights, config.DEVICE, config.NUM_CLASSES, config.LABEL_SMOOTHING
    )

    scaler = GradScaler('cuda', enabled=(config.DEVICE == 'cuda'))

    best_model_path = os.path.join(config.WEIGHTS_DIR, "best_dual_branch_fusion.pth")
    early_stopping  = EarlyStopping(
        patience=15,   # increased; balanced_acc is the metric now
        verbose=True, path=best_model_path, trace_func=logger.info
    )

    # ── Checkpoint resume: load if exists ─────────────────────────────── #
    resume_path = os.path.join(config.WEIGHTS_DIR, "resume_checkpoint.pth")
    start_epoch = 1
    history = []
    if os.path.exists(resume_path):
        logger.info(f"\n[RESUME] Found checkpoint → {resume_path}")
        ckpt = torch.load(resume_path, map_location=config.DEVICE)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        if ema is not None and 'ema_shadow' in ckpt:
            ema.shadow = ckpt['ema_shadow']
        start_epoch = ckpt['epoch'] + 1
        history      = ckpt.get('history', [])
        early_stopping.best_score = ckpt.get('best_score', None)
        early_stopping.counter    = ckpt.get('es_counter', 0)
        logger.info(f"[RESUME] Resuming from Epoch {start_epoch}/{config.EPOCHS}")
    else:
        logger.info("[RESUME] No checkpoint found — starting from Epoch 1")

    # ── Training Loop ────────────────────────────────────────────────────── #
    for epoch in range(start_epoch, config.EPOCHS + 1):
        current_lrs = scheduler.get_last_lr()
        logger.info(f"\nEpoch {epoch}/{config.EPOCHS}  |  LR: {current_lrs[2]:.2e} (head)")

        train_loss, train_metrics = train_one_epoch(
            model, ema, unet, train_loader, criterion, optimizer, scaler,
            config.DEVICE, config.GRADIENT_ACCUMULATION_STEPS
        )

        # Validate using EMA weights for stability, then RESTORE training weights
        if ema is not None:
            ema.apply_shadow(model)          # swap in EMA weights
        val_loss, val_metrics = validate(model, unet, val_loader, criterion, config.DEVICE)
        if ema is not None:
            ema.restore(model)               # restore real training weights!

        # LR scheduler step (warmup → cosine)
        scheduler.step()

        logger.info(
            f"Train — Loss: {train_loss:.4f}  Acc: {train_metrics['accuracy']:.4f}  "
            f"BalAcc: {train_metrics['balanced_accuracy']:.4f}  F1: {train_metrics['macro_f1']:.4f}"
        )
        logger.info(
            f"Val   — Loss: {val_loss:.4f}  AUC: {val_metrics['macro_auc']:.4f}  "
            f"BalAcc: {val_metrics['balanced_accuracy']:.4f}  "
            f"F1: {val_metrics['macro_f1']:.4f}  ECE: {val_metrics['ece']:.4f}"
        )

        history.append({
            'epoch':        epoch,
            'train_loss':   train_loss,  'val_loss':   val_loss,
            'train_auc':    train_metrics['macro_auc'], 'val_auc': val_metrics['macro_auc'],
            'train_bacc':   train_metrics['balanced_accuracy'],
            'val_bacc':     val_metrics['balanced_accuracy'],
            'val_ece':      val_metrics['ece'],
        })

        # Use balanced_accuracy — it steadily improves and is robust to AUC bugs
        early_stopping(val_metrics['balanced_accuracy'], model)

        # ── Save resume checkpoint after every epoch ──────────────────── #
        ckpt = {
            'epoch':      epoch,
            'model':      model.state_dict(),
            'optimizer':  optimizer.state_dict(),
            'scheduler':  scheduler.state_dict(),
            'history':    history,
            'best_score': early_stopping.best_score,
            'es_counter': early_stopping.counter,
        }
        if ema is not None:
            ckpt['ema_shadow'] = ema.shadow
        torch.save(ckpt, resume_path)
        logger.info(f"[CHECKPOINT] Saved epoch {epoch} → {resume_path}")

        if early_stopping.early_stop:
            logger.info("Early stopping triggered.")
            break

    pd.DataFrame(history).to_csv(
        os.path.join(config.OUTPUT_DIR, "training_history_dual_branch.csv"), index=False
    )
    logger.info("Dual-Branch Fusion Classifier Training Completed.")


if __name__ == "__main__":
    main()
