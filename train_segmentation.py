import os
import torch
import torch.optim as optim
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from configs.config import config
from utils.seed import seed_everything
from utils.logger import setup_logger
from datasets.unified_dataset import get_unified_dataloaders
from models.transformer_unet import SwinTransformerUNet, AdvancedSegLoss
from models.unet import LightweightUNet   # kept as fallback
from training.losses import CombinedSegLoss
from evaluation.metrics import compute_dice_score
from training.train_utils import AverageMeter, EarlyStopping

def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    losses = AverageMeter()
    dice_scores = AverageMeter()
    
    pbar = tqdm(loader, desc='Train')
    for batch in pbar:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device).float()  # BCE needs float32
        
        optimizer.zero_grad()
        
        with autocast('cuda', enabled=(device == 'cuda')):
            logits = model(images)
            loss = criterion(logits, masks)
            
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)  # unscale before clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # prevent explosion
        scaler.step(optimizer)
        scaler.update()
        
        # Calculate train dice score
        with torch.no_grad():
            dice = compute_dice_score(logits, masks)
        
        loss_val = loss.item()
        if not (loss_val != loss_val):  # skip NaN losses
            losses.update(loss_val, images.size(0))
        dice_scores.update(dice, images.size(0))
        
        pbar.set_postfix({"Loss": f"{losses.avg:.4f}", "Dice": f"{dice_scores.avg:.4f}"})
        
    return losses.avg, dice_scores.avg

def validate(model, loader, criterion, device):
    model.eval()
    losses = AverageMeter()
    dice_scores = AverageMeter()
    
    pbar = tqdm(loader, desc='Val')
    with torch.no_grad():
        for batch in pbar:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device).float()  # BCE needs float32
            
            with autocast('cuda', enabled=(device == 'cuda')):
                logits = model(images)
                loss = criterion(logits, masks)
                
            dice = compute_dice_score(logits, masks)
            
            loss_val = loss.item()
            if not (loss_val != loss_val):  # skip NaN
                losses.update(loss_val, images.size(0))
            dice_scores.update(dice, images.size(0))
            
            pbar.set_postfix({"Loss": f"{losses.avg:.4f}", "Dice": f"{dice_scores.avg:.4f}"})
            
    return losses.avg, dice_scores.avg

def main():
    seed_everything(config.SEED)
    config.setup_dirs()   # ← ensures outputs/weights/ etc. are created
    logger = setup_logger("train_seg", os.path.join(config.OUTPUT_DIR, "train_segmentation.log"))
    logger.info("Starting Lesion Segmentation Training")
    

    # ── Dataset (uses unified multi-dataset loader) ────────────────────── #
    # Use SEG_BATCH_SIZE (larger) instead of BATCH_SIZE:
    # Swin-Tiny (95M, 224px) is much lighter than EVA-02 Large — fits batch=8+ per GPU
    seg_batch = getattr(config, 'SEG_BATCH_SIZE', 8)
    train_loader, val_loader, test_loader, _ = get_unified_dataloaders(
        config.DATA_DIR,
        masks_dir=os.path.join(config.DATA_DIR, "masks"),
        batch_size=seg_batch,
    )

    # ── Segmentation model: Swin U-Net (or fallback) ─────────────────────── #
    if config.SEG_MODEL == 'swin_unet':
        model = SwinTransformerUNet(pretrained=True).to(config.DEVICE)
        logger.info("Using SwinTransformerUNet (2026 SOTA)")
    else:
        model = LightweightUNet(n_channels=3, n_classes=1).to(config.DEVICE)
        logger.info("Using LightweightUNet (legacy fallback)")

    # ── Multi-GPU: DataParallel only when per-GPU batch is ≥ 2 ───────────── #
    # DataParallel overhead > gain when each GPU sees only 1 image.
    # With SEG_BATCH_SIZE=8 and 2 GPUs: each GPU gets 4 images → worthwhile.
    n_gpus = torch.cuda.device_count()
    per_gpu_batch = seg_batch // max(n_gpus, 1)
    use_ddp = (n_gpus > 1) and (per_gpu_batch >= 2)
    if use_ddp:
        logger.info(f"Using {n_gpus} GPUs with DataParallel (batch={seg_batch}, {per_gpu_batch}/GPU)")
        model = torch.nn.DataParallel(model)
    else:
        if n_gpus > 1:
            logger.info(f"Skipping DataParallel: per-GPU batch {per_gpu_batch} < 2 (overhead > gain)")
        logger.info(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # Helper: unwrap DataParallel to get raw model for weight saving
    raw_model = model.module if isinstance(model, torch.nn.DataParallel) else model

    # Optimizer & Scheduler

    optimizer = optim.AdamW(model.parameters(), lr=config.SEG_LR, weight_decay=config.SEG_WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)
    
    # ── Loss: Use stable BCE (Dice/Tversky → NaN when masks are all-zero) ── #
    criterion = torch.nn.BCEWithLogitsLoss()

    # AMP Scaler
    scaler = GradScaler('cuda', enabled=(config.DEVICE == 'cuda'))
    
    # Early Stopping
    best_model_path = os.path.join(config.WEIGHTS_DIR, "best_unet.pth")
    early_stopping = EarlyStopping(patience=config.PATIENCE, verbose=True, path=best_model_path, trace_func=logger.info)

    # ── Resume checkpoint: continue from last saved epoch ──────────────── #
    resume_path = os.path.join(config.WEIGHTS_DIR, "resume_seg_checkpoint.pth")
    start_epoch = 1
    best_dice   = 0.0
    if os.path.exists(resume_path):
        logger.info(f"[RESUME] Found segmentation checkpoint → {resume_path}")
        ckpt = torch.load(resume_path, map_location=config.DEVICE)
        raw_model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_dice   = ckpt.get('best_dice', 0.0)
        early_stopping.best_score = ckpt.get('best_score', None)
        early_stopping.counter    = ckpt.get('es_counter', 0)
        logger.info(f"[RESUME] Resuming from Epoch {start_epoch}/{config.EPOCHS}  best_dice={best_dice:.4f}")
    else:
        logger.info("[RESUME] No checkpoint found — starting from Epoch 1")

    # Training Loop
    for epoch in range(start_epoch, config.EPOCHS + 1):
        logger.info(f"Epoch {epoch}/{config.EPOCHS}")
        
        train_loss, train_dice = train_one_epoch(model, train_loader, criterion, optimizer, scaler, config.DEVICE)
        val_loss, val_dice = validate(model, val_loader, criterion, config.DEVICE)
        
        scheduler.step()
        
        logger.info(f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}")
        
        early_stopping(val_dice, raw_model)

        # ── Save resume checkpoint after every epoch ────────────────────── #
        ckpt = {
            'epoch':      epoch,
            'model':      raw_model.state_dict(),
            'optimizer':  optimizer.state_dict(),
            'scheduler':  scheduler.state_dict(),
            'best_dice':  max(best_dice, val_dice),
            'best_score': early_stopping.best_score,
            'es_counter': early_stopping.counter,
        }
        torch.save(ckpt, resume_path)
        logger.info(f"[CHECKPOINT] Epoch {epoch} saved → resume from next run will start at epoch {epoch+1}")

        if early_stopping.early_stop:
            logger.info("Early stopping triggered")
            break

    logger.info("Segmentation Training Completed.")


if __name__ == "__main__":
    main()
