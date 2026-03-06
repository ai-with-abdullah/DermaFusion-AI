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
    train_loader, val_loader, test_loader, _ = get_unified_dataloaders(
        config.DATA_DIR, masks_dir=os.path.join(config.DATA_DIR, "masks")
    )

    # ── Segmentation model: Swin U-Net (or fallback) ─────────────────────── #
    if config.SEG_MODEL == 'swin_unet':
        model = SwinTransformerUNet(pretrained=True).to(config.DEVICE)
        logger.info("Using SwinTransformerUNet (2026 SOTA)")
    else:
        model = LightweightUNet(n_channels=3, n_classes=1).to(config.DEVICE)
        logger.info("Using LightweightUNet (legacy fallback)")

    # ── Multi-GPU: wrap with DataParallel if 2+ GPUs available ────────────── #
    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        logger.info(f"Using {n_gpus} GPUs with DataParallel: {[torch.cuda.get_device_name(i) for i in range(n_gpus)]}")
        model = torch.nn.DataParallel(model)
    else:
        logger.info(f"Using single GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

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
    
    # Training Loop
    best_dice = 0.0
    for epoch in range(1, config.EPOCHS + 1):
        logger.info(f"Epoch {epoch}/{config.EPOCHS}")
        
        train_loss, train_dice = train_one_epoch(model, train_loader, criterion, optimizer, scaler, config.DEVICE)
        val_loss, val_dice = validate(model, val_loader, criterion, config.DEVICE)
        
        scheduler.step()
        
        logger.info(f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}")
        
        # We want to maximize val_dice, so we use val_dice directly.
        early_stopping(val_dice, raw_model)
        if early_stopping.early_stop:
            logger.info("Early stopping triggered")
            break

    logger.info("Segmentation Training Completed.")

if __name__ == "__main__":
    main()
