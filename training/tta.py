"""
Test-Time Augmentation (TTA) — 2026 SOTA Inference
====================================================
TTA averages model predictions across N deterministically augmented views,
acting as a lightweight ensemble without retraining any model.

Typical gain on ISIC benchmarks: +0.5–2.0% AUC at zero training cost.

Usage:
    tta = TTAInference(model, unet, device, n_views=5)
    probs = tta.predict(images)          # (B, num_classes)
    probs = tta.predict_batch(loader)    # (N, num_classes) full dataset
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional
from tqdm import tqdm

from training.train_utils import apply_mask


# =========================================================================== #
#                           TTA TRANSFORMS (in-tensor)                        #
# =========================================================================== #

def _apply_tta_view(images: torch.Tensor, view_idx: int) -> torch.Tensor:
    """
    Apply a deterministic in-tensor transform for TTA.
    All transforms preserve label semantics for skin lesions.

    Extended from 5 → 8 views (ISIC 2024 winners used 10–16 views):
        0: Original (no-op)
        1: Horizontal flip
        2: Vertical flip
        3: 90° CCW rotation
        4: 180° rotation
        5: 270° CCW rotation   (NEW — completes full rotation set)
        6: Diagonal flip       (NEW — H-flip + V-flip combined)
        7: Brightness shift    (NEW — simulates illumination variation)
    """
    if view_idx == 0:
        return images
    elif view_idx == 1:
        return torch.flip(images, dims=[-1])              # H-flip
    elif view_idx == 2:
        return torch.flip(images, dims=[-2])              # V-flip
    elif view_idx == 3:
        return torch.rot90(images, k=1, dims=[-2, -1])   # CCW 90°
    elif view_idx == 4:
        return torch.rot90(images, k=2, dims=[-2, -1])   # 180°
    elif view_idx == 5:
        return torch.rot90(images, k=3, dims=[-2, -1])   # CW 90° (= CCW 270°)
    elif view_idx == 6:
        # Diagonal flip: H-flip + V-flip combined (transpose equivalent)
        return torch.flip(images, dims=[-1, -2])
    elif view_idx == 7:
        # Brightness shift: +10% intensity (simulates different dermoscope light)
        return torch.clamp(images * 1.10, -3.0, 3.0)
    else:
        return images


# =========================================================================== #
#                           TTA INFERENCE CLASS                                #
# =========================================================================== #

class TTAInference:
    """
    Test-Time Augmentation wrapper for DualBranchFusionClassifier.

    Runs N forward passes with different deterministic augmentations
    and averages the softmax probability outputs.

    Args:
        model:    DualBranchFusionClassifier (in eval mode)
        unet:     Segmentation model (for generating ConvNeXt branch input)
        device:   'cuda' | 'cpu'
        n_views:  Number of TTA views (1 = no TTA, 5 = full TTA)
    """

    def __init__(self, model, unet, device: str, n_views: int = 8):
        self.model   = model
        self.unet    = unet
        self.device  = device
        self.n_views = n_views

        self.model.eval()
        self.unet.eval()

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> np.ndarray:
        """
        Run TTA on a single batch.

        Args:
            images: (B, 3, H, W) normalized tensor on device

        Returns:
            probs: (B, num_classes) averaged softmax probabilities (numpy)
        """
        images = images.to(self.device)
        all_probs = []

        for view_idx in range(self.n_views):
            aug_images = _apply_tta_view(images, view_idx)

            # Generate segmentation mask for augmented image
            with torch.amp.autocast('cuda', enabled=(self.device == 'cuda')):
                mask_logits = self.unet(aug_images)
                images_seg  = apply_mask(aug_images, mask_logits)

                logits, _ = self.model(aug_images, images_seg)

            probs = torch.softmax(logits, dim=1).cpu().float().numpy()  # (B, C)
            all_probs.append(probs)

        # Average across all views → better calibrated probabilities
        avg_probs = np.mean(all_probs, axis=0)   # (B, C)
        return avg_probs

    @torch.no_grad()
    def predict_batch(
        self,
        loader,
        desc: str = 'TTA Inference',
    ):
        """
        Run TTA over an entire DataLoader.

        Returns:
            all_targets: (N,) ground truth labels
            all_probs:   (N, num_classes) TTA-averaged probabilities
            dataset_names: List[str] of dataset names per sample
        """
        all_targets      = []
        all_probs        = []
        all_dataset_names = []

        for batch in tqdm(loader, desc=desc):
            images = batch['image'].to(self.device)
            labels = batch['label']
            ds_names = batch.get('dataset_name', ['unknown'] * len(labels))

            probs = self.predict(images)    # (B, C) numpy

            all_targets.extend(labels.numpy().tolist())
            all_probs.extend(probs.tolist())
            all_dataset_names.extend(ds_names if isinstance(ds_names, list) else ds_names)

        return (
            np.array(all_targets),
            np.array(all_probs),
            all_dataset_names,
        )
