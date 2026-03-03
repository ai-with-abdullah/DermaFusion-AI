"""
Advanced Augmentation Pipeline — 2026 SOTA
============================================
Training pipeline:
  RandomResizedCrop → HorizontalFlip → VerticalFlip → Rotate
  → CLAHE → GridDistortion → ElasticTransform
  → ColorJitter → GaussNoise → CoarseDropout (Cutout)
  → Normalize → ToTensor

Validation / Test pipeline:
  Resize → CenterCrop → Normalize → ToTensor

TTA (Test-Time Augmentation) returns a list of 5 transforms.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
from configs.config import config
from typing import List
import numpy as np
import cv2


# =========================================================================== #
#                     HAIR ARTIFACT AUGMENTATION                               #
# =========================================================================== #

class HairAugmentation(A.ImageOnlyTransform):
    """
    Simulates dermoscopy hair artifacts — random thin dark lines overlaid
    on the image. Hair is the #1 artifact causing AI misclassification.

    Studies show models trained WITH hair augmentation generalize significantly
    better across different imaging devices and patient skin tones.
    """
    def apply(self, img, **params):
        img = img.copy()
        num_hairs = np.random.randint(5, 25)
        h, w = img.shape[:2]
        for _ in range(num_hairs):
            x1 = np.random.randint(0, w)
            y1 = np.random.randint(0, h)
            angle = np.random.uniform(0, 2 * np.pi)
            length = np.random.randint(30, 120)
            x2 = int(x1 + length * np.cos(angle))
            y2 = int(y1 + length * np.sin(angle))
            # Dark brown/black hair color
            color = (
                np.random.randint(10, 50),
                np.random.randint(5, 35),
                np.random.randint(0, 20),
            )
            thickness = np.random.randint(1, 3)
            cv2.line(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
        return img

    def get_transform_init_args_names(self):
        return ()


# =========================================================================== #
#                     TRAINING TRANSFORMS                                      #
# =========================================================================== #

def get_train_transforms() -> A.Compose:
    """
    Full 2026 SOTA augmentation stack:
      - Geometric: crop, flip, rotate, grid distortion, elastic
      - Color:     CLAHE, color jitter, Gaussian noise
      - Cutout:    CoarseDropout (simulates occlusion / artifacts)
      - Hair simulation: thin random lines (hair artifact augmentation)
    """
    return A.Compose([
        # Geometric
        A.RandomResizedCrop(
            size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
            scale=(0.5, 1.0),   # ↑ Widened from (0.75,1.0) — more diverse views
            ratio=(0.9, 1.1),
            p=1.0,
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Rotate(limit=30, border_mode=0, p=0.6),
        A.Transpose(p=0.3),

        # Dermoscopy-specific distortions
        A.OneOf([
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=1.0),
            A.OpticalDistortion(distort_limit=0.3, p=1.0),
        ], p=0.4),

        # Contrast & Color — simulate different dermoscopes
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
        A.OneOf([
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=1.0),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=40, val_shift_limit=20, p=1.0),
            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1.0),
        ], p=0.7),

        # Blur / noise — simulate acquisition variation
        A.OneOf([
            A.GaussNoise(std_range=(0.02, 0.1), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
            A.MotionBlur(blur_limit=5, p=1.0),
        ], p=0.3),

        # Cutout / occlusion — simulate hair artifacts and patches
        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(int(config.IMAGE_SIZE * 0.04), int(config.IMAGE_SIZE * 0.08)),
            hole_width_range=(int(config.IMAGE_SIZE * 0.04), int(config.IMAGE_SIZE * 0.08)),
            fill=128,   # Grey fill — more realistic than black for skin occlusion
            p=0.4,
        ),

        # Hair artifact simulation — critical for dermoscopy generalization
        HairAugmentation(p=0.4),

        # Normalize & convert
        A.Normalize(mean=config.MEAN, std=config.STD),
        ToTensorV2(),
    ], additional_targets={'mask': 'mask'})


# =========================================================================== #
#                     VALIDATION TRANSFORMS                                    #
# =========================================================================== #

def get_valid_transforms() -> A.Compose:
    """Deterministic validation / test transforms."""
    return A.Compose([
        A.Resize(
            height=int(config.IMAGE_SIZE * 1.14),
            width=int(config.IMAGE_SIZE * 1.14),
        ),
        A.CenterCrop(height=config.IMAGE_SIZE, width=config.IMAGE_SIZE),
        A.Normalize(mean=config.MEAN, std=config.STD),
        ToTensorV2(),
    ], additional_targets={'mask': 'mask'})


# =========================================================================== #
#                     TTA TRANSFORMS                                           #
# =========================================================================== #

def get_tta_transforms() -> List[A.Compose]:
    """
    Returns 5 deterministic transforms used for Test-Time Augmentation (TTA).
    Final prediction = mean of softmax across all 5 views.
    """
    base_transforms = [
        A.Normalize(mean=config.MEAN, std=config.STD),
        ToTensorV2(),
    ]
    crop_resize = [
        A.Resize(int(config.IMAGE_SIZE * 1.14), int(config.IMAGE_SIZE * 1.14)),
        A.CenterCrop(config.IMAGE_SIZE, config.IMAGE_SIZE),
    ]

    augmentations = [
        [],                                              # 1. Original (center crop)
        [A.HorizontalFlip(p=1.0)],                      # 2. Horizontal flip
        [A.VerticalFlip(p=1.0)],                         # 3. Vertical flip
        [A.Rotate(limit=(90, 90), p=1.0)],               # 4. 90° rotation
        [A.Rotate(limit=(180, 180), p=1.0)],             # 5. 180° rotation
    ]

    return [
        A.Compose(
            crop_resize + aug + base_transforms,
            additional_targets={'mask': 'mask'}
        )
        for aug in augmentations
    ]
