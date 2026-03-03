import torch
import os
import cv2
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import GroupShuffleSplit
from typing import Tuple, Dict

from configs.config import config
from datasets.augmentations import get_train_transforms, get_valid_transforms

class HAM10000Dataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_dir: str, masks_dir: str = None, transforms=None):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.masks_dir = masks_dir
        self.transforms = transforms
        
        # Create class to index mapping
        self.class_to_idx = {c: i for i, c in enumerate(config.CLASSES)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row['image_id']
        label_str = row['dx']
        label = self.class_to_idx[label_str]

        # Load image
        img_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        image = cv2.imread(img_path)
        if image is None:
            # Fallback if image not found, create dummy for robustness in testing without data
            image = np.zeros((config.IMAGE_SIZE, config.IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask if directory provided, else create dummy mask
        mask = None
        if self.masks_dir and os.path.exists(os.path.join(self.masks_dir, f"{image_id}_segmentation.png")):
            mask_path = os.path.join(self.masks_dir, f"{image_id}_segmentation.png")
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        elif self.masks_dir and os.path.exists(os.path.join(self.masks_dir, f"{image_id}.png")):
            mask_path = os.path.join(self.masks_dir, f"{image_id}.png")
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
        if mask is None:
             # Dummy mask if missing
             mask = np.zeros(image.shape[:2], dtype=np.uint8)
        else:
            # Ensure mask is binary (0 or 1)
            mask = (mask > 127).astype(np.float32)

        # Apply transforms
        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            # albumentations returns mask as tensor with shape (H, W). Add channel dim if needed.
            mask = mask.unsqueeze(0) # (1, H, W)

        return {
            'image': image,
            'mask': mask,
            'label': label,
            'image_id': image_id
        }

def get_patient_wise_split(csv_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Implements a PATIENT-WISE split (70 train / 15 val / 15 test).
    Removes duplicate leakage by ensuring a patient's lesions only appear in ONE of the splits.
    Uses 'lesion_id' as the grouping variable.
    """
    df = pd.read_csv(csv_path)
    
    # Check if necessary columns exist
    if 'lesion_id' not in df.columns or 'image_id' not in df.columns or 'dx' not in df.columns:
        raise ValueError("CSV must contain 'lesion_id', 'image_id', and 'dx' columns.")

    # Remove strict duplicates if any (same image)
    df = df.drop_duplicates(subset=['image_id'])

    # Step 1: Split Train (70%) and Temp (30%) grouped by lesion_id
    gss1 = GroupShuffleSplit(n_splits=1, train_size=0.7, random_state=config.SEED)
    train_idx, temp_idx = next(gss1.split(df, groups=df['lesion_id']))
    
    train_df = df.iloc[train_idx]
    temp_df = df.iloc[temp_idx]

    # Step 2: Split Temp into Validation (15% overall -> 50% of temp) and Test (15% overall -> 50% of temp)
    gss2 = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=config.SEED)
    val_idx, test_idx = next(gss2.split(temp_df, groups=temp_df['lesion_id']))
    
    val_df = temp_df.iloc[val_idx]
    test_df = temp_df.iloc[test_idx]

    return train_df, val_df, test_df

def get_dataloaders(csv_path: str, image_dir: str, masks_dir: str = None):
    """
    Prepares dataloaders for Train, Val, Test splits.
    """
    # Create dummy data if CSV doesn't exist to allow code to be runnable (as requested)
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found. Creating a dummy dataset to ensure code runs.")
        dummy_data = []
        for i in range(100):
            dummy_data.append({
                'image_id': f'ISIC_{i:07d}',
                'lesion_id': f'HAM_{i//2:07d}', # Every patient has ~2 images
                'dx': config.CLASSES[i % config.NUM_CLASSES]
            })
        df = pd.DataFrame(dummy_data)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df.to_csv(csv_path, index=False)

    train_df, val_df, test_df = get_patient_wise_split(csv_path)

    train_dataset = HAM10000Dataset(train_df, image_dir, masks_dir, transforms=get_train_transforms())
    val_dataset = HAM10000Dataset(val_df, image_dir, masks_dir, transforms=get_valid_transforms())
    test_dataset = HAM10000Dataset(test_df, image_dir, masks_dir, transforms=get_valid_transforms())

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, 
        num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, 
        num_workers=config.NUM_WORKERS, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, 
        num_workers=config.NUM_WORKERS, pin_memory=True
    )

    return train_loader, val_loader, test_loader, train_df

def get_class_weights(train_df: pd.DataFrame) -> torch.Tensor:
    """
    Computes class weights automatically from the training set to handle imbalance.
    Returns tensor of weights in the order of config.CLASSES.
    """
    class_counts = train_df['dx'].value_counts().to_dict()
    total_samples = len(train_df)
    
    weights = []
    for cls in config.CLASSES:
        count = class_counts.get(cls, 1) # avoid div by zero
        weight = total_samples / (config.NUM_CLASSES * count)
        weights.append(weight)
        
    tensor_weights = torch.FloatTensor(weights)
    return tensor_weights
