"""
Unified Multi-Dataset Skin Lesion Loader
=========================================
Supports simultaneous loading from:
  ✓ HAM10000    (~10,015 images, 7 classes)
  ✓ ISIC 2019  (~25,331 images, 8 classes → remapped to 7)
  ✓ ISIC 2020  (~33,126 images, binary mel/benign → remapped)
  ✓ ISIC 2024  (~401K images, binary → remapped, Kaggle SLICE-3D)
  ✓ PH2        (~200 images, 3 classes → remapped)

Each dataset that is NOT downloaded is silently skipped — the project still
runs with whatever datasets are available.

Label Remapping Strategy:
  All datasets are mapped to the HAM10000 7-class scheme:
    akiec, bcc, bkl, df, mel, nv, vasc
  Binary datasets (ISIC 2020/2024): target=1 → 'mel', target=0 → 'nv'
  PH2: melanoma=2 → 'mel', atypical=1 → 'bkl', benign=0 → 'nv'
  ISIC 2019: has 'SCC' (squamous cell) → mapped to 'akiec'
"""

import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from typing import Optional, Tuple, List, Dict

from configs.config import config
from datasets.augmentations import get_train_transforms, get_valid_transforms


# =========================================================================== #
#                   LABEL REMAPPING TABLES                                     #
# =========================================================================== #

# ISIC 2019 has 8 classes; map to our 7
ISIC2019_REMAP = {
    'MEL':   'mel',
    'NV':    'nv',
    'BCC':   'bcc',
    'AK':    'akiec',
    'BKL':   'bkl',
    'DF':    'df',
    'VASC':  'vasc',
    'SCC':   'akiec',  # Squamous Cell Carcinoma ≈ Actinic Keratosis
}

# PH2 numeric codes
PH2_REMAP = {
    0: 'nv',     # Common naevus
    1: 'bkl',    # Atypical naevus
    2: 'mel',    # Melanoma
}


# =========================================================================== #
#                   CORE DATASET CLASS                                         #
# =========================================================================== #

class SkinLesionRecord:
    """Lightweight struct holding per-sample metadata."""
    __slots__ = ('image_path', 'label_idx', 'patient_id', 'dataset_name', 'mask_path')

    def __init__(self, image_path, label_idx, patient_id, dataset_name, mask_path=None):
        self.image_path   = image_path
        self.label_idx    = label_idx
        self.patient_id   = patient_id
        self.dataset_name = dataset_name
        self.mask_path    = mask_path


class UnifiedSkinDataset(Dataset):
    """
    Unified skin lesion dataset across multiple ISIC/HAM collections.

    Args:
        records:    List of SkinLesionRecord objects
        transforms: Albumentations transform pipeline
    """

    CLASS_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(config.CLASSES)}

    def __init__(self, records: List[SkinLesionRecord], transforms=None):
        self.records    = records
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]

        # ── Load image ─────────────────────────────────────────────────────── #
        image = cv2.imread(rec.image_path) if os.path.exists(rec.image_path) else None
        if image is None:
            image = np.zeros(
                (config.IMAGE_SIZE, config.IMAGE_SIZE, 3), dtype=np.uint8
            )
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # ── Load mask (optional) ────────────────────────────────────────────── #
        mask = None
        if rec.mask_path and os.path.exists(rec.mask_path):
            mask = cv2.imread(rec.mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
        else:
            mask = (mask > 127).astype(np.float32)

        # ── Apply transforms ────────────────────────────────────────────────── #
        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented['image']
            mask  = augmented['mask'].unsqueeze(0)   # (1, H, W)

        return {
            'image':        image,
            'mask':         mask,
            'label':        rec.label_idx,
            'image_id':     os.path.splitext(os.path.basename(rec.image_path))[0],
            'dataset_name': rec.dataset_name,
        }


# =========================================================================== #
#                   DATASET PARSERS                                            #
# =========================================================================== #

def _parse_ham10000(data_dir: str, masks_dir: Optional[str] = None) -> List[SkinLesionRecord]:
    """Parse HAM10000. Supports two CSV formats:
    1. Standard: columns image_id + dx (text label)
    2. One-hot:  columns image + MEL, NV, BCC, AKIEC, BKL, DF, VASC (0.0/1.0)
    Searches multiple locations for the CSV (supports local and Kaggle layouts).
    """
    records: List[SkinLesionRecord] = []
    cls_map = UnifiedSkinDataset.CLASS_TO_IDX

    ONE_HOT_MAP = {
        'MEL': 'mel', 'NV': 'nv', 'BCC': 'bcc',
        'AKIEC': 'akiec', 'BKL': 'bkl', 'DF': 'df', 'VASC': 'vasc',
    }

    # HAM10000 lives inside data/ham10000/ subfolder on Kaggle
    ham_root = os.path.join(data_dir, 'ham10000')

    # Search candidate CSV paths (root dir first, then ham10000 subfolder)
    csv_candidates = [
        os.path.join(data_dir,  'HAM10000_metadata.csv'),
        os.path.join(ham_root,  'HAM10000_metadata.csv'),
        os.path.join(data_dir,  'ISIC_2024_metadata.csv'),
        os.path.join(data_dir,  'metadata.csv'),
        os.path.join(data_dir, 'isic_2019', 'ISIC_2019_Training_GroundTruth.csv'),
    ]

    df = None
    fmt = None  # 'standard' or 'onehot'
    for csv_path in csv_candidates:
        if not os.path.exists(csv_path):
            continue
        _df = pd.read_csv(csv_path)
        if {'image_id', 'dx'}.issubset(_df.columns):
            df, fmt = _df, 'standard'
            break
        if 'image' in _df.columns and any(c in _df.columns for c in ONE_HOT_MAP):
            df, fmt = _df, 'onehot'
            break

    if df is None:
        return records

    # Image directories — search flat dir, then part_1/part_2 subdirs (Kaggle layout)
    img_dirs = []
    for candidate in [
        os.path.join(data_dir, 'images'),
        os.path.join(ham_root, 'images'),
        os.path.join(ham_root, 'HAM10000_images_part_1'),
        os.path.join(ham_root, 'HAM10000_images_part_2'),
        os.path.join(ham_root, 'ham10000_images_part_1'),
        os.path.join(ham_root, 'ham10000_images_part_2'),
    ]:
        if os.path.exists(candidate):
            img_dirs.append(candidate)

    if not img_dirs:
        return records

    def _find_image(image_id: str) -> Optional[str]:
        for d in img_dirs:
            p = os.path.join(d, f"{image_id}.jpg")
            if os.path.exists(p):
                return p
        return None

    if fmt == 'standard':
        for _, row in df.iterrows():
            label_str = str(row['dx']).lower().strip()
            if label_str not in cls_map:
                continue
            img_path = _find_image(str(row['image_id']))
            if img_path is None:
                continue
            mask_path = None
            if masks_dir:
                for suffix in [f"{row['image_id']}_segmentation.png", f"{row['image_id']}.png"]:
                    candidate = os.path.join(masks_dir, suffix)
                    if os.path.exists(candidate):
                        mask_path = candidate
                        break
            records.append(SkinLesionRecord(
                image_path=img_path,
                label_idx=cls_map[label_str],
                patient_id=str(row.get('lesion_id', row['image_id'])),
                dataset_name='HAM10000',
                mask_path=mask_path,
            ))

    elif fmt == 'onehot':
        label_cols = [c for c in ONE_HOT_MAP if c in df.columns]
        for _, row in df.iterrows():
            label_str = None
            for col in label_cols:
                if row.get(col, 0) == 1.0:
                    label_str = ONE_HOT_MAP[col]
                    break
            if label_str is None or label_str not in cls_map:
                continue
            img_id   = str(row['image'])
            img_path = _find_image(img_id)
            if img_path is None:
                continue
            records.append(SkinLesionRecord(
                image_path=img_path,
                label_idx=cls_map[label_str],
                patient_id=img_id,
                dataset_name='HAM10000',
            ))

    print(f"  [HAM10000] Loaded {len(records)} records (fmt={fmt}).")
    return records



def _parse_isic2019(data_dir: str) -> List[SkinLesionRecord]:
    """
    Parse ISIC 2019 dataset. Supports two layouts:
    1. Flat layout:      data/isic_2019/ISIC_2019_Training_Input/{image}.jpg
    2. Per-class layout: data/isic_2019/MEL/{image}.jpg  (Kaggle dataset)

    The GroundTruth CSV is used in both cases for class labels.
    Also maps AK/SCC → akiec to handle the 8-class Kaggle variant.
    """
    records: List[SkinLesionRecord] = []
    cls_map = UnifiedSkinDataset.CLASS_TO_IDX
    base = os.path.join(data_dir, 'isic_2019')

    # Extended remap: handles AK and SCC present in the Kaggle per-class layout
    REMAP = {**ISIC2019_REMAP, 'AK': 'akiec', 'SCC': 'akiec'}

    csv_path = os.path.join(base, 'ISIC_2019_Training_GroundTruth.csv')
    if not os.path.exists(csv_path):
        return records

    df = pd.read_csv(csv_path)

    # Detect layout: per-class folders (MEL/, NV/, ...) vs flat ISIC_2019_Training_Input/
    class_folders = {
        k: os.path.join(base, k)
        for k in list(ISIC2019_REMAP.keys()) + ['AK', 'SCC']
        if os.path.isdir(os.path.join(base, k))
    }
    flat_dir = os.path.join(base, 'ISIC_2019_Training_Input')
    use_per_class = len(class_folders) > 0

    # Build all possible image search dirs — handle Kaggle's andrewmvd/isic-2019 layout
    # which may nest images under subfolders. Collect ALL subdirs inside base.
    all_search_dirs = []
    if os.path.exists(flat_dir):
        all_search_dirs.append(flat_dir)
    if use_per_class:
        all_search_dirs += list(class_folders.values())
    # Also scan any other subdirectory directly inside base (Kaggle variant layouts)
    if os.path.exists(base):
        for sub in os.listdir(base):
            sub_path = os.path.join(base, sub)
            if os.path.isdir(sub_path) and sub_path not in all_search_dirs:
                all_search_dirs.append(sub_path)

    if not all_search_dirs:
        return records   # no image source found

    label_cols = [c for c in REMAP.keys() if c in df.columns]
    missing_count = 0

    def _find_img(image_id: str, label_col: str) -> Optional[str]:
        # Priority: per-class folder → flat dir → all other subdirs
        search_dirs = []
        if use_per_class and label_col in class_folders:
            search_dirs.append(class_folders[label_col])
            search_dirs += [d for k, d in class_folders.items() if k != label_col]
        search_dirs += [d for d in all_search_dirs if d not in search_dirs]
        for d in search_dirs:
            for ext in ['.jpg', '.JPG', '.jpeg', '.png']:
                p = os.path.join(d, f"{image_id}{ext}")
                if os.path.exists(p):
                    return p
        return None

    for _, row in df.iterrows():
        label_col = None
        label_str = None
        for col in label_cols:
            if row.get(col, 0) == 1.0:
                label_col = col
                label_str = REMAP.get(col)
                break
        if label_str is None or label_str not in cls_map:
            continue

        img_path = _find_img(str(row['image']), label_col or '')
        if img_path is None:
            missing_count += 1
            continue

        records.append(SkinLesionRecord(
            image_path=img_path,
            label_idx=cls_map[label_str],
            patient_id=str(row['image']),
            dataset_name='ISIC2019',
        ))

    if missing_count > 0:
        print(f"  [ISIC 2019] Skipped {missing_count} rows with missing image files.")
    print(f"  [ISIC 2019] Loaded {len(records)} records (per-class={use_per_class}).")
    return records


def _parse_isic2020(data_dir: str) -> List[SkinLesionRecord]:
    """
    Parse ISIC 2020 dataset.
    Expected layout:
      data/isic_2020/train/          (images)
      data/isic_2020/train.csv       (columns: image_name, patient_id, target)

    Strategy (FIXED — re-enabled positive-only):
    ─────────────────────────────────────────────
    ISIC 2020 target=0 means 'not melanoma', NOT specifically 'nv'.
    It includes a mix of BCC, BKL, AK, DF, VASC, NV — mapping all of
    these to 'nv' would corrupt the NV/MEL and every other decision
    boundary with ~28K mislabeled samples.

    SAFE approach (now active):
    • target=1 → 'mel'  (confirmed melanoma — ~1,755 clean positives)
    • target=0 → SKIP   (heterogeneous negatives, label noise)

    This recovers ~1,755 high-quality confirmed melanoma cases that
    were previously being discarded entirely. The benign class is not
    augmented from this dataset to avoid introducing noise.
    """
    records: List[SkinLesionRecord] = []
    cls_map = UnifiedSkinDataset.CLASS_TO_IDX
    base = os.path.join(data_dir, 'isic_2020')

    csv_path = os.path.join(base, 'train.csv')
    # KAGGLE FIX: SIIM-ISIC competition stores JPEG images in jpeg/train/, NOT train/
    # train/ contains DICOM files. jpeg/train/ has the actual .jpg images.
    img_dir = os.path.join(base, 'jpeg', 'train')
    if not os.path.exists(img_dir):
        img_dir = os.path.join(base, 'train')  # fallback for non-Kaggle layout
    if not os.path.exists(csv_path) or not os.path.exists(img_dir):
        return records

    df = pd.read_csv(csv_path)
    if 'target' not in df.columns or 'image_name' not in df.columns:
        return records

    for _, row in df.iterrows():
        if int(row['target']) != 1:
            continue  # Skip target=0 — heterogeneous negatives (label noise)
        label_str = 'mel'

        img_path = None
        for ext in ['.jpg', '.png', '.jpeg']:
            candidate = os.path.join(img_dir, f"{row['image_name']}{ext}")
            if os.path.exists(candidate):
                img_path = candidate
                break
        if img_path is None:
            continue

        records.append(SkinLesionRecord(
            image_path=img_path,
            label_idx=cls_map[label_str],
            patient_id=str(row.get('patient_id', row['image_name'])),
            dataset_name='ISIC2020',
        ))

    print(f"  [ISIC 2020] Loaded {len(records)} confirmed mel-positive records "
          f"(target=0 negatives excluded — label noise).")
    return records


def _parse_isic2024(data_dir: str) -> List[SkinLesionRecord]:
    """
    Parse ISIC 2024 (SLICE-3D Kaggle challenge) dataset.
    Expected layout:
      data/isic_2024/train-image/image/     (images as .jpg)
      data/isic_2024/train-metadata.csv     (columns: isic_id, target)

    FIXED (Fix #4): Downsampling is NO LONGER done here. Loading all records
    and returning them so that the split happens on the full distribution.
    Downsampling of ISIC2024 negatives is applied AFTER the train/val/test
    split, only on train_records, inside get_unified_dataloaders().
    This prevents val/test being drawn from a pre-balanced pool.

    FIXED (Fix #3): If the metadata CSV contains a 'patient_id' column, use
    it as the patient identifier so GroupShuffleSplit works correctly.
    """
    records: List[SkinLesionRecord] = []
    cls_map = UnifiedSkinDataset.CLASS_TO_IDX
    base = os.path.join(data_dir, 'isic_2024')

    csv_path = os.path.join(base, 'train-metadata.csv')
    img_dir  = os.path.join(base, 'train-image', 'image')
    if not os.path.exists(csv_path):
        csv_path = os.path.join(base, 'train.csv')
        img_dir  = os.path.join(base, 'images')
    if not os.path.exists(csv_path):
        return records

    df = pd.read_csv(csv_path)
    id_col = 'isic_id' if 'isic_id' in df.columns else 'image_name'
    if id_col not in df.columns or 'target' not in df.columns:
        return records

    # Use real patient_id if available (prevents same patient in train+test)
    has_patient_col = 'patient_id' in df.columns

    for _, row in df.iterrows():
        label_str = 'mel' if int(row['target']) == 1 else 'nv'
        img_path = os.path.join(img_dir, f"{row[id_col]}.jpg")

        pid = str(row['patient_id']) if has_patient_col else str(row[id_col])
        records.append(SkinLesionRecord(
            image_path=img_path,
            label_idx=cls_map[label_str],
            patient_id=pid,
            dataset_name='ISIC2024',
        ))

    n_pos = sum(1 for r in records if r.label_idx == cls_map['mel'])
    if not has_patient_col:
        print("  [ISIC 2024] WARNING: no 'patient_id' column found — using isic_id as patient ID."
              " Patient-level split not guaranteed.")
    print(f"  [ISIC 2024] Loaded {len(records)} records ({n_pos} mel positives). "
          f"Downsampling applied after split in get_unified_dataloaders().")
    return records


def _parse_ph2(data_dir: str) -> List[SkinLesionRecord]:
    """
    Parse PH2 dataset.
    Expected layout:
      data/ph2/PH2Dataset/         (each case is a subfolder: IMDxxx/)
        IMD001/
          IMD001_Dermoscopic_Image/IMD001.bmp   (image)
          IMD001_lesion/IMD001_lesion.bmp        (mask)
      data/ph2/PH2_dataset.txt    (tab-separated, column 'clinical_diagnosis')
    """
    records: List[SkinLesionRecord] = []
    cls_map = UnifiedSkinDataset.CLASS_TO_IDX
    base = os.path.join(data_dir, 'ph2')

    txt_path = os.path.join(base, 'PH2_dataset.txt')
    img_base = os.path.join(base, 'PH2Dataset')
    if not os.path.exists(txt_path) or not os.path.exists(img_base):
        return records

    try:
        df = pd.read_csv(txt_path, sep='\t', comment='#')
        # The numeric diagnosis column varies by version; try common names
        diag_col = None
        for col in ['Classification', 'clinical_diagnosis', 'Diagnosis', 'diag']:
            if col in df.columns:
                diag_col = col
                break
        if diag_col is None:
            return records

        name_col = None
        for col in ['Name', 'image_name', 'Image Name', 'name']:
            if col in df.columns:
                name_col = col
                break
        if name_col is None:
            return records

        for _, row in df.iterrows():
            img_name = str(row[name_col]).strip()
            diag     = int(row[diag_col])
            label_str = PH2_REMAP.get(diag)
            if label_str is None or label_str not in cls_map:
                continue

            img_path  = os.path.join(img_base, img_name,
                                     f"{img_name}_Dermoscopic_Image", f"{img_name}.bmp")
            mask_path = os.path.join(img_base, img_name,
                                     f"{img_name}_lesion", f"{img_name}_lesion.bmp")

            records.append(SkinLesionRecord(
                image_path=img_path,
                label_idx=cls_map[label_str],
                patient_id=img_name,
                dataset_name='PH2',
                mask_path=mask_path if os.path.exists(mask_path) else None,
            ))
    except Exception as e:
        print(f"  [PH2] Parse warning: {e}")
        return records

    print(f"  [PH2] Loaded {len(records)} records.")
    return records


# =========================================================================== #
#                   SPLIT & DATALOADER BUILDERS                                #
# =========================================================================== #

def _patient_aware_split(
    records: List[SkinLesionRecord],
    train_ratio: float = 0.70,
    val_ratio: float   = 0.15,
    seed: int          = config.SEED,
) -> Tuple[List[SkinLesionRecord], List[SkinLesionRecord], List[SkinLesionRecord]]:
    """
    Splits records into train/val/test ensuring no patient appears in two splits.
    Falls back to stratified split if patient IDs are all unique (ISIC 2020/2024).
    """
    patient_ids = [r.patient_id for r in records]
    labels      = [r.label_idx  for r in records]

    unique_patients = list(set(patient_ids))
    indices = list(range(len(records)))

    # Patient-grouped split
    gss1 = GroupShuffleSplit(n_splits=1, train_size=train_ratio, random_state=seed)
    train_idx, temp_idx = next(gss1.split(indices, groups=patient_ids))

    temp_records = [records[i] for i in temp_idx]
    temp_pids    = [patient_ids[i] for i in temp_idx]

    val_frac = val_ratio / (1.0 - train_ratio)
    gss2 = GroupShuffleSplit(n_splits=1, train_size=val_frac, random_state=seed)
    val_idx_local, test_idx_local = next(gss2.split(
        list(range(len(temp_records))), groups=temp_pids
    ))

    train_records = [records[i]            for i in train_idx]
    val_records   = [temp_records[i]       for i in val_idx_local]
    test_records  = [temp_records[i]       for i in test_idx_local]

    return train_records, val_records, test_records


def get_weighted_sampler(records: List[SkinLesionRecord]) -> WeightedRandomSampler:
    """
    Creates a WeightedRandomSampler to balance class frequencies during training.
    Replaces simple shuffle to handle extreme imbalance in multi-dataset training.

    Bug #3 fix: Removed the inverted a_max clip that was halving rare-class weights.
    Previously: np.clip(a_max=max_w * 0.5) cut the rarest class weight IN HALF.
    The 2× melanoma boost in get_class_weights_from_records() was also halved.

    Current strategy:
      - Base weight: inverse class frequency (standard)
      - Mel gets extra weight via get_class_weights_from_records (3× boost)
      - No artificial cap — let the sampler do its job
      - Safety floor: clip very small weights up to 10% of mean to prevent
        near-zero sampling of any class (avoids completely ignoring small classes)
    """
    labels = [r.label_idx for r in records]
    class_weights = get_class_weights_from_records(records).numpy()
    sample_weights = np.array([class_weights[l] for l in labels], dtype=np.float32)

    # Safety floor: ensure no class gets near-zero sampling probability
    # (different from the old a_max clip which wrongly capped rare classes down)
    mean_w = sample_weights.mean()
    sample_weights = np.clip(sample_weights, a_min=mean_w * 0.1, a_max=None)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler


def get_unified_dataloaders(data_dir: str, masks_dir: Optional[str] = None):
    """
    Main entry point. Loads all available datasets, merges, splits, and
    returns (train_loader, val_loader, test_loader, train_records).

    Datasets not found on disk are silently skipped.
    """
    print("\n" + "="*70)
    print("[UnifiedDataset] Scanning available datasets...")

    all_records: List[SkinLesionRecord] = []

    all_records += _parse_ham10000(data_dir, masks_dir)
    all_records += _parse_isic2019(data_dir)
    all_records += _parse_isic2020(data_dir)
    all_records += _parse_isic2024(data_dir)
    all_records += _parse_ph2(data_dir)

    if len(all_records) == 0:
        print("  [UnifiedDataset] No records found — generating dummy data.")
        all_records = _make_dummy_records(data_dir)

    # Report
    from collections import Counter
    label_dist = Counter(r.label_idx for r in all_records)
    dataset_dist = Counter(r.dataset_name for r in all_records)
    print(f"\n[UnifiedDataset] Total records: {len(all_records)}")
    print(f"  Dataset breakdown: {dict(dataset_dist)}")
    cls_names = config.CLASSES
    print(f"  Class distribution: {{ {', '.join(f'{cls_names[k]}: {v}' for k, v in sorted(label_dist.items()))} }}")
    print("="*70 + "\n")

    # Split
    train_records, val_records, test_records = _patient_aware_split(all_records)

    # FIXED (Fix #4): Balance ISIC2024 negatives AFTER the split (not before).
    # Previously _parse_isic2024() downsampled before splitting, so val/test were
    # drawn from a pre-balanced pool that doesn't represent real class distribution.
    train_records = _downsample_isic2024_train(train_records)

    # Build datasets
    train_ds = UnifiedSkinDataset(train_records, transforms=get_train_transforms())
    val_ds   = UnifiedSkinDataset(val_records,   transforms=get_valid_transforms())
    test_ds  = UnifiedSkinDataset(test_records,  transforms=get_valid_transforms())

    # Weighted sampler for train to handle class imbalance
    sampler = get_weighted_sampler(train_records)

    train_loader = DataLoader(
        train_ds, batch_size=config.BATCH_SIZE,
        sampler=sampler,
        num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True
    )

    return train_loader, val_loader, test_loader, train_records


def _downsample_isic2024_train(
    train_records: List[SkinLesionRecord],
    neg_to_pos_ratio: int = 3,
    seed: int = config.SEED,
) -> List[SkinLesionRecord]:
    """
    Downsamples ISIC2024 negative (nv) records in train split to neg_to_pos_ratio × positives.

    FIXED (Fix #4): Downsampling is applied ONLY to train_records AFTER the split,
    so val and test sets reflect the real ISIC2024 class distribution, not a
    pre-balanced sample that would bias evaluation metrics.
    """
    from configs.config import config as _cfg
    mel_idx = _cfg.CLASSES.index('mel') if 'mel' in _cfg.CLASSES else 4
    nv_idx  = _cfg.CLASSES.index('nv')  if 'nv'  in _cfg.CLASSES else 5

    isic24_pos = [r for r in train_records if r.dataset_name == 'ISIC2024' and r.label_idx == mel_idx]
    isic24_neg = [r for r in train_records if r.dataset_name == 'ISIC2024' and r.label_idx == nv_idx]
    other      = [r for r in train_records
                  if r.dataset_name != 'ISIC2024'
                  or (r.label_idx != mel_idx and r.label_idx != nv_idx)]

    if not isic24_pos or not isic24_neg:
        return train_records   # Nothing to downsample

    target_neg = min(len(isic24_pos) * neg_to_pos_ratio, len(isic24_neg))
    rng = np.random.default_rng(seed)
    neg_keep_idx = rng.choice(len(isic24_neg), size=target_neg, replace=False)
    isic24_neg_sampled = [isic24_neg[i] for i in sorted(neg_keep_idx)]

    result = other + isic24_pos + isic24_neg_sampled
    print(f"  [ISIC2024 Train Balance] pos={len(isic24_pos)}, "
          f"neg={len(isic24_neg)} → {len(isic24_neg_sampled)} "
          f"({neg_to_pos_ratio}:1 neg:pos ratio)")
    return result


# =========================================================================== #
#                   DUMMY DATA (no datasets downloaded)                        #
# =========================================================================== #

def _make_dummy_records(data_dir: str) -> List[SkinLesionRecord]:
    """Creates 120 dummy records so the pipeline runs without real data."""
    dummy_img_dir = os.path.join(data_dir, 'images')
    os.makedirs(dummy_img_dir, exist_ok=True)
    cls_map = UnifiedSkinDataset.CLASS_TO_IDX
    records = []

    for i in range(120):
        img_id = f'DUMMY_{i:05d}'
        img_path = os.path.join(dummy_img_dir, f'{img_id}.jpg')
        # Create a tiny black JPEG if missing
        if not os.path.exists(img_path):
            import cv2 as _cv2
            _cv2.imwrite(img_path, np.zeros((64, 64, 3), dtype=np.uint8))
        label_idx = i % config.NUM_CLASSES
        records.append(SkinLesionRecord(
            image_path=img_path,
            label_idx=label_idx,
            patient_id=f'patient_{i // 2}',
            dataset_name='DUMMY',
        ))
    return records


# =========================================================================== #
#              CLASS WEIGHTS (compatible with old code)                        #
# =========================================================================== #

def get_class_weights_from_records(records: List[SkinLesionRecord]) -> torch.Tensor:
    """
    Returns class weights as a tensor for use with CombinedClassLoss.

    Strategy:
      - Sqrt inverse-frequency dampens extremes: nv(13K) vs vasc(284) → 6.8:1 not 47:1
      - 2× melanoma boost: mel is most dangerous, highest clinical FN cost.

    FIXED: Reduced 3× → 2× to avoid triple-stacking with CombinedClassLoss gamma and
    the now-removed AsymmetricMelFocalLoss fn_weight. The previous 3× class weight ×
    gamma=3.0 × fn_weight=3.0 produced up to 27× signal for mel FNs — causing mel
    over-prediction and collapse of VASC/DF classes.
    """
    labels = [r.label_idx for r in records]
    counts = np.bincount(labels, minlength=config.NUM_CLASSES).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    total  = counts.sum()

    weights = np.sqrt(total / (config.NUM_CLASSES * counts))

    # 2× boost for melanoma — balanced single-layer priority (not triple-stacked)
    if 'mel' in config.CLASSES:
        mel_idx = config.CLASSES.index('mel')
        weights[mel_idx] *= 2.0

    return torch.FloatTensor(weights)
